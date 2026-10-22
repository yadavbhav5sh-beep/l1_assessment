from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CATALOG = "tpch_dev"

BRONZE_TABLE = f"{CATALOG}.bronze.supplier"

SUPPLIER_TABLE = f"{CATALOG}.silver.supplier"
QUARANTINE_TABLE = f"{CATALOG}.silver.supplier_quarantine"

# Referential integrity target
NATION_TABLE = f"{CATALOG}.silver.nation"

# ─────────────────────────────────────────────
# 1. Standardize suppliers (S1-S6)
# ─────────────────────────────────────────────
def standardize_suppliers(df: DataFrame) -> DataFrame:
    """
    Standardization happens before deduplication and validation.

    S1: supplier key trim; empty -> NULL
    S2: supplier name trim; empty -> NULL
    S3: address trim; empty -> NULL
    S4: nation key trim; empty -> NULL
    S5: phone / comment trim; empty -> NULL
    S6: account balance cast to decimal(18,2)
    """

    # Preserve the complete original Bronze row as JSON.
    df = df.withColumn(
        "raw_record",
        F.to_json(F.struct(*[F.col(c) for c in df.columns]))
    )

    # S1: keep the raw text key for quarantine traceability.
    df = df.withColumn(
        "supplier_key_raw",
        F.when(
            F.trim(F.col("s_suppkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("s_suppkey").cast("string"))
        )
    )

    # S1b: strict integer parse - anything else becomes NULL
    df = df.withColumn(
        "supplier_key",
        F.when(
            F.col("supplier_key_raw").rlike("^-?\\d+$"),
            F.col("supplier_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S2
    df = df.withColumn(
        "supplier_name",
        F.when(
            F.trim(F.col("s_name").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("s_name").cast("string"))
        )
    )

    # S3
    df = df.withColumn(
        "supplier_address",
        F.when(
            F.trim(F.col("s_address").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("s_address").cast("string"))
        )
    )

    # S4: nation key - raw text + strict integer parse
    df = df.withColumn(
        "nation_key_raw",
        F.when(
            F.trim(F.col("s_nationkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("s_nationkey").cast("string"))
        )
    )

    df = df.withColumn(
        "nation_key",
        F.when(
            F.col("nation_key_raw").rlike("^-?\\d+$"),
            F.col("nation_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S5
    df = df.withColumn(
        "supplier_phone",
        F.when(
            F.trim(F.col("s_phone").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("s_phone").cast("string"))
        )
    )

    df = df.withColumn(
        "supplier_comment",
        F.when(
            F.trim(F.col("s_comment").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("s_comment").cast("string"))
        )
    )

    # S6: account balance - cast to decimal(18,2), invalid -> NULL
    df = df.withColumn(
        "supplier_acctbal",
        F.col("s_acctbal").cast("decimal(18,2)")
    )

    # Keep _ingestion_at until deduplication has completed.
    return df.drop("_source_file")


# ─────────────────────────────────────────────
# 2. Latest version per supplier key (C2)
# ─────────────────────────────────────────────
def latest_supplier_versions(df: DataFrame) -> DataFrame:
    """
    Validate only the most recently ingested version of each key.

    _ingestion_at is used as the ordering key.
    """

    window_spec = (
        Window
        .partitionBy("supplier_key_raw")
        .orderBy(F.col("_ingestion_at").desc_nulls_last())
    )

    return (
        df
        .withColumn("_rn", F.row_number().over(window_spec))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

# ─────────────────────────────────────────────
# 3. Data quality rules (R1-R3)
# ─────────────────────────────────────────────
def validate_suppliers(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Reject a record if ANY rule fails. Every failed rule is
    collected into rejection_reason as a comma-separated list.

    R1 PRIMARY_KEY_NULL       supplier key missing or blank
    R2 PRIMARY_KEY_DUPLICATE  supplier key occurs more than once in the batch
    R3 UNKNOWN_NATION         nation key does not exist in silver.nation
    """

    # Read valid nation keys from silver.nation for referential integrity.
    valid_nation_keys = (
        spark.table(NATION_TABLE)
        .select(F.col("nation_key").alias("_valid_nation_key"))
        .distinct()
    )

    # Left join to mark suppliers whose nation_key is missing from silver.nation.
    df = df.join(
        F.broadcast(valid_nation_keys),
        F.col("nation_key") == F.col("_valid_nation_key"),
        "left"
    )

    # How many rows share this key in this batch.
    duplicate_count = F.count(F.lit(1)).over(
        Window.partitionBy("supplier_key_raw")
    )

    checked = (
        df.withColumn("_key_dup_count", duplicate_count)
        # R1
        .withColumn(
            "_r1",
            F.col("supplier_key_raw").isNull()
        )
        # R2: only non-null keys can be duplicates.
        .withColumn(
            "_r2",
            F.col("supplier_key_raw").isNotNull()
            & (F.col("_key_dup_count") > 1)
        )
        # R3: nation_key not found in silver.nation (also catches NULL nation_key).
        .withColumn(
            "_r3",
            F.col("_valid_nation_key").isNull()
        )
        # Comma-separated list of EVERY failed rule for this record.
        .withColumn(
            "rejection_reason",
            F.concat_ws(
                ",",
                F.when(F.col("_r1"), F.lit("PRIMARY_KEY_NULL")),
                F.when(F.col("_r2"), F.lit("PRIMARY_KEY_DUPLICATE")),
                F.when(F.col("_r3"), F.lit("UNKNOWN_NATION"))
            )
        )
        .drop("_key_dup_count", "_r1", "_r2", "_r3", "_valid_nation_key")
    )

    invalid_df = checked.filter(
        F.length(F.col("rejection_reason")) > 0
    )

    valid_df = checked.filter(
        F.length(F.col("rejection_reason")) == 0
    )

    return valid_df, invalid_df


# ─────────────────────────────────────────────
# 4. Create output tables only if missing
# ─────────────────────────────────────────────
def create_tables_if_not_exists():

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {SUPPLIER_TABLE} (
            supplier_key INT,
            supplier_name STRING,
            supplier_address STRING,
            nation_key INT,
            supplier_phone STRING,
            supplier_acctbal DECIMAL(18,2),
            supplier_comment STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {QUARANTINE_TABLE} (
            supplier_key STRING,
            rejection_reason STRING,
            raw_record STRING,
            quarantined_at TIMESTAMP
        ) USING DELTA
    """)

# ─────────────────────────────────────────────
# 5. Prepare valid and quarantine outputs
# ─────────────────────────────────────────────
def prepare_outputs(
    valid_df: DataFrame,
    invalid_df: DataFrame
) -> tuple[DataFrame, DataFrame]:

    suppliers_out = valid_df.select(
        "supplier_key",
        "supplier_name",
        "supplier_address",
        "nation_key",
        "supplier_phone",
        "supplier_acctbal",
        "supplier_comment"
    )

    # NULL keys must stay joinable, so coalesce to a sentinel.
    quarantine_out = invalid_df.select(
        F.coalesce(
            F.col("supplier_key_raw"), F.lit("<NULL>")
        ).alias("supplier_key"),
        "rejection_reason",
        "raw_record",
        F.current_timestamp().alias("quarantined_at")
    )

    return suppliers_out, quarantine_out

# ─────────────────────────────────────────────
# 6. MERGE suppliers and remove stale invalid rows
# ─────────────────────────────────────────────
def merge_suppliers(
    latest_df: DataFrame,
    suppliers_out: DataFrame
):
    """
    A supplier must be in Silver or quarantine, never both.
    If its latest version is rejected, remove its old Silver row.
    """

    latest_df.select("supplier_key").distinct().createOrReplaceTempView(
        "_latest_supplier_ids"
    )

    suppliers_out.select("supplier_key").distinct().createOrReplaceTempView(
        "_valid_supplier_ids"
    )

    spark.sql(f"""
        MERGE INTO {SUPPLIER_TABLE} AS t
        USING (
            SELECT l.supplier_key
            FROM _latest_supplier_ids l
            LEFT ANTI JOIN _valid_supplier_ids v
              ON l.supplier_key = v.supplier_key
        ) AS s
        ON t.supplier_key = s.supplier_key
        WHEN MATCHED THEN DELETE
    """)

    suppliers_out.createOrReplaceTempView("_suppliers_source")

    spark.sql(f"""
        MERGE INTO {SUPPLIER_TABLE} AS t
        USING _suppliers_source AS s
        ON t.supplier_key = s.supplier_key

        WHEN MATCHED AND (
            NOT (t.supplier_name <=> s.supplier_name)
            OR NOT (t.supplier_address <=> s.supplier_address)
            OR NOT (t.nation_key <=> s.nation_key)
            OR NOT (t.supplier_phone <=> s.supplier_phone)
            OR NOT (t.supplier_acctbal <=> s.supplier_acctbal)
            OR NOT (t.supplier_comment <=> s.supplier_comment)
        )
        THEN UPDATE SET
            t.supplier_name = s.supplier_name,
            t.supplier_address = s.supplier_address,
            t.nation_key = s.nation_key,
            t.supplier_phone = s.supplier_phone,
            t.supplier_acctbal = s.supplier_acctbal,
            t.supplier_comment = s.supplier_comment

        WHEN NOT MATCHED THEN INSERT (
            supplier_key, supplier_name, supplier_address, nation_key,
            supplier_phone, supplier_acctbal, supplier_comment
        )
        VALUES (
            s.supplier_key, s.supplier_name, s.supplier_address, s.nation_key,
            s.supplier_phone, s.supplier_acctbal, s.supplier_comment
        )
    """)

# ─────────────────────────────────────────────
# 7. MERGE quarantine and recover valid suppliers
# ─────────────────────────────────────────────
def merge_quarantine(
    latest_df: DataFrame,
    quarantine_out: DataFrame
):
    """
    Keep one quarantine record per rejected key.
    Remove quarantine records for suppliers now valid.
    """

    latest_df.select(
        F.coalesce(
            F.col("supplier_key_raw"), F.lit("<NULL>")
        ).alias("supplier_key")
    ).distinct().createOrReplaceTempView("_processed_ids")

    quarantine_out.select("supplier_key").distinct().createOrReplaceTempView(
        "_rejected_ids"
    )

    spark.sql(f"""
        DELETE FROM {QUARANTINE_TABLE}
        WHERE supplier_key IN (
            SELECT p.supplier_key
            FROM _processed_ids p
            LEFT ANTI JOIN _rejected_ids r
              ON p.supplier_key = r.supplier_key
        )
    """)

    quarantine_out.createOrReplaceTempView("_quarantine_source")

    spark.sql(f"""
        MERGE INTO {QUARANTINE_TABLE} AS t
        USING _quarantine_source AS s
        ON t.supplier_key = s.supplier_key

        WHEN MATCHED AND (
            NOT (t.rejection_reason <=> s.rejection_reason)
            OR NOT (t.raw_record <=> s.raw_record)
        )
        THEN UPDATE SET
            t.rejection_reason = s.rejection_reason,
            t.raw_record = s.raw_record,
            t.quarantined_at = s.quarantined_at

        WHEN NOT MATCHED THEN INSERT (
            supplier_key, rejection_reason, raw_record, quarantined_at
        )
        VALUES (
            s.supplier_key, s.rejection_reason, s.raw_record, s.quarantined_at
        )
    """)

# ─────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────
def main():

    # Read Bronze
    bronze_df = spark.read.table(BRONZE_TABLE)

    # Standardize first
    standardized_df = standardize_suppliers(bronze_df)

    # Keep latest version per supplier key
    latest_df = latest_supplier_versions(standardized_df)

    # Validate latest version only
    valid_df, invalid_df = validate_suppliers(latest_df)

    # Prepare output DataFrames
    suppliers_out, quarantine_out = prepare_outputs(valid_df, invalid_df)

    # Create tables once
    create_tables_if_not_exists()

    # Incremental target updates
    merge_suppliers(latest_df, suppliers_out)
    merge_quarantine(latest_df, quarantine_out)

    print("=" * 45)
    print("Latest supplier versions:", latest_df.count())
    print("Valid suppliers:", suppliers_out.count())
    print("Rejected suppliers:", quarantine_out.count())
    print("=" * 45)


if __name__ == "__main__":
    main()