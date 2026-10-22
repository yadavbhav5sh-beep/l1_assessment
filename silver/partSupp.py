from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CATALOG = "tpch_dev"

BRONZE_TABLE = f"{CATALOG}.bronze.partsupp"

PARTSUPP_TABLE = f"{CATALOG}.silver.partsupp"
QUARANTINE_TABLE = f"{CATALOG}.silver.partsupp_quarantine"

# Referential integrity targets
PART_TABLE = f"{CATALOG}.silver.part"
SUPPLIER_TABLE = f"{CATALOG}.silver.supplier"

# ─────────────────────────────────────────────
# 1. Standardize partsupps (S1-S4)
# ─────────────────────────────────────────────
def standardize_partsupps(df: DataFrame) -> DataFrame:
    """
    Standardization happens before deduplication and validation.

    S1: part key trim; empty -> NULL
    S2: supplier key trim; empty -> NULL
    S3: available qty cast to int; supply cost cast to decimal(18,2)
    S4: comment trim; empty -> NULL
    """

    # Preserve the complete original Bronze row as JSON.
    df = df.withColumn(
        "raw_record",
        F.to_json(F.struct(*[F.col(c) for c in df.columns]))
    )

    # S1: part key - raw text + strict integer parse
    df = df.withColumn(
        "part_key_raw",
        F.when(
            F.trim(F.col("ps_partkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("ps_partkey").cast("string"))
        )
    )

    df = df.withColumn(
        "part_key",
        F.when(
            F.col("part_key_raw").rlike("^-?\\d+$"),
            F.col("part_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S2: supplier key - raw text + strict integer parse
    df = df.withColumn(
        "supplier_key_raw",
        F.when(
            F.trim(F.col("ps_suppkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("ps_suppkey").cast("string"))
        )
    )

    df = df.withColumn(
        "supplier_key",
        F.when(
            F.col("supplier_key_raw").rlike("^-?\\d+$"),
            F.col("supplier_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S3
    df = df.withColumn(
        "availqty",
        F.col("ps_availqty").cast("int")
    )

    df = df.withColumn(
        "supplycost",
        F.col("ps_supplycost").cast("decimal(18,2)")
    )

    # S4
    df = df.withColumn(
        "partsupp_comment",
        F.when(
            F.trim(F.col("ps_comment").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("ps_comment").cast("string"))
        )
    )

    # Keep _ingestion_at until deduplication has completed.
    return df.drop("_source_file")


# ─────────────────────────────────────────────
# 2. Latest version per composite key (C2)
# ─────────────────────────────────────────────
def latest_partsupp_versions(df: DataFrame) -> DataFrame:
    """
    Validate only the most recently ingested version of each
    composite key (part_key_raw + supplier_key_raw).

    _ingestion_at is used as the ordering key.
    """

    window_spec = (
        Window
        .partitionBy("part_key_raw", "supplier_key_raw")
        .orderBy(F.col("_ingestion_at").desc_nulls_last())
    )

    return (
        df
        .withColumn("_rn", F.row_number().over(window_spec))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

# ─────────────────────────────────────────────
# 3. Data quality rules (R1-R4)
# ─────────────────────────────────────────────
def validate_partsupps(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Reject a record if ANY rule fails. Every failed rule is
    collected into rejection_reason as a comma-separated list.

    R1 PRIMARY_KEY_NULL_PART      part key missing or blank
    R2 PRIMARY_KEY_NULL_SUPPLIER  supplier key missing or blank
    R3 PRIMARY_KEY_DUPLICATE     (part_key + supplier_key) occurs more than once
    R4 UNKNOWN_PART               part key not in silver.part
    R5 UNKNOWN_SUPPLIER          supplier key not in silver.supplier
    """

    # Read valid part keys from silver.part for referential integrity.
    valid_part_keys = (
        spark.table(PART_TABLE)
        .select(F.col("part_key").alias("_valid_part_key"))
        .distinct()
    )

    # Read valid supplier keys from silver.supplier for referential integrity.
    valid_supplier_keys = (
        spark.table(SUPPLIER_TABLE)
        .select(F.col("supplier_key").alias("_valid_supplier_key"))
        .distinct()
    )

    # Left joins to mark partsupps whose part_key / supplier_key is missing.
    df = df.join(
        F.broadcast(valid_part_keys),
        F.col("part_key") == F.col("_valid_part_key"),
        "left"
    ).join(
        F.broadcast(valid_supplier_keys),
        F.col("supplier_key") == F.col("_valid_supplier_key"),
        "left"
    )

    # How many rows share this composite key in this batch.
    duplicate_count = F.count(F.lit(1)).over(
        Window.partitionBy("part_key_raw", "supplier_key_raw")
    )

    checked = (
        df.withColumn("_key_dup_count", duplicate_count)
        # R1: part key null
        .withColumn(
            "_r1",
            F.col("part_key_raw").isNull()
        )
        # R2: supplier key null
        .withColumn(
            "_r2",
            F.col("supplier_key_raw").isNull()
        )
        # R3: composite key duplicate (only when both keys are non-null)
        .withColumn(
            "_r3",
            F.col("part_key_raw").isNotNull()
            & F.col("supplier_key_raw").isNotNull()
            & (F.col("_key_dup_count") > 1)
        )
        # R4: part_key not in silver.part
        .withColumn(
            "_r4",
            F.col("_valid_part_key").isNull()
        )
        # R5: supplier_key not in silver.supplier
        .withColumn(
            "_r5",
            F.col("_valid_supplier_key").isNull()
        )
        # Comma-separated list of EVERY failed rule for this record.
        .withColumn(
            "rejection_reason",
            F.concat_ws(
                ",",
                F.when(F.col("_r1"), F.lit("NULL_PART")),
                F.when(F.col("_r2"), F.lit("NULL_SUPPLIER")),
                F.when(F.col("_r3"), F.lit("PRIMARY_KEY_DUPLICATE")),
                F.when(F.col("_r4"), F.lit("UNKNOWN_PART")),
                F.when(F.col("_r5"), F.lit("UNKNOWN_SUPPLIER"))
            )
        )
        .drop("_key_dup_count", "_r1", "_r2", "_r3", "_r4", "_r5",
              "_valid_part_key", "_valid_supplier_key")
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
        CREATE TABLE IF NOT EXISTS {PARTSUPP_TABLE} (
            part_key INT,
            supplier_key INT,
            availqty INT,
            supplycost DECIMAL(18,2),
            partsupp_comment STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {QUARANTINE_TABLE} (
            partsupp_key STRING,
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

    partsupps_out = valid_df.select(
        "part_key",
        "supplier_key",
        "availqty",
        "supplycost",
        "partsupp_comment"
    )

    # Build a composite key string for quarantine traceability.
    # NULL keys must stay joinable, so coalesce to a sentinel.
    quarantine_out = invalid_df.select(
        F.concat_ws(
            "|",
            F.coalesce(F.col("part_key_raw"), F.lit("<NULL>")),
            F.coalesce(F.col("supplier_key_raw"), F.lit("<NULL>"))
        ).alias("partsupp_key"),
        "rejection_reason",
        "raw_record",
        F.current_timestamp().alias("quarantined_at")
    )

    return partsupps_out, quarantine_out

# ─────────────────────────────────────────────
# 6. MERGE partsupps and remove stale invalid rows
# ─────────────────────────────────────────────
def merge_partsupps(
    latest_df: DataFrame,
    partsupps_out: DataFrame
):
    """
    A partsupp must be in Silver or quarantine, never both.
    If its latest version is rejected, remove its old Silver row.
    """

    latest_df.select("part_key", "supplier_key").distinct().createOrReplaceTempView(
        "_latest_partsupp_ids"
    )

    partsupps_out.select("part_key", "supplier_key").distinct().createOrReplaceTempView(
        "_valid_partsupp_ids"
    )

    spark.sql(f"""
        MERGE INTO {PARTSUPP_TABLE} AS t
        USING (
            SELECT l.part_key, l.supplier_key
            FROM _latest_partsupp_ids l
            LEFT ANTI JOIN _valid_partsupp_ids v
              ON l.part_key = v.part_key
             AND l.supplier_key = v.supplier_key
        ) AS s
        ON t.part_key = s.part_key
        AND t.supplier_key = s.supplier_key
        WHEN MATCHED THEN DELETE
    """)

    partsupps_out.createOrReplaceTempView("_partsupps_source")

    spark.sql(f"""
        MERGE INTO {PARTSUPP_TABLE} AS t
        USING _partsupps_source AS s
        ON t.part_key = s.part_key
        AND t.supplier_key = s.supplier_key

        WHEN MATCHED AND (
            NOT (t.availqty <=> s.availqty)
            OR NOT (t.supplycost <=> s.supplycost)
            OR NOT (t.partsupp_comment <=> s.partsupp_comment)
        )
        THEN UPDATE SET
            t.availqty = s.availqty,
            t.supplycost = s.supplycost,
            t.partsupp_comment = s.partsupp_comment

        WHEN NOT MATCHED THEN INSERT (
            part_key, supplier_key, availqty, supplycost, partsupp_comment
        )
        VALUES (
            s.part_key, s.supplier_key, s.availqty, s.supplycost, s.partsupp_comment
        )
    """)

# ─────────────────────────────────────────────
# 7. MERGE quarantine and recover valid partsupps
# ─────────────────────────────────────────────
def merge_quarantine(
    latest_df: DataFrame,
    quarantine_out: DataFrame
):
    """
    Keep one quarantine record per rejected composite key.
    Remove quarantine records for partsupps now valid.
    """

    latest_df.select(
        F.concat_ws(
            "|",
            F.coalesce(F.col("part_key_raw"), F.lit("<NULL>")),
            F.coalesce(F.col("supplier_key_raw"), F.lit("<NULL>"))
        ).alias("partsupp_key")
    ).distinct().createOrReplaceTempView("_processed_ids")

    quarantine_out.select("partsupp_key").distinct().createOrReplaceTempView(
        "_rejected_ids"
    )

    spark.sql(f"""
        DELETE FROM {QUARANTINE_TABLE}
        WHERE partsupp_key IN (
            SELECT p.partsupp_key
            FROM _processed_ids p
            LEFT ANTI JOIN _rejected_ids r
              ON p.partsupp_key = r.partsupp_key
        )
    """)

    quarantine_out.createOrReplaceTempView("_quarantine_source")

    spark.sql(f"""
        MERGE INTO {QUARANTINE_TABLE} AS t
        USING _quarantine_source AS s
        ON t.partsupp_key = s.partsupp_key

        WHEN MATCHED AND (
            NOT (t.rejection_reason <=> s.rejection_reason)
            OR NOT (t.raw_record <=> s.raw_record)
        )
        THEN UPDATE SET
            t.rejection_reason = s.rejection_reason,
            t.raw_record = s.raw_record,
            t.quarantined_at = s.quarantined_at

        WHEN NOT MATCHED THEN INSERT (
            partsupp_key, rejection_reason, raw_record, quarantined_at
        )
        VALUES (
            s.partsupp_key, s.rejection_reason, s.raw_record, s.quarantined_at
        )
    """)

# ─────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────
def main():

    # Read Bronze
    bronze_df = spark.read.table(BRONZE_TABLE)

    # Standardize first
    standardized_df = standardize_partsupps(bronze_df)

    # Keep latest version per composite key
    latest_df = latest_partsupp_versions(standardized_df)

    # Validate latest version only
    valid_df, invalid_df = validate_partsupps(latest_df)

    # Prepare output DataFrames
    partsupps_out, quarantine_out = prepare_outputs(valid_df, invalid_df)

    # Create tables once
    create_tables_if_not_exists()

    # Incremental target updates
    merge_partsupps(latest_df, partsupps_out)
    merge_quarantine(latest_df, quarantine_out)

    print("=" * 45)
    print("Latest partsupp versions:", latest_df.count())
    print("Valid partsupps:", partsupps_out.count())
    print("Rejected partsupps:", quarantine_out.count())
    print("=" * 45)


if __name__ == "__main__":
    main()