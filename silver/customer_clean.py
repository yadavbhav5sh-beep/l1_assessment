from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CATALOG = "tpch_dev"

BRONZE_TABLE = f"{CATALOG}.bronze.customer"

CUSTOMER_TABLE = f"{CATALOG}.silver.customer"
QUARANTINE_TABLE = f"{CATALOG}.silver.customer_quarantine"

# Referential integrity target
NATION_TABLE = f"{CATALOG}.silver.nation"

# ─────────────────────────────────────────────
# 1. Standardize customers (S1-S6)
# ─────────────────────────────────────────────
def standardize_customers(df: DataFrame) -> DataFrame:
    """
    Standardization happens before deduplication and validation.

    S1: customer key trim; empty -> NULL
    S2: customer name trim; empty -> NULL
    S3: address trim; empty -> NULL
    S4: nation key trim; empty -> NULL
    S5: phone / mktsegment / comment trim; empty -> NULL
    S6: account balance cast to decimal(18,2)
    """

    # Preserve the complete original Bronze row as JSON.
    df = df.withColumn(
        "raw_record",
        F.to_json(F.struct(*[F.col(c) for c in df.columns]))
    )

    # S1: keep the raw text key for quarantine traceability.
    df = df.withColumn(
        "customer_key_raw",
        F.when(
            F.trim(F.col("c_custkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("c_custkey").cast("string"))
        )
    )

    # S1b: strict integer parse - anything else becomes NULL
    df = df.withColumn(
        "customer_key",
        F.when(
            F.col("customer_key_raw").rlike("^-?\\d+$"),
            F.col("customer_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S2
    df = df.withColumn(
        "customer_name",
        F.when(
            F.trim(F.col("c_name").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("c_name").cast("string"))
        )
    )

    # S3
    df = df.withColumn(
        "customer_address",
        F.when(
            F.trim(F.col("c_address").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("c_address").cast("string"))
        )
    )

    # S4: nation key - raw text + strict integer parse
    df = df.withColumn(
        "nation_key_raw",
        F.when(
            F.trim(F.col("c_nationkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("c_nationkey").cast("string"))
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
        "customer_phone",
        F.when(
            F.trim(F.col("c_phone").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("c_phone").cast("string"))
        )
    )

    df = df.withColumn(
        "customer_mktsegment",
        F.when(
            F.trim(F.col("c_mktsegment").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.upper(F.trim(F.col("c_mktsegment").cast("string")))
        )
    )

    df = df.withColumn(
        "customer_comment",
        F.when(
            F.trim(F.col("c_comment").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("c_comment").cast("string"))
        )
    )

    # S6: account balance - cast to decimal(18,2), invalid -> NULL
    df = df.withColumn(
        "customer_acctbal",
        F.col("c_acctbal").cast("decimal(18,2)")
    )

    # Keep _ingestion_at until deduplication has completed.
    return df.drop("_source_file")


# ─────────────────────────────────────────────
# 2. Latest version per customer key (C2)
# ─────────────────────────────────────────────
def latest_customer_versions(df: DataFrame) -> DataFrame:
    """
    Validate only the most recently ingested version of each key.

    _ingestion_at is used as the ordering key.
    """

    window_spec = (
        Window
        .partitionBy("customer_key_raw")
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
def validate_customers(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Reject a record if ANY rule fails. Every failed rule is
    collected into rejection_reason as a comma-separated list.

    R1 PRIMARY_KEY_NULL       customer key missing or blank
    R2 PRIMARY_KEY_DUPLICATE  customer key occurs more than once in the batch
    R3 REFERENTIAL_INTEGRITY  nation key does not exist in silver.nation
    """

    # Read valid nation keys from silver.nation for referential integrity.
    valid_nation_keys = (
        spark.table(NATION_TABLE)
        .select(F.col("nation_key").alias("_valid_nation_key"))
        .distinct()
    )

    # Left join to mark customers whose nation_key is missing from silver.nation.
    df = df.join(
        F.broadcast(valid_nation_keys),
        F.col("nation_key") == F.col("_valid_nation_key"),
        "left"
    )

    # How many rows share this key in this batch.
    duplicate_count = F.count(F.lit(1)).over(
        Window.partitionBy("customer_key_raw")
    )

    checked = (
        df.withColumn("_key_dup_count", duplicate_count)
        # R1
        .withColumn(
            "_r1",
            F.col("customer_key_raw").isNull()
        )
        # R2: only non-null keys can be duplicates.
        .withColumn(
            "_r2",
            F.col("customer_key_raw").isNotNull()
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
                F.when(F.col("_r3"), F.lit("UNKOWN_NATION"))
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
        CREATE TABLE IF NOT EXISTS {CUSTOMER_TABLE} (
            customer_key INT,
            customer_name STRING,
            customer_address STRING,
            nation_key INT,
            customer_phone STRING,
            customer_acctbal DECIMAL(18,2),
            customer_mktsegment STRING,
            customer_comment STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {QUARANTINE_TABLE} (
            customer_key STRING,
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

    customers_out = valid_df.select(
        "customer_key",
        "customer_name",
        "customer_address",
        "nation_key",
        "customer_phone",
        "customer_acctbal",
        "customer_mktsegment",
        "customer_comment"
    )

    # NULL keys must stay joinable, so coalesce to a sentinel.
    quarantine_out = invalid_df.select(
        F.coalesce(
            F.col("customer_key_raw"), F.lit("<NULL>")
        ).alias("customer_key"),
        "rejection_reason",
        "raw_record",
        F.current_timestamp().alias("quarantined_at")
    )

    return customers_out, quarantine_out

# ─────────────────────────────────────────────
# 6. MERGE customers and remove stale invalid rows
# ─────────────────────────────────────────────
def merge_customers(
    latest_df: DataFrame,
    customers_out: DataFrame
):
    """
    A customer must be in Silver or quarantine, never both.
    If its latest version is rejected, remove its old Silver row.
    """

    latest_df.select("customer_key").distinct().createOrReplaceTempView(
        "_latest_customer_ids"
    )

    customers_out.select("customer_key").distinct().createOrReplaceTempView(
        "_valid_customer_ids"
    )

    spark.sql(f"""
        MERGE INTO {CUSTOMER_TABLE} AS t
        USING (
            SELECT l.customer_key
            FROM _latest_customer_ids l
            LEFT ANTI JOIN _valid_customer_ids v
              ON l.customer_key = v.customer_key
        ) AS s
        ON t.customer_key = s.customer_key
        WHEN MATCHED THEN DELETE
    """)

    customers_out.createOrReplaceTempView("_customers_source")

    spark.sql(f"""
        MERGE INTO {CUSTOMER_TABLE} AS t
        USING _customers_source AS s
        ON t.customer_key = s.customer_key

        WHEN MATCHED AND (
            NOT (t.customer_name <=> s.customer_name)
            OR NOT (t.customer_address <=> s.customer_address)
            OR NOT (t.nation_key <=> s.nation_key)
            OR NOT (t.customer_phone <=> s.customer_phone)
            OR NOT (t.customer_acctbal <=> s.customer_acctbal)
            OR NOT (t.customer_mktsegment <=> s.customer_mktsegment)
            OR NOT (t.customer_comment <=> s.customer_comment)
        )
        THEN UPDATE SET
            t.customer_name = s.customer_name,
            t.customer_address = s.customer_address,
            t.nation_key = s.nation_key,
            t.customer_phone = s.customer_phone,
            t.customer_acctbal = s.customer_acctbal,
            t.customer_mktsegment = s.customer_mktsegment,
            t.customer_comment = s.customer_comment

        WHEN NOT MATCHED THEN INSERT (
            customer_key, customer_name, customer_address, nation_key,
            customer_phone, customer_acctbal, customer_mktsegment, customer_comment
        )
        VALUES (
            s.customer_key, s.customer_name, s.customer_address, s.nation_key,
            s.customer_phone, s.customer_acctbal, s.customer_mktsegment, s.customer_comment
        )
    """)

# ─────────────────────────────────────────────
# 7. MERGE quarantine and recover valid customers
# ─────────────────────────────────────────────
def merge_quarantine(
    latest_df: DataFrame,
    quarantine_out: DataFrame
):
    """
    Keep one quarantine record per rejected key.
    Remove quarantine records for customers now valid.
    """

    latest_df.select(
        F.coalesce(
            F.col("customer_key_raw"), F.lit("<NULL>")
        ).alias("customer_key")
    ).distinct().createOrReplaceTempView("_processed_ids")

    quarantine_out.select("customer_key").distinct().createOrReplaceTempView(
        "_rejected_ids"
    )

    spark.sql(f"""
        DELETE FROM {QUARANTINE_TABLE}
        WHERE customer_key IN (
            SELECT p.customer_key
            FROM _processed_ids p
            LEFT ANTI JOIN _rejected_ids r
              ON p.customer_key = r.customer_key
        )
    """)

    quarantine_out.createOrReplaceTempView("_quarantine_source")

    spark.sql(f"""
        MERGE INTO {QUARANTINE_TABLE} AS t
        USING _quarantine_source AS s
        ON t.customer_key = s.customer_key

        WHEN MATCHED AND (
            NOT (t.rejection_reason <=> s.rejection_reason)
            OR NOT (t.raw_record <=> s.raw_record)
        )
        THEN UPDATE SET
            t.rejection_reason = s.rejection_reason,
            t.raw_record = s.raw_record,
            t.quarantined_at = s.quarantined_at

        WHEN NOT MATCHED THEN INSERT (
            customer_key, rejection_reason, raw_record, quarantined_at
        )
        VALUES (
            s.customer_key, s.rejection_reason, s.raw_record, s.quarantined_at
        )
    """)

# ─────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────
def main():

    # Read Bronze
    bronze_df = spark.read.table(BRONZE_TABLE)

    # Standardize first
    standardized_df = standardize_customers(bronze_df)

    # Keep latest version per customer key
    latest_df = latest_customer_versions(standardized_df)

    # Validate latest version only
    valid_df, invalid_df = validate_customers(latest_df)

    # Prepare output DataFrames
    customers_out, quarantine_out = prepare_outputs(valid_df, invalid_df)

    # Create tables once
    create_tables_if_not_exists()

    # Incremental target updates
    merge_customers(latest_df, customers_out)
    merge_quarantine(latest_df, quarantine_out)

    print("=" * 45)
    print("Latest customer versions:", latest_df.count())
    print("Valid customers:", customers_out.count())
    print("Rejected customers:", quarantine_out.count())
    print("=" * 45)


if __name__ == "__main__":
    main()