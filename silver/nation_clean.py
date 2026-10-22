from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CATALOG = "tpch_dev"

BRONZE_TABLE = f"{CATALOG}.bronze.nation"

NATION_TABLE = f"{CATALOG}.silver.nation"
QUARANTINE_TABLE = f"{CATALOG}.silver.nation_quarantine"

# Referential integrity target
REGION_TABLE = f"{CATALOG}.silver.region"

# ─────────────────────────────────────────────
# 1. Standardize nations (S1-S4)
# ─────────────────────────────────────────────
def standardize_nations(df: DataFrame) -> DataFrame:
    """
    Standardization happens before deduplication and validation.

    S1: nation key trim; empty -> NULL
    S2: nation name trim + uppercase; empty -> NULL
    S3: region key trim; empty -> NULL
    S4: nation comment trim; empty -> NULL
    """

    # Preserve the complete original Bronze row as JSON.
    df = df.withColumn(
        "raw_record",
        F.to_json(F.struct(*[F.col(c) for c in df.columns]))
    )

    # S1: keep the raw text key for quarantine traceability.
    df = df.withColumn(
        "nation_key_raw",
        F.when(
            F.trim(F.col("n_nationkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("n_nationkey").cast("string"))
        )
    )

    # S1b: strict integer parse - anything else becomes NULL
    df = df.withColumn(
        "nation_key",
        F.when(
            F.col("nation_key_raw").rlike("^-?\\d+$"),
            F.col("nation_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S2
    df = df.withColumn(
        "nation_name",
        F.when(
            F.trim(F.col("n_name").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.upper(F.trim(F.col("n_name").cast("string")))
        )
    )

    # S3: region key - raw text + strict integer parse
    df = df.withColumn(
        "region_key_raw",
        F.when(
            F.trim(F.col("n_regionkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("n_regionkey").cast("string"))
        )
    )

    df = df.withColumn(
        "region_key",
        F.when(
            F.col("region_key_raw").rlike("^-?\\d+$"),
            F.col("region_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S4
    df = df.withColumn(
        "nation_comment",
        F.when(
            F.trim(F.col("n_comment").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("n_comment").cast("string"))
        )
    )

    # Keep _ingestion_at until deduplication has completed.
    return df.drop("_source_file")


# ─────────────────────────────────────────────
# 2. Latest version per nation key (C2)
# ─────────────────────────────────────────────
def latest_nation_versions(df: DataFrame) -> DataFrame:
    """
    Validate only the most recently ingested version of each key.

    _ingestion_at is used as the ordering key.
    """

    window_spec = (
        Window
        .partitionBy("nation_key_raw")
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
def validate_nations(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Reject a record if ANY rule fails. Every failed rule is
    collected into rejection_reason as a comma-separated list.

    R1 PRIMARY_KEY_NULL       nation key missing or blank
    R2 PRIMARY_KEY_DUPLICATE  nation key occurs more than once in the batch
    R3 REFERENTIAL_INTEGRITY  region key does not exist in silver.region
    """

    # Read valid region keys from silver.region for referential integrity.
    valid_region_keys = (
        spark.table(REGION_TABLE)
        .select(F.col("region_key").alias("_valid_region_key"))
        .distinct()
    )

    # Left join to mark nations whose region_key is missing from silver.region.
    df = df.join(
        F.broadcast(valid_region_keys),
        F.col("region_key") == F.col("_valid_region_key"),
        "left"
    )

    # How many rows share this key in this batch.
    duplicate_count = F.count(F.lit(1)).over(
        Window.partitionBy("nation_key_raw")
    )

    checked = (
        df.withColumn("_key_dup_count", duplicate_count)
        # R1
        .withColumn(
            "_r1",
            F.col("nation_key_raw").isNull()
        )
        # R2: only non-null keys can be duplicates.
        .withColumn(
            "_r2",
            F.col("nation_key_raw").isNotNull()
            & (F.col("_key_dup_count") > 1)
        )
        # R3: region_key not found in silver.region (also catches NULL region_key).
        .withColumn(
            "_r3",
            F.col("_valid_region_key").isNull()
        )
        # Comma-separated list of EVERY failed rule for this record.
        .withColumn(
            "rejection_reason",
            F.concat_ws(
                ",",
                F.when(F.col("_r1"), F.lit("PRIMARY_KEY_NULL")),
                F.when(F.col("_r2"), F.lit("PRIMARY_KEY_DUPLICATE")),
                F.when(F.col("_r3"), F.lit("UNKNOWN_REGION"))
            )
        )
        .drop("_key_dup_count", "_r1", "_r2", "_r3", "_valid_region_key")
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
        CREATE TABLE IF NOT EXISTS {NATION_TABLE} (
            nation_key INT,
            nation_name STRING,
            region_key INT,
            nation_comment STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {QUARANTINE_TABLE} (
            nation_key STRING,
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

    nations_out = valid_df.select(
        "nation_key",
        "nation_name",
        "region_key",
        "nation_comment"
    )

    # NULL keys must stay joinable, so coalesce to a sentinel.
    quarantine_out = invalid_df.select(
        F.coalesce(
            F.col("nation_key_raw"), F.lit("<NULL>")
        ).alias("nation_key"),
        "rejection_reason",
        "raw_record",
        F.current_timestamp().alias("quarantined_at")
    )

    return nations_out, quarantine_out

# ─────────────────────────────────────────────
# 6. MERGE nations and remove stale invalid rows
# ─────────────────────────────────────────────
def merge_nations(
    latest_df: DataFrame,
    nations_out: DataFrame
):
    """
    A nation must be in Silver or quarantine, never both.
    If its latest version is rejected, remove its old Silver row.
    """

    latest_df.select("nation_key").distinct().createOrReplaceTempView(
        "_latest_nation_ids"
    )

    nations_out.select("nation_key").distinct().createOrReplaceTempView(
        "_valid_nation_ids"
    )

    spark.sql(f"""
        MERGE INTO {NATION_TABLE} AS t
        USING (
            SELECT l.nation_key
            FROM _latest_nation_ids l
            LEFT ANTI JOIN _valid_nation_ids v
              ON l.nation_key = v.nation_key
        ) AS s
        ON t.nation_key = s.nation_key
        WHEN MATCHED THEN DELETE
    """)

    nations_out.createOrReplaceTempView("_nations_source")

    spark.sql(f"""
        MERGE INTO {NATION_TABLE} AS t
        USING _nations_source AS s
        ON t.nation_key = s.nation_key

        WHEN MATCHED AND (
            NOT (t.nation_name <=> s.nation_name)
            OR NOT (t.region_key <=> s.region_key)
            OR NOT (t.nation_comment <=> s.nation_comment)
        )
        THEN UPDATE SET
            t.nation_name = s.nation_name,
            t.region_key = s.region_key,
            t.nation_comment = s.nation_comment

        WHEN NOT MATCHED THEN INSERT (
            nation_key, nation_name, region_key, nation_comment
        )
        VALUES (
            s.nation_key, s.nation_name, s.region_key, s.nation_comment
        )
    """)

# ─────────────────────────────────────────────
# 7. MERGE quarantine and recover valid nations
# ─────────────────────────────────────────────
def merge_quarantine(
    latest_df: DataFrame,
    quarantine_out: DataFrame
):
    """
    Keep one quarantine record per rejected key.
    Remove quarantine records for nations now valid.
    """

    latest_df.select(
        F.coalesce(
            F.col("nation_key_raw"), F.lit("<NULL>")
        ).alias("nation_key")
    ).distinct().createOrReplaceTempView("_processed_ids")

    quarantine_out.select("nation_key").distinct().createOrReplaceTempView(
        "_rejected_ids"
    )

    spark.sql(f"""
        DELETE FROM {QUARANTINE_TABLE}
        WHERE nation_key IN (
            SELECT p.nation_key
            FROM _processed_ids p
            LEFT ANTI JOIN _rejected_ids r
              ON p.nation_key = r.nation_key
        )
    """)

    quarantine_out.createOrReplaceTempView("_quarantine_source")

    spark.sql(f"""
        MERGE INTO {QUARANTINE_TABLE} AS t
        USING _quarantine_source AS s
        ON t.nation_key = s.nation_key

        WHEN MATCHED AND (
            NOT (t.rejection_reason <=> s.rejection_reason)
            OR NOT (t.raw_record <=> s.raw_record)
        )
        THEN UPDATE SET
            t.rejection_reason = s.rejection_reason,
            t.raw_record = s.raw_record,
            t.quarantined_at = s.quarantined_at

        WHEN NOT MATCHED THEN INSERT (
            nation_key, rejection_reason, raw_record, quarantined_at
        )
        VALUES (
            s.nation_key, s.rejection_reason, s.raw_record, s.quarantined_at
        )
    """)

# ─────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────
def main():

    # Read Bronze
    bronze_df = spark.read.table(BRONZE_TABLE)

    # Standardize first
    standardized_df = standardize_nations(bronze_df)

    # Keep latest version per nation key
    latest_df = latest_nation_versions(standardized_df)

    # Validate latest version only
    valid_df, invalid_df = validate_nations(latest_df)

    # Prepare output DataFrames
    nations_out, quarantine_out = prepare_outputs(valid_df, invalid_df)

    # Create tables once
    create_tables_if_not_exists()

    # Incremental target updates
    merge_nations(latest_df, nations_out)
    merge_quarantine(latest_df, quarantine_out)

    print("=" * 45)
    print("Latest nation versions:", latest_df.count())
    print("Valid nations:", nations_out.count())
    print("Rejected nations:", quarantine_out.count())
    print("=" * 45)


if __name__ == "__main__":
    main()