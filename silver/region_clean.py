from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CATALOG = "tpch_dev"

BRONZE_TABLE = f"{CATALOG}.bronze.region"

REGION_TABLE = f"{CATALOG}.silver.region"
QUARANTINE_TABLE = f"{CATALOG}.silver.region_quarantine"

# ─────────────────────────────────────────────
# 1. Standardize regions (S1-S3)
# ─────────────────────────────────────────────
def standardize_regions(df: DataFrame) -> DataFrame:
    """
    Standardization happens before deduplication and validation.

    S1: region key trim; empty -> NULL
    S2: region name trim + uppercase; empty -> NULL
    S3: region comment trim; empty -> NULL
    """

    # Preserve the complete original Bronze row as JSON.
    df = df.withColumn(
        "raw_record",
        F.to_json(F.struct(*[F.col(c) for c in df.columns]))
    )

    # S1: keep the raw text key for quarantine traceability.
    df = df.withColumn(
        "region_key_raw",
        F.when(
            F.trim(F.col("r_regionkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("r_regionkey").cast("string"))
        )
    )

    # S1b: strict integer parse - anything else becomes NULL
    df = df.withColumn(
        "region_key",
        F.when(
            F.col("region_key_raw").rlike("^-?\\d+$"),
            F.col("region_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S2
    df = df.withColumn(
        "region_name",
        F.when(
            F.trim(F.col("r_name").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.upper(F.trim(F.col("r_name").cast("string")))
        )
    )

    # S3
    df = df.withColumn(
        "region_comment",
        F.when(
            F.trim(F.col("r_comment").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("r_comment").cast("string"))
        )
    )

    # Keep _ingestion_at until deduplication has completed.
    return df.drop("_source_file")


# ─────────────────────────────────────────────
# 2. Latest version per region key (C2)
# ─────────────────────────────────────────────
def latest_region_versions(df: DataFrame) -> DataFrame:
    """
    Validate only the most recently ingested version of each key.

    _ingestion_at is used as the ordering key.
    """

    window_spec = (
        Window
        .partitionBy("region_key_raw")
        .orderBy(F.col("_ingestion_at").desc_nulls_last())
    )

    return (
        df
        .withColumn("_rn", F.row_number().over(window_spec))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

# ─────────────────────────────────────────────
# 3. Data quality rules (R1-R2)
# ─────────────────────────────────────────────
def validate_regions(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Reject a record if ANY rule fails. Every failed rule is
    collected into rejection_reason as a comma-separated list.

    R1 PRIMARY_KEY_NULL       region key missing or blank
    R2 PRIMARY_KEY_DUPLICATE  region key occurs more than once in the batch
    """

    # How many rows share this key in this batch.
    duplicate_count = F.count(F.lit(1)).over(
        Window.partitionBy("region_key_raw")
    )

    checked = (
        df.withColumn("_key_dup_count", duplicate_count)
        # R1
        .withColumn(
            "_r1",
            F.col("region_key_raw").isNull()
        )
        # R2: only non-null keys can be duplicates.
        .withColumn(
            "_r2",
            F.col("region_key_raw").isNotNull()
            & (F.col("_key_dup_count") > 1)
        )
        # Comma-separated list of EVERY failed rule for this record.
        .withColumn(
            "rejection_reason",
            F.concat_ws(
                ",",
                F.when(F.col("_r1"), F.lit("PRIMARY_KEY_NULL")),
                F.when(F.col("_r2"), F.lit("PRIMARY_KEY_DUPLICATE"))
            )
        )
        .drop("_key_dup_count", "_r1", "_r2")
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
        CREATE TABLE IF NOT EXISTS {REGION_TABLE} (
            region_key INT,
            region_name STRING,
            region_comment STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {QUARANTINE_TABLE} (
            region_key STRING,
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

    regions_out = valid_df.select(
        "region_key",
        "region_name",
        "region_comment"
    )

    # NULL keys must stay joinable, so coalesce to a sentinel.
    quarantine_out = invalid_df.select(
        F.coalesce(
            F.col("region_key_raw"), F.lit("<NULL>")
        ).alias("region_key"),
        "rejection_reason",
        "raw_record",
        F.current_timestamp().alias("quarantined_at")
    )

    return regions_out, quarantine_out

# ─────────────────────────────────────────────
# 6. MERGE regions and remove stale invalid rows
# ─────────────────────────────────────────────
def merge_regions(
    latest_df: DataFrame,
    regions_out: DataFrame
):
    """
    A region must be in Silver or quarantine, never both.
    If its latest version is rejected, remove its old Silver row.
    """

    latest_df.select("region_key").distinct().createOrReplaceTempView(
        "_latest_region_ids"
    )

    regions_out.select("region_key").distinct().createOrReplaceTempView(
        "_valid_region_ids"
    )

    spark.sql(f"""
        MERGE INTO {REGION_TABLE} AS t
        USING (
            SELECT l.region_key
            FROM _latest_region_ids l
            LEFT ANTI JOIN _valid_region_ids v
              ON l.region_key = v.region_key
        ) AS s
        ON t.region_key = s.region_key
        WHEN MATCHED THEN DELETE
    """)

    regions_out.createOrReplaceTempView("_regions_source")

    spark.sql(f"""
        MERGE INTO {REGION_TABLE} AS t
        USING _regions_source AS s
        ON t.region_key = s.region_key

        WHEN MATCHED AND (
            NOT (t.region_name <=> s.region_name)
            OR NOT (t.region_comment <=> s.region_comment)
        )
        THEN UPDATE SET
            t.region_name = s.region_name,
            t.region_comment = s.region_comment

        WHEN NOT MATCHED THEN INSERT (
            region_key, region_name, region_comment
        )
        VALUES (
            s.region_key, s.region_name, s.region_comment
        )
    """)

# ─────────────────────────────────────────────
# 7. MERGE quarantine and recover valid regions
# ─────────────────────────────────────────────
def merge_quarantine(
    latest_df: DataFrame,
    quarantine_out: DataFrame
):
    """
    Keep one quarantine record per rejected key.
    Remove quarantine records for regions now valid.
    """

    latest_df.select(
        F.coalesce(
            F.col("region_key_raw"), F.lit("<NULL>")
        ).alias("region_key")
    ).distinct().createOrReplaceTempView("_processed_ids")

    quarantine_out.select("region_key").distinct().createOrReplaceTempView(
        "_rejected_ids"
    )

    spark.sql(f"""
        DELETE FROM {QUARANTINE_TABLE}
        WHERE region_key IN (
            SELECT p.region_key
            FROM _processed_ids p
            LEFT ANTI JOIN _rejected_ids r
              ON p.region_key = r.region_key
        )
    """)

    quarantine_out.createOrReplaceTempView("_quarantine_source")

    spark.sql(f"""
        MERGE INTO {QUARANTINE_TABLE} AS t
        USING _quarantine_source AS s
        ON t.region_key = s.region_key

        WHEN MATCHED AND (
            NOT (t.rejection_reason <=> s.rejection_reason)
            OR NOT (t.raw_record <=> s.raw_record)
        )
        THEN UPDATE SET
            t.rejection_reason = s.rejection_reason,
            t.raw_record = s.raw_record,
            t.quarantined_at = s.quarantined_at

        WHEN NOT MATCHED THEN INSERT (
            region_key, rejection_reason, raw_record, quarantined_at
        )
        VALUES (
            s.region_key, s.rejection_reason, s.raw_record, s.quarantined_at
        )
    """)

# ─────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────
def main():

    # Read Bronze
    bronze_df = spark.read.table(BRONZE_TABLE)

    # Standardize first
    standardized_df = standardize_regions(bronze_df)

    # Keep latest version per region key
    latest_df = latest_region_versions(standardized_df)

    # Validate latest version only
    valid_df, invalid_df = validate_regions(latest_df)

    # Prepare output DataFrames
    regions_out, quarantine_out = prepare_outputs(valid_df, invalid_df)

    # Create tables once
    create_tables_if_not_exists()

    # Incremental target updates
    merge_regions(latest_df, regions_out)
    merge_quarantine(latest_df, quarantine_out)

    print("=" * 45)
    print("Latest region versions:", latest_df.count())
    print("Valid regions:", regions_out.count())
    print("Rejected regions:", quarantine_out.count())
    print("=" * 45)


if __name__ == "__main__":
    main()