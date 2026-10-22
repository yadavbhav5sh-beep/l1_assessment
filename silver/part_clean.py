from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CATALOG = "tpch_dev"

BRONZE_TABLE = f"{CATALOG}.bronze.part"

PART_TABLE = f"{CATALOG}.silver.part"
QUARANTINE_TABLE = f"{CATALOG}.silver.part_quarantine"

# ─────────────────────────────────────────────
# 1. Standardize parts (S1-S5)
# ─────────────────────────────────────────────
def standardize_parts(df: DataFrame) -> DataFrame:
    """
    Standardization happens before deduplication and validation.

    S1: part key trim; empty -> NULL
    S2: part name trim; empty -> NULL
    S3: mfgr / brand / type / container trim; empty -> NULL
    S4: size cast to int; retail price cast to decimal(18,2)
    S5: comment trim; empty -> NULL
    """

    # Preserve the complete original Bronze row as JSON.
    df = df.withColumn(
        "raw_record",
        F.to_json(F.struct(*[F.col(c) for c in df.columns]))
    )

    # S1: keep the raw text key for quarantine traceability.
    df = df.withColumn(
        "part_key_raw",
        F.when(
            F.trim(F.col("p_partkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("p_partkey").cast("string"))
        )
    )

    # S1b: strict integer parse - anything else becomes NULL
    df = df.withColumn(
        "part_key",
        F.when(
            F.col("part_key_raw").rlike("^-?\\d+$"),
            F.col("part_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S2
    df = df.withColumn(
        "part_name",
        F.when(
            F.trim(F.col("p_name").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("p_name").cast("string"))
        )
    )

    # S3
    df = df.withColumn(
        "part_mfgr",
        F.when(
            F.trim(F.col("p_mfgr").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("p_mfgr").cast("string"))
        )
    )

    df = df.withColumn(
        "part_brand",
        F.when(
            F.trim(F.col("p_brand").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("p_brand").cast("string"))
        )
    )

    df = df.withColumn(
        "part_type",
        F.when(
            F.trim(F.col("p_type").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("p_type").cast("string"))
        )
    )

    df = df.withColumn(
        "part_container",
        F.when(
            F.trim(F.col("p_container").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("p_container").cast("string"))
        )
    )

    # S4: size cast to int, retail price cast to decimal(18,2)
    df = df.withColumn(
        "part_size",
        F.col("p_size").cast("int")
    )

    df = df.withColumn(
        "part_retailprice",
        F.col("p_retailprice").cast("decimal(18,2)")
    )

    # S5
    df = df.withColumn(
        "part_comment",
        F.when(
            F.trim(F.col("p_comment").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("p_comment").cast("string"))
        )
    )

    # Keep _ingestion_at until deduplication has completed.
    return df.drop("_source_file")


# ─────────────────────────────────────────────
# 2. Latest version per part key (C2)
# ─────────────────────────────────────────────
def latest_part_versions(df: DataFrame) -> DataFrame:
    """
    Validate only the most recently ingested version of each key.

    _ingestion_at is used as the ordering key.
    """

    window_spec = (
        Window
        .partitionBy("part_key_raw")
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
def validate_parts(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Reject a record if ANY rule fails. Every failed rule is
    collected into rejection_reason as a comma-separated list.

    R1 PRIMARY_KEY_NULL       part key missing or blank
    R2 PRIMARY_KEY_DUPLICATE  part key occurs more than once in the batch
    """

    # How many rows share this key in this batch.
    duplicate_count = F.count(F.lit(1)).over(
        Window.partitionBy("part_key_raw")
    )

    checked = (
        df.withColumn("_key_dup_count", duplicate_count)
        # R1
        .withColumn(
            "_r1",
            F.col("part_key_raw").isNull()
        )
        # R2: only non-null keys can be duplicates.
        .withColumn(
            "_r2",
            F.col("part_key_raw").isNotNull()
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
        CREATE TABLE IF NOT EXISTS {PART_TABLE} (
            part_key INT,
            part_name STRING,
            part_mfgr STRING,
            part_brand STRING,
            part_type STRING,
            part_size INT,
            part_container STRING,
            part_retailprice DECIMAL(18,2),
            part_comment STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {QUARANTINE_TABLE} (
            part_key STRING,
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

    parts_out = valid_df.select(
        "part_key",
        "part_name",
        "part_mfgr",
        "part_brand",
        "part_type",
        "part_size",
        "part_container",
        "part_retailprice",
        "part_comment"
    )

    # NULL keys must stay joinable, so coalesce to a sentinel.
    quarantine_out = invalid_df.select(
        F.coalesce(
            F.col("part_key_raw"), F.lit("<NULL>")
        ).alias("part_key"),
        "rejection_reason",
        "raw_record",
        F.current_timestamp().alias("quarantined_at")
    )

    return parts_out, quarantine_out

# ─────────────────────────────────────────────
# 6. MERGE parts and remove stale invalid rows
# ─────────────────────────────────────────────
def merge_parts(
    latest_df: DataFrame,
    parts_out: DataFrame
):
    """
    A part must be in Silver or quarantine, never both.
    If its latest version is rejected, remove its old Silver row.
    """

    latest_df.select("part_key").distinct().createOrReplaceTempView(
        "_latest_part_ids"
    )

    parts_out.select("part_key").distinct().createOrReplaceTempView(
        "_valid_part_ids"
    )

    spark.sql(f"""
        MERGE INTO {PART_TABLE} AS t
        USING (
            SELECT l.part_key
            FROM _latest_part_ids l
            LEFT ANTI JOIN _valid_part_ids v
              ON l.part_key = v.part_key
        ) AS s
        ON t.part_key = s.part_key
        WHEN MATCHED THEN DELETE
    """)

    parts_out.createOrReplaceTempView("_parts_source")

    spark.sql(f"""
        MERGE INTO {PART_TABLE} AS t
        USING _parts_source AS s
        ON t.part_key = s.part_key

        WHEN MATCHED AND (
            NOT (t.part_name <=> s.part_name)
            OR NOT (t.part_mfgr <=> s.part_mfgr)
            OR NOT (t.part_brand <=> s.part_brand)
            OR NOT (t.part_type <=> s.part_type)
            OR NOT (t.part_size <=> s.part_size)
            OR NOT (t.part_container <=> s.part_container)
            OR NOT (t.part_retailprice <=> s.part_retailprice)
            OR NOT (t.part_comment <=> s.part_comment)
        )
        THEN UPDATE SET
            t.part_name = s.part_name,
            t.part_mfgr = s.part_mfgr,
            t.part_brand = s.part_brand,
            t.part_type = s.part_type,
            t.part_size = s.part_size,
            t.part_container = s.part_container,
            t.part_retailprice = s.part_retailprice,
            t.part_comment = s.part_comment

        WHEN NOT MATCHED THEN INSERT (
            part_key, part_name, part_mfgr, part_brand, part_type,
            part_size, part_container, part_retailprice, part_comment
        )
        VALUES (
            s.part_key, s.part_name, s.part_mfgr, s.part_brand, s.part_type,
            s.part_size, s.part_container, s.part_retailprice, s.part_comment
        )
    """)

# ─────────────────────────────────────────────
# 7. MERGE quarantine and recover valid parts
# ─────────────────────────────────────────────
def merge_quarantine(
    latest_df: DataFrame,
    quarantine_out: DataFrame
):
    """
    Keep one quarantine record per rejected key.
    Remove quarantine records for parts now valid.
    """

    latest_df.select(
        F.coalesce(
            F.col("part_key_raw"), F.lit("<NULL>")
        ).alias("part_key")
    ).distinct().createOrReplaceTempView("_processed_ids")

    quarantine_out.select("part_key").distinct().createOrReplaceTempView(
        "_rejected_ids"
    )

    spark.sql(f"""
        DELETE FROM {QUARANTINE_TABLE}
        WHERE part_key IN (
            SELECT p.part_key
            FROM _processed_ids p
            LEFT ANTI JOIN _rejected_ids r
              ON p.part_key = r.part_key
        )
    """)

    quarantine_out.createOrReplaceTempView("_quarantine_source")

    spark.sql(f"""
        MERGE INTO {QUARANTINE_TABLE} AS t
        USING _quarantine_source AS s
        ON t.part_key = s.part_key

        WHEN MATCHED AND (
            NOT (t.rejection_reason <=> s.rejection_reason)
            OR NOT (t.raw_record <=> s.raw_record)
        )
        THEN UPDATE SET
            t.rejection_reason = s.rejection_reason,
            t.raw_record = s.raw_record,
            t.quarantined_at = s.quarantined_at

        WHEN NOT MATCHED THEN INSERT (
            part_key, rejection_reason, raw_record, quarantined_at
        )
        VALUES (
            s.part_key, s.rejection_reason, s.raw_record, s.quarantined_at
        )
    """)

# ─────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────
def main():

    # Read Bronze
    bronze_df = spark.read.table(BRONZE_TABLE)

    # Standardize first
    standardized_df = standardize_parts(bronze_df)

    # Keep latest version per part key
    latest_df = latest_part_versions(standardized_df)

    # Validate latest version only
    valid_df, invalid_df = validate_parts(latest_df)

    # Prepare output DataFrames
    parts_out, quarantine_out = prepare_outputs(valid_df, invalid_df)

    # Create tables once
    create_tables_if_not_exists()

    # Incremental target updates
    merge_parts(latest_df, parts_out)
    merge_quarantine(latest_df, quarantine_out)

    print("=" * 45)
    print("Latest part versions:", latest_df.count())
    print("Valid parts:", parts_out.count())
    print("Rejected parts:", quarantine_out.count())
    print("=" * 45)


if __name__ == "__main__":
    main()