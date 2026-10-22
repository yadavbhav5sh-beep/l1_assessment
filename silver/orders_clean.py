from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CATALOG = "tpch_dev"

BRONZE_TABLE = f"{CATALOG}.bronze.orders"

ORDERS_TABLE = f"{CATALOG}.silver.orders"
QUARANTINE_TABLE = f"{CATALOG}.silver.orders_quarantine"

# Referential integrity target
CUSTOMER_TABLE = f"{CATALOG}.silver.customer"

# Date range bounds (inclusive)
DATE_FROM = F.lit("1992-01-01").cast("date")
DATE_TO   = F.lit("1998-12-31").cast("date")

# ─────────────────────────────────────────────
# 1. Standardize orders (S1-S7)
# ─────────────────────────────────────────────
def standardize_orders(df: DataFrame) -> DataFrame:
    """
    Standardization happens before deduplication and validation.

    S1: order key trim; empty -> NULL
    S2: customer key trim; empty -> NULL
    S3: order status trim + uppercase; empty -> NULL
    S4: total price cast to decimal(18,2)
    S5: order date cast to date
    S6: order priority / clerk trim; empty -> NULL; ship priority cast to int
    S7: comment trim; empty -> NULL
    """

    # Preserve the complete original Bronze row as JSON.
    df = df.withColumn(
        "raw_record",
        F.to_json(F.struct(*[F.col(c) for c in df.columns]))
    )

    # S1: keep the raw text key for quarantine traceability.
    df = df.withColumn(
        "order_key_raw",
        F.when(
            F.trim(F.col("o_orderkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("o_orderkey").cast("string"))
        )
    )

    # S1b: strict integer parse - anything else becomes NULL
    df = df.withColumn(
        "order_key",
        F.when(
            F.col("order_key_raw").rlike("^-?\\d+$"),
            F.col("order_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S2: customer key - raw text + strict integer parse
    df = df.withColumn(
        "customer_key_raw",
        F.when(
            F.trim(F.col("o_custkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("o_custkey").cast("string"))
        )
    )

    df = df.withColumn(
        "customer_key",
        F.when(
            F.col("customer_key_raw").rlike("^-?\\d+$"),
            F.col("customer_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S3
    df = df.withColumn(
        "order_status",
        F.when(
            F.trim(F.col("o_orderstatus").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.upper(F.trim(F.col("o_orderstatus").cast("string")))
        )
    )

    # S4
    df = df.withColumn(
        "total_price",
        F.col("o_totalprice").cast("decimal(18,2)")
    )

    # S5
    df = df.withColumn(
        "order_date",
        F.col("o_orderdate").cast("date")
    )

    # S6
    df = df.withColumn(
        "order_priority",
        F.when(
            F.trim(F.col("o_orderpriority").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("o_orderpriority").cast("string"))
        )
    )

    df = df.withColumn(
        "clerk",
        F.when(
            F.trim(F.col("o_clerk").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("o_clerk").cast("string"))
        )
    )

    df = df.withColumn(
        "ship_priority",
        F.col("o_shippriority").cast("int")
    )

    # S7
    df = df.withColumn(
        "order_comment",
        F.when(
            F.trim(F.col("o_comment").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("o_comment").cast("string"))
        )
    )

    # Keep _ingestion_at until deduplication has completed.
    return df.drop("_source_file")


# ─────────────────────────────────────────────
# 2. Latest version per order key (C2)
# ─────────────────────────────────────────────
def latest_order_versions(df: DataFrame) -> DataFrame:
    """
    Validate only the most recently ingested version of each key.

    _ingestion_at is used as the ordering key.
    """

    window_spec = (
        Window
        .partitionBy("order_key_raw")
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
def validate_orders(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Reject a record if ANY rule fails. Every failed rule is
    collected into rejection_reason as a comma-separated list.

    R1 PRIMARY_KEY_NULL       order key missing or blank
    R2 PRIMARY_KEY_DUPLICATE  order key occurs more than once in the batch
    R3 INVALID_DATE_RANGE             order_date outside 1992-01-01 to 1998-12-31 (inclusive)
    R4 UNKNOWN_CUSTOMER       customer key does not exist in silver.customer
    """

    # Read valid customer keys from silver.customer for referential integrity.
    valid_customer_keys = (
        spark.table(CUSTOMER_TABLE)
        .select(F.col("customer_key").alias("_valid_customer_key"))
        .distinct()
    )

    # Left join to mark orders whose customer_key is missing from silver.customer.
    df = df.join(
        F.broadcast(valid_customer_keys),
        F.col("customer_key") == F.col("_valid_customer_key"),
        "left"
    )

    # How many rows share this key in this batch.
    duplicate_count = F.count(F.lit(1)).over(
        Window.partitionBy("order_key_raw")
    )

    checked = (
        df.withColumn("_key_dup_count", duplicate_count)
        # R1
        .withColumn(
            "_r1",
            F.col("order_key_raw").isNull()
        )
        # R2: only non-null keys can be duplicates.
        .withColumn(
            "_r2",
            F.col("order_key_raw").isNotNull()
            & (F.col("_key_dup_count") > 1)
        )
        # R3: order_date outside [1992-01-01, 1998-12-31] inclusive.
        .withColumn(
            "_r3",
            F.col("order_date").isNull()
            | (F.col("order_date") < DATE_FROM)
            | (F.col("order_date") > DATE_TO)
        )
        # R4: customer_key not found in silver.customer.
        .withColumn(
            "_r4",
            F.col("_valid_customer_key").isNull()
        )
        # Comma-separated list of EVERY failed rule for this record.
        .withColumn(
            "rejection_reason",
            F.concat_ws(
                ",",
                F.when(F.col("_r1"), F.lit("PRIMARY_KEY_NULL")),
                F.when(F.col("_r2"), F.lit("PRIMARY_KEY_DUPLICATE")),
                F.when(F.col("_r3"), F.lit("INVALID_DATE_RANGE")),
                F.when(F.col("_r4"), F.lit("UNKNOWN_CUSTOMER"))
            )
        )
        .drop("_key_dup_count", "_r1", "_r2", "_r3", "_r4",
              "_valid_customer_key")
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
        CREATE TABLE IF NOT EXISTS {ORDERS_TABLE} (
            order_key INT,
            customer_key INT,
            order_status STRING,
            total_price DECIMAL(18,2),
            order_date DATE,
            order_priority STRING,
            clerk STRING,
            ship_priority INT,
            order_comment STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {QUARANTINE_TABLE} (
            order_key STRING,
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

    orders_out = valid_df.select(
        "order_key",
        "customer_key",
        "order_status",
        "total_price",
        "order_date",
        "order_priority",
        "clerk",
        "ship_priority",
        "order_comment"
    )

    # NULL keys must stay joinable, so coalesce to a sentinel.
    quarantine_out = invalid_df.select(
        F.coalesce(
            F.col("order_key_raw"), F.lit("<NULL>")
        ).alias("order_key"),
        "rejection_reason",
        "raw_record",
        F.current_timestamp().alias("quarantined_at")
    )

    return orders_out, quarantine_out

# ─────────────────────────────────────────────
# 6. MERGE orders and remove stale invalid rows
# ─────────────────────────────────────────────
def merge_orders(
    latest_df: DataFrame,
    orders_out: DataFrame
):
    """
    An order must be in Silver or quarantine, never both.
    If its latest version is rejected, remove its old Silver row.
    """

    latest_df.select("order_key").distinct().createOrReplaceTempView(
        "_latest_order_ids"
    )

    orders_out.select("order_key").distinct().createOrReplaceTempView(
        "_valid_order_ids"
    )

    spark.sql(f"""
        MERGE INTO {ORDERS_TABLE} AS t
        USING (
            SELECT l.order_key
            FROM _latest_order_ids l
            LEFT ANTI JOIN _valid_order_ids v
              ON l.order_key = v.order_key
        ) AS s
        ON t.order_key = s.order_key
        WHEN MATCHED THEN DELETE
    """)

    orders_out.createOrReplaceTempView("_orders_source")

    spark.sql(f"""
        MERGE INTO {ORDERS_TABLE} AS t
        USING _orders_source AS s
        ON t.order_key = s.order_key

        WHEN MATCHED AND (
            NOT (t.customer_key <=> s.customer_key)
            OR NOT (t.order_status <=> s.order_status)
            OR NOT (t.total_price <=> s.total_price)
            OR NOT (t.order_date <=> s.order_date)
            OR NOT (t.order_priority <=> s.order_priority)
            OR NOT (t.clerk <=> s.clerk)
            OR NOT (t.ship_priority <=> s.ship_priority)
            OR NOT (t.order_comment <=> s.order_comment)
        )
        THEN UPDATE SET
            t.customer_key = s.customer_key,
            t.order_status = s.order_status,
            t.total_price = s.total_price,
            t.order_date = s.order_date,
            t.order_priority = s.order_priority,
            t.clerk = s.clerk,
            t.ship_priority = s.ship_priority,
            t.order_comment = s.order_comment

        WHEN NOT MATCHED THEN INSERT (
            order_key, customer_key, order_status, total_price,
            order_date, order_priority, clerk, ship_priority, order_comment
        )
        VALUES (
            s.order_key, s.customer_key, s.order_status, s.total_price,
            s.order_date, s.order_priority, s.clerk, s.ship_priority, s.order_comment
        )
    """)

# ─────────────────────────────────────────────
# 7. MERGE quarantine and recover valid orders
# ─────────────────────────────────────────────
def merge_quarantine(
    latest_df: DataFrame,
    quarantine_out: DataFrame
):
    """
    Keep one quarantine record per rejected key.
    Remove quarantine records for orders now valid.
    """

    latest_df.select(
        F.coalesce(
            F.col("order_key_raw"), F.lit("<NULL>")
        ).alias("order_key")
    ).distinct().createOrReplaceTempView("_processed_ids")

    quarantine_out.select("order_key").distinct().createOrReplaceTempView(
        "_rejected_ids"
    )

    spark.sql(f"""
        DELETE FROM {QUARANTINE_TABLE}
        WHERE order_key IN (
            SELECT p.order_key
            FROM _processed_ids p
            LEFT ANTI JOIN _rejected_ids r
              ON p.order_key = r.order_key
        )
    """)

    quarantine_out.createOrReplaceTempView("_quarantine_source")

    spark.sql(f"""
        MERGE INTO {QUARANTINE_TABLE} AS t
        USING _quarantine_source AS s
        ON t.order_key = s.order_key

        WHEN MATCHED AND (
            NOT (t.rejection_reason <=> s.rejection_reason)
            OR NOT (t.raw_record <=> s.raw_record)
        )
        THEN UPDATE SET
            t.rejection_reason = s.rejection_reason,
            t.raw_record = s.raw_record,
            t.quarantined_at = s.quarantined_at

        WHEN NOT MATCHED THEN INSERT (
            order_key, rejection_reason, raw_record, quarantined_at
        )
        VALUES (
            s.order_key, s.rejection_reason, s.raw_record, s.quarantined_at
        )
    """)

# ─────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────
def main():

    # Read Bronze
    bronze_df = spark.read.table(BRONZE_TABLE)

    # Standardize first
    standardized_df = standardize_orders(bronze_df)

    # Keep latest version per order key
    latest_df = latest_order_versions(standardized_df)

    # Validate latest version only
    valid_df, invalid_df = validate_orders(latest_df)

    # Prepare output DataFrames
    orders_out, quarantine_out = prepare_outputs(valid_df, invalid_df)

    # Create tables once
    create_tables_if_not_exists()

    # Incremental target updates
    merge_orders(latest_df, orders_out)
    merge_quarantine(latest_df, quarantine_out)

    print("=" * 45)
    print("Latest order versions:", latest_df.count())
    print("Valid orders:", orders_out.count())
    print("Rejected orders:", quarantine_out.count())
    print("=" * 45)


if __name__ == "__main__":
    main()