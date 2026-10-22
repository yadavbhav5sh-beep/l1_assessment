from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CATALOG = "tpch_dev"

BRONZE_TABLE = f"{CATALOG}.bronze.lineitem"

LINEITEM_TABLE = f"{CATALOG}.silver.lineitem"
QUARANTINE_TABLE = f"{CATALOG}.silver.lineitem_quarantine"

# Referential integrity targets
ORDERS_TABLE = f"{CATALOG}.silver.orders"
PART_TABLE = f"{CATALOG}.silver.part"
SUPPLIER_TABLE = f"{CATALOG}.silver.supplier"

# Date range bounds (inclusive)
DATE_FROM = F.lit("1992-01-01").cast("date")
DATE_TO   = F.lit("1998-12-31").cast("date")

# ─────────────────────────────────────────────
# 1. Standardize lineitems (S1-S8)
# ─────────────────────────────────────────────
def standardize_lineitems(df: DataFrame) -> DataFrame:
    """
    Standardization happens before deduplication and validation.

    S1: order key trim; empty -> NULL
    S2: part key trim; empty -> NULL
    S3: supplier key trim; empty -> NULL
    S4: line number trim; empty -> NULL
    S5: quantity / extended price / discount / tax cast to decimal(18,2)
    S6: return flag / line status trim + uppercase; empty -> NULL
    S7: ship date / commit date / receipt date cast to date
    S8: ship instruct / ship mode / comment trim; empty -> NULL
    """

    # Preserve the complete original Bronze row as JSON.
    df = df.withColumn(
        "raw_record",
        F.to_json(F.struct(*[F.col(c) for c in df.columns]))
    )

    # S1: order key - raw text + strict integer parse
    df = df.withColumn(
        "order_key_raw",
        F.when(
            F.trim(F.col("l_orderkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("l_orderkey").cast("string"))
        )
    )

    df = df.withColumn(
        "order_key",
        F.when(
            F.col("order_key_raw").rlike("^-?\\d+$"),
            F.col("order_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S2: part key - raw text + strict integer parse
    df = df.withColumn(
        "part_key_raw",
        F.when(
            F.trim(F.col("l_partkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("l_partkey").cast("string"))
        )
    )

    df = df.withColumn(
        "part_key",
        F.when(
            F.col("part_key_raw").rlike("^-?\\d+$"),
            F.col("part_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S3: supplier key - raw text + strict integer parse
    df = df.withColumn(
        "supplier_key_raw",
        F.when(
            F.trim(F.col("l_suppkey").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("l_suppkey").cast("string"))
        )
    )

    df = df.withColumn(
        "supplier_key",
        F.when(
            F.col("supplier_key_raw").rlike("^-?\\d+$"),
            F.col("supplier_key_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S4: line number - raw text + strict integer parse
    df = df.withColumn(
        "line_number_raw",
        F.when(
            F.trim(F.col("l_linenumber").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("l_linenumber").cast("string"))
        )
    )

    df = df.withColumn(
        "line_number",
        F.when(
            F.col("line_number_raw").rlike("^-?\\d+$"),
            F.col("line_number_raw").cast("int")
        ).otherwise(F.lit(None).cast("int"))
    )

    # S5: numeric casts
    df = df.withColumn("quantity", F.col("l_quantity").cast("decimal(18,2)"))
    df = df.withColumn("extended_price", F.col("l_extendedprice").cast("decimal(18,2)"))
    df = df.withColumn("discount", F.col("l_discount").cast("decimal(18,2)"))
    df = df.withColumn("tax", F.col("l_tax").cast("decimal(18,2)"))

    # S6: status flags
    df = df.withColumn(
        "return_flag",
        F.when(
            F.trim(F.col("l_returnflag").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.upper(F.trim(F.col("l_returnflag").cast("string")))
        )
    )

    df = df.withColumn(
        "line_status",
        F.when(
            F.trim(F.col("l_linestatus").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.upper(F.trim(F.col("l_linestatus").cast("string")))
        )
    )

    # S7: date casts
    df = df.withColumn("ship_date", F.col("l_shipdate").cast("date"))
    df = df.withColumn("commit_date", F.col("l_commitdate").cast("date"))
    df = df.withColumn("receipt_date", F.col("l_receiptdate").cast("date"))

    # S8: string columns
    df = df.withColumn(
        "ship_instruct",
        F.when(
            F.trim(F.col("l_shipinstruct").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("l_shipinstruct").cast("string"))
        )
    )

    df = df.withColumn(
        "ship_mode",
        F.when(
            F.trim(F.col("l_shipmode").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.upper(F.trim(F.col("l_shipmode").cast("string")))
        )
    )

    df = df.withColumn(
        "line_comment",
        F.when(
            F.trim(F.col("l_comment").cast("string")) == "",
            F.lit(None).cast("string")
        ).otherwise(
            F.trim(F.col("l_comment").cast("string"))
        )
    )

    # Keep _ingestion_at until deduplication has completed.
    return df.drop("_source_file")


# ─────────────────────────────────────────────
# 2. Latest version per composite key (C2)
# ─────────────────────────────────────────────
def latest_lineitem_versions(df: DataFrame) -> DataFrame:
    """
    Validate only the most recently ingested version of each
    composite key (order_key_raw + line_number_raw).

    _ingestion_at is used as the ordering key.
    """

    window_spec = (
        Window
        .partitionBy("order_key_raw", "line_number_raw")
        .orderBy(F.col("_ingestion_at").desc_nulls_last())
    )

    return (
        df
        .withColumn("_rn", F.row_number().over(window_spec))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

# ─────────────────────────────────────────────
# 3. Data quality rules (R1-R9)
# ─────────────────────────────────────────────
def validate_lineitems(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Reject a record if ANY rule fails. Every failed rule is
    collected into rejection_reason as a comma-separated list.

    R1 PRIMARY_KEY_NULL_ORDER     order key missing or blank
    R2 PRIMARY_KEY_NULL_LINE      line number missing or blank
    R3 PRIMARY_KEY_DUPLICATE     (order_key + line_number) occurs more than once
    R4 NUMERIC_RANGE             quantity <= 0, extended_price <= 0,
                                 discount < 0 or > 0.10, tax < 0 or > 0.08
    R5 DATE_RANGE                ship_date outside 1992-01-01 to 1998-12-31
    R6 DATE_ORDER                ship_date < order_date from silver.orders
    R7 UNKNOWN_ORDER              order key not in silver.orders
    R8 UNKNOWN_PART               part key not in silver.part
    R9 UNKNOWN_SUPPLIER          supplier key not in silver.supplier
    """

    # Read valid order keys + order dates from silver.orders.
    valid_orders = (
        spark.table(ORDERS_TABLE)
        .select(
            F.col("order_key").alias("_valid_order_key"),
            F.col("order_date").alias("_order_date")
        )
        .distinct()
    )

    # Read valid part keys from silver.part.
    valid_part_keys = (
        spark.table(PART_TABLE)
        .select(F.col("part_key").alias("_valid_part_key"))
        .distinct()
    )

    # Read valid supplier keys from silver.supplier.
    valid_supplier_keys = (
        spark.table(SUPPLIER_TABLE)
        .select(F.col("supplier_key").alias("_valid_supplier_key"))
        .distinct()
    )

    # Left joins for referential integrity + order date lookup.
    df = df.join(
        F.broadcast(valid_orders),
        F.col("order_key") == F.col("_valid_order_key"),
        "left"
    ).join(
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
        Window.partitionBy("order_key_raw", "line_number_raw")
    )

    checked = (
        df.withColumn("_key_dup_count", duplicate_count)
        # R1: order key null
        .withColumn(
            "_r1",
            F.col("order_key_raw").isNull()
        )
        # R2: line number null
        .withColumn(
            "_r2",
            F.col("line_number_raw").isNull()
        )
        # R3: composite key duplicate (only when both keys are non-null)
        .withColumn(
            "_r3",
            F.col("order_key_raw").isNotNull()
            & F.col("line_number_raw").isNotNull()
            & (F.col("_key_dup_count") > 1)
        )
        # R4: numeric range - quantity > 0, extended_price > 0,
        #     discount 0-0.10, tax 0-0.08 (all inclusive bounds)
        .withColumn(
            "_r4",
            F.col("quantity").isNull()
            | (F.col("quantity") <= 0)
            | F.col("extended_price").isNull()
            | (F.col("extended_price") <= 0)
            | F.col("discount").isNull()
            | (F.col("discount") < 0)
            | (F.col("discount") > 0.10)
            | F.col("tax").isNull()
            | (F.col("tax") < 0)
            | (F.col("tax") > 0.08)
        )
        # R5: ship_date outside [1992-01-01, 1998-12-31] inclusive
        .withColumn(
            "_r5",
            F.col("ship_date").isNull()
            | (F.col("ship_date") < DATE_FROM)
            | (F.col("ship_date") > DATE_TO)
        )
        # R6: ship_date < order_date (only when both are present)
        .withColumn(
            "_r6",
            F.col("_order_date").isNotNull()
            & F.col("ship_date").isNotNull()
            & (F.col("ship_date") < F.col("_order_date"))
        )
        # R7: order key not in silver.orders
        .withColumn(
            "_r7",
            F.col("_valid_order_key").isNull()
        )
        # R8: part key not in silver.part
        .withColumn(
            "_r8",
            F.col("_valid_part_key").isNull()
        )
        # R9: supplier key not in silver.supplier
        .withColumn(
            "_r9",
            F.col("_valid_supplier_key").isNull()
        )
        # Comma-separated list of EVERY failed rule for this record.
        .withColumn(
            "rejection_reason",
            F.concat_ws(
                ",",
                F.when(F.col("_r1"), F.lit("NULL_ORDER")),
                F.when(F.col("_r2"), F.lit("NULL_LINE")),
                F.when(F.col("_r3"), F.lit("PRIMARY_KEY_DUPLICATE")),
                F.when(F.col("_r4"), F.lit("INVALID_NUMERIC_RANGE")),
                F.when(F.col("_r5"), F.lit("INVALID_DATE_RANGE")),
                F.when(F.col("_r6"), F.lit("INVALID_DATE_ORDER")),
                F.when(F.col("_r7"), F.lit("UNKNOWN_ORDER")),
                F.when(F.col("_r8"), F.lit("UNKNOWN_PART")),
                F.when(F.col("_r9"), F.lit("UNKNOWN_SUPPLIER"))
            )
        )
        .drop("_key_dup_count", "_r1", "_r2", "_r3", "_r4", "_r5",
              "_r6", "_r7", "_r8", "_r9",
              "_valid_order_key", "_order_date",
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
        CREATE TABLE IF NOT EXISTS {LINEITEM_TABLE} (
            order_key INT,
            part_key INT,
            supplier_key INT,
            line_number INT,
            quantity DECIMAL(18,2),
            extended_price DECIMAL(18,2),
            discount DECIMAL(18,2),
            tax DECIMAL(18,2),
            return_flag STRING,
            line_status STRING,
            ship_date DATE,
            commit_date DATE,
            receipt_date DATE,
            ship_instruct STRING,
            ship_mode STRING,
            line_comment STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {QUARANTINE_TABLE} (
            lineitem_key STRING,
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

    lineitems_out = valid_df.select(
        "order_key",
        "part_key",
        "supplier_key",
        "line_number",
        "quantity",
        "extended_price",
        "discount",
        "tax",
        "return_flag",
        "line_status",
        "ship_date",
        "commit_date",
        "receipt_date",
        "ship_instruct",
        "ship_mode",
        "line_comment"
    )

    # Build a composite key string for quarantine traceability.
    # NULL keys must stay joinable, so coalesce to a sentinel.
    quarantine_out = invalid_df.select(
        F.concat_ws(
            "|",
            F.coalesce(F.col("order_key_raw"), F.lit("<NULL>")),
            F.coalesce(F.col("line_number_raw"), F.lit("<NULL>"))
        ).alias("lineitem_key"),
        "rejection_reason",
        "raw_record",
        F.current_timestamp().alias("quarantined_at")
    )

    return lineitems_out, quarantine_out

# ─────────────────────────────────────────────
# 6. MERGE lineitems and remove stale invalid rows
# ─────────────────────────────────────────────
def merge_lineitems(
    latest_df: DataFrame,
    lineitems_out: DataFrame
):
    """
    A lineitem must be in Silver or quarantine, never both.
    If its latest version is rejected, remove its old Silver row.
    """

    latest_df.select("order_key", "line_number").distinct().createOrReplaceTempView(
        "_latest_lineitem_ids"
    )

    lineitems_out.select("order_key", "line_number").distinct().createOrReplaceTempView(
        "_valid_lineitem_ids"
    )

    spark.sql(f"""
        MERGE INTO {LINEITEM_TABLE} AS t
        USING (
            SELECT l.order_key, l.line_number
            FROM _latest_lineitem_ids l
            LEFT ANTI JOIN _valid_lineitem_ids v
              ON l.order_key = v.order_key
             AND l.line_number = v.line_number
        ) AS s
        ON t.order_key = s.order_key
        AND t.line_number = s.line_number
        WHEN MATCHED THEN DELETE
    """)

    lineitems_out.createOrReplaceTempView("_lineitems_source")

    spark.sql(f"""
        MERGE INTO {LINEITEM_TABLE} AS t
        USING _lineitems_source AS s
        ON t.order_key = s.order_key
        AND t.line_number = s.line_number

        WHEN MATCHED AND (
            NOT (t.part_key <=> s.part_key)
            OR NOT (t.supplier_key <=> s.supplier_key)
            OR NOT (t.quantity <=> s.quantity)
            OR NOT (t.extended_price <=> s.extended_price)
            OR NOT (t.discount <=> s.discount)
            OR NOT (t.tax <=> s.tax)
            OR NOT (t.return_flag <=> s.return_flag)
            OR NOT (t.line_status <=> s.line_status)
            OR NOT (t.ship_date <=> s.ship_date)
            OR NOT (t.commit_date <=> s.commit_date)
            OR NOT (t.receipt_date <=> s.receipt_date)
            OR NOT (t.ship_instruct <=> s.ship_instruct)
            OR NOT (t.ship_mode <=> s.ship_mode)
            OR NOT (t.line_comment <=> s.line_comment)
        )
        THEN UPDATE SET
            t.part_key = s.part_key,
            t.supplier_key = s.supplier_key,
            t.quantity = s.quantity,
            t.extended_price = s.extended_price,
            t.discount = s.discount,
            t.tax = s.tax,
            t.return_flag = s.return_flag,
            t.line_status = s.line_status,
            t.ship_date = s.ship_date,
            t.commit_date = s.commit_date,
            t.receipt_date = s.receipt_date,
            t.ship_instruct = s.ship_instruct,
            t.ship_mode = s.ship_mode,
            t.line_comment = s.line_comment

        WHEN NOT MATCHED THEN INSERT (
            order_key, part_key, supplier_key, line_number,
            quantity, extended_price, discount, tax,
            return_flag, line_status, ship_date, commit_date, receipt_date,
            ship_instruct, ship_mode, line_comment
        )
        VALUES (
            s.order_key, s.part_key, s.supplier_key, s.line_number,
            s.quantity, s.extended_price, s.discount, s.tax,
            s.return_flag, s.line_status, s.ship_date, s.commit_date, s.receipt_date,
            s.ship_instruct, s.ship_mode, s.line_comment
        )
    """)

# ─────────────────────────────────────────────
# 7. MERGE quarantine and recover valid lineitems
# ─────────────────────────────────────────────
def merge_quarantine(
    latest_df: DataFrame,
    quarantine_out: DataFrame
):
    """
    Keep one quarantine record per rejected composite key.
    Remove quarantine records for lineitems now valid.
    """

    latest_df.select(
        F.concat_ws(
            "|",
            F.coalesce(F.col("order_key_raw"), F.lit("<NULL>")),
            F.coalesce(F.col("line_number_raw"), F.lit("<NULL>"))
        ).alias("lineitem_key")
    ).distinct().createOrReplaceTempView("_processed_ids")

    quarantine_out.select("lineitem_key").distinct().createOrReplaceTempView(
        "_rejected_ids"
    )

    spark.sql(f"""
        DELETE FROM {QUARANTINE_TABLE}
        WHERE lineitem_key IN (
            SELECT p.lineitem_key
            FROM _processed_ids p
            LEFT ANTI JOIN _rejected_ids r
              ON p.lineitem_key = r.lineitem_key
        )
    """)

    quarantine_out.createOrReplaceTempView("_quarantine_source")

    spark.sql(f"""
        MERGE INTO {QUARANTINE_TABLE} AS t
        USING _quarantine_source AS s
        ON t.lineitem_key = s.lineitem_key

        WHEN MATCHED AND (
            NOT (t.rejection_reason <=> s.rejection_reason)
            OR NOT (t.raw_record <=> s.raw_record)
        )
        THEN UPDATE SET
            t.rejection_reason = s.rejection_reason,
            t.raw_record = s.raw_record,
            t.quarantined_at = s.quarantined_at

        WHEN NOT MATCHED THEN INSERT (
            lineitem_key, rejection_reason, raw_record, quarantined_at
        )
        VALUES (
            s.lineitem_key, s.rejection_reason, s.raw_record, s.quarantined_at
        )
    """)

# ─────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────
def main():

    # Read Bronze
    bronze_df = spark.read.table(BRONZE_TABLE)

    # Standardize first
    standardized_df = standardize_lineitems(bronze_df)

    # Keep latest version per composite key
    latest_df = latest_lineitem_versions(standardized_df)

    # Validate latest version only
    valid_df, invalid_df = validate_lineitems(latest_df)

    # Prepare output DataFrames
    lineitems_out, quarantine_out = prepare_outputs(valid_df, invalid_df)

    # Create tables once
    create_tables_if_not_exists()

    # Incremental target updates
    merge_lineitems(latest_df, lineitems_out)
    merge_quarantine(latest_df, quarantine_out)

    print("=" * 45)
    print("Latest lineitem versions:", latest_df.count())
    print("Valid lineitems:", lineitems_out.count())
    print("Rejected lineitems:", quarantine_out.count())
    print("=" * 45)


if __name__ == "__main__":
    main()