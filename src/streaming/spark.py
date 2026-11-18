"""Spark Structured Streaming SQL analytics layer.

Pipeline: Kafka -> unwrap envelope -> parse payload -> row-level SQL rules
          -> tumbling-window aggregation (watermark) -> console sink.

Notes:
    - Structured Streaming does NOT support ROWS-based window functions;
      all time-based aggregation uses groupBy(window(...)) with a watermark.
    - No Python UDFs: everything is Spark SQL, so the job runs on executors
      without a Python runtime. ML inference runs in the Python streaming
      service (src/streaming/processor.py); this module is the distributed
      SQL analytics/rule layer.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from src.config.loader import load_config

logger = logging.getLogger(__name__)


def get_spark_session(app_name: Optional[str] = None, master: Optional[str] = None):
    try:
        from pyspark.sql import SparkSession
        config = load_config("spark")
        builder = (
            SparkSession.builder
            .appName(app_name or config.spark.app_name)
            .master(master or os.getenv("SPARK_MASTER") or config.spark.master)
            .config("spark.sql.shuffle_partitions", str(config.spark.sql.shuffle_partitions))
            .config("spark.driver.memory", config.spark.memory)
            .config("spark.sql.streaming.schemaInference", "true")
        )
        return builder.getOrCreate()
    except ImportError:
        logger.warning("PySpark not available")
        return None


def create_streaming_dataframe(spark, kafka_bootstrap: str, topic: str):
    try:
        return (
            spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", kafka_bootstrap)
            .option("subscribe", topic)
            .option("startingOffsets", "latest")
            .option("failOnDataLoss", "false")
            .option("maxOffsetsPerTrigger", 10000)
            .load()
        )
    except Exception as e:
        logger.error("Failed to create streaming DataFrame: %s", e)
        return None


def _payload_types():
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, IntegerType, LongType,
    )

    physical = StructType([
        StructField("device_id", StringType()),
        StructField("timestamp", StringType()),
        StructField("temperature", DoubleType()),
        StructField("humidity", DoubleType()),
        StructField("pressure", DoubleType()),
        StructField("voltage", DoubleType()),
        StructField("battery", DoubleType()),
        StructField("cpu_usage", DoubleType()),
        StructField("memory_usage", DoubleType()),
    ])
    network = StructType([
        StructField("device_id", StringType()),
        StructField("timestamp", StringType()),
        StructField("source_ip", StringType()),
        StructField("destination_ip", StringType()),
        StructField("source_port", IntegerType()),
        StructField("destination_port", IntegerType()),
        StructField("protocol", StringType()),
        StructField("packet_count", IntegerType()),
        StructField("packet_rate", DoubleType()),
        StructField("bytes", LongType()),
        StructField("network_bytes", LongType()),
    ])
    return physical, network


def _envelope_type(payload_type):
    from pyspark.sql.types import StructType, StructField, StringType
    return StructType([
        StructField("event_id", StringType()),
        StructField("timestamp", StringType()),
        StructField("source", StringType()),
        StructField("payload", payload_type),
    ])


def _unwrap_envelope(raw_df, payload_type):
    from pyspark.sql import functions as F

    envelope = _envelope_type(payload_type)
    return (
        raw_df.selectExpr("CAST(value AS STRING) AS json_str")
        .select(F.from_json(F.col("json_str"), envelope).alias("envelope"))
        .select(
            "envelope.payload.*",
            F.col("envelope.timestamp").alias("envelope_timestamp"),
        )
        .withColumn(
            "event_time",
            F.coalesce(
                F.to_timestamp(F.col("timestamp")),
                F.to_timestamp(F.col("envelope_timestamp")),
            ),
        )
        .filter(F.col("device_id").isNotNull())
    )


def parse_sensor_messages(raw_df):
    physical, _ = _payload_types()
    return _unwrap_envelope(raw_df, physical)


def parse_network_messages(raw_df):
    _, network = _payload_types()
    return _unwrap_envelope(raw_df, network)


def add_sensor_rules(sensor_df):
    """Row-level plausible-range rules (industrial sensor datasheet bounds)."""
    from pyspark.sql import functions as F

    return (
        sensor_df
        .withColumn(
            "temp_out_of_range",
            F.when((F.col("temperature") < -40) | (F.col("temperature") > 125), 1)
            .otherwise(0),
        )
        .withColumn(
            "voltage_out_of_range",
            F.when((F.col("voltage") < 0) | (F.col("voltage") > 30), 1)
            .otherwise(0),
        )
        .withColumn(
            "battery_out_of_range",
            F.when((F.col("battery") < 0) | (F.col("battery") > 100), 1)
            .otherwise(0),
        )
        .withColumn(
            "range_violation",
            F.greatest("temp_out_of_range", "voltage_out_of_range",
                       "battery_out_of_range"),
        )
    )


def add_network_rules(network_df, local_prefix: str = "10.0.0."):
    """Row-level cyber rules with documented, simulator-calibrated thresholds.

    - spoofed_source:    traffic from outside the local subnet
    - ddos_like:         volume far above normal envelope
                         (normal: packet_count<=100, packet_rate<=50, bytes<=10k)
    - bruteforce_like:   repeated SSH traffic rate above normal envelope
    """
    from pyspark.sql import functions as F

    return (
        network_df
        .withColumn(
            "spoofed_source",
            F.when(
                F.col("source_ip").isNotNull()
                & ~F.col("source_ip").startswith(local_prefix),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "ddos_like",
            F.when(
                (F.col("packet_count") > 1000)
                | (F.col("packet_rate") > 100.0)
                | (F.col("bytes") > 50000),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "bruteforce_like",
            F.when(
                (F.col("destination_port") == 22)
                & (F.col("packet_rate") > 50.0),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "rule_flag",
            F.greatest("spoofed_source", "ddos_like", "bruteforce_like"),
        )
    )


def run_physical_pipeline(spark, kafka_bootstrap: str, sensor_topic: str,
                          checkpoint_dir: str = "checkpoints/physical",
                          trigger: str = "5 seconds",
                          window_seconds: int = 5,
                          watermark: str = "10 seconds"):
    from pyspark.sql import functions as F

    raw_df = create_streaming_dataframe(spark, kafka_bootstrap, sensor_topic)
    if raw_df is None:
        logger.error("Cannot create physical streaming DataFrame")
        return None

    parsed = add_sensor_rules(parse_sensor_messages(raw_df))
    watermarked = parsed.withWatermark("event_time", watermark)

    window_spec = "%d seconds" % window_seconds
    agg = (
        watermarked
        .groupBy(F.window("event_time", window_spec), "device_id")
        .agg(
            F.count("*").alias("message_count"),
            F.avg("temperature").alias("avg_temperature"),
            F.stddev("temperature").alias("std_temperature"),
            F.min("temperature").alias("min_temperature"),
            F.max("temperature").alias("max_temperature"),
            F.min("voltage").alias("min_voltage"),
            F.avg("battery").alias("avg_battery"),
            F.sum("range_violation").alias("range_violations"),
        )
        .withColumn(
            "thermal_swing",
            F.col("max_temperature") - F.col("min_temperature"),
        )
        .withColumn(
            "anomaly_flag",
            (
                (F.col("range_violations") > 0)
                | (F.col("thermal_swing") > 10.0)
            ).cast("int"),
        )
    )

    query = (
        agg.writeStream
        .outputMode("update")
        .format("console")
        .option("checkpointLocation", checkpoint_dir)
        .trigger(processingTime=trigger)
        .start()
    )
    logger.info("Physical SQL pipeline started: %s", query.id)
    return query


def run_cyber_pipeline(spark, kafka_bootstrap: str, network_topic: str,
                       checkpoint_dir: str = "checkpoints/cyber",
                       trigger: str = "5 seconds",
                       window_seconds: int = 5,
                       watermark: str = "10 seconds"):
    from pyspark.sql import functions as F

    raw_df = create_streaming_dataframe(spark, kafka_bootstrap, network_topic)
    if raw_df is None:
        logger.error("Cannot create cyber streaming DataFrame")
        return None

    parsed = add_network_rules(parse_network_messages(raw_df))
    watermarked = parsed.withWatermark("event_time", watermark)

    window_spec = "%d seconds" % window_seconds
    agg = (
        watermarked
        .groupBy(F.window("event_time", window_spec), "device_id")
        .agg(
            F.count("*").alias("message_count"),
            F.sum("spoofed_source").alias("spoofed_count"),
            F.sum("ddos_like").alias("ddos_count"),
            F.sum("bruteforce_like").alias("bruteforce_count"),
            F.sum("rule_flag").alias("rule_hits"),
            F.max("packet_rate").alias("max_packet_rate"),
            F.max("bytes").alias("max_bytes"),
            F.approx_count_distinct("destination_port").alias("distinct_dest_ports"),
        )
        .withColumn(
            "threat_flag",
            (F.col("rule_hits") > 0).cast("int"),
        )
    )

    query = (
        agg.writeStream
        .outputMode("update")
        .format("console")
        .option("checkpointLocation", checkpoint_dir)
        .trigger(processingTime=trigger)
        .start()
    )
    logger.info("Cyber SQL pipeline started: %s", query.id)
    return query


def run_full_pipeline(kafka_bootstrap: str = "localhost:9092"):
    spark = get_spark_session()
    if spark is None:
        logger.error("Cannot create Spark session")
        return

    config = load_config("kafka")
    sensor_topic = config.kafka.topics.sensor_raw
    network_topic = config.kafka.topics.network_raw
    checkpoint_root = os.getenv("SPARK_CHECKPOINT_DIR", "checkpoints")

    logger.info("Starting Spark SQL streaming analytics")
    logger.info("Kafka: %s, Sensor topic: %s, Network topic: %s",
                kafka_bootstrap, sensor_topic, network_topic)

    physical_query = run_physical_pipeline(
        spark, kafka_bootstrap, sensor_topic,
        checkpoint_dir=os.path.join(checkpoint_root, "physical"),
    )
    cyber_query = run_cyber_pipeline(
        spark, kafka_bootstrap, network_topic,
        checkpoint_dir=os.path.join(checkpoint_root, "cyber"),
    )

    if physical_query:
        logger.info("Physical SQL pipeline active")
    if cyber_query:
        logger.info("Cyber SQL pipeline active")

    if physical_query or cyber_query:
        spark.streams.awaitAnyTermination()
    else:
        logger.error("No pipelines started")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Spark Structured Streaming SQL analytics")
    parser.add_argument("--kafka", default=None, help="Kafka bootstrap servers")
    parser.add_argument("--pipeline", choices=["physical", "cyber", "full"], default="full")
    args = parser.parse_args()

    spark = get_spark_session()
    if spark is None:
        print("PySpark not available. Install with: pip install pyspark")
        return

    config = load_config("kafka")
    kafka_bootstrap = args.kafka or os.getenv("KAFKA_BOOTSTRAP_SERVERS") \
        or config.kafka.bootstrap_servers
    checkpoint_root = os.getenv("SPARK_CHECKPOINT_DIR", "checkpoints")

    if args.pipeline in ("physical", "full"):
        run_physical_pipeline(
            spark, kafka_bootstrap, config.kafka.topics.sensor_raw,
            checkpoint_dir=os.path.join(checkpoint_root, "physical"),
        )
    if args.pipeline in ("cyber", "full"):
        run_cyber_pipeline(
            spark, kafka_bootstrap, config.kafka.topics.network_raw,
            checkpoint_dir=os.path.join(checkpoint_root, "cyber"),
        )

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
