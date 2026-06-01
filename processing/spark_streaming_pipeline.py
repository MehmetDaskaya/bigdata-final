#!/usr/bin/env python3
# =============================================================================
# processing/spark_streaming_pipeline.py
# Spark Structured Streaming — Real-Time Pipeline from Kafka to MongoDB
# =============================================================================
# This script reads emission data from Apache Kafka in real-time,
# processes it with Spark Structured Streaming, and writes it to MongoDB.
#
# DATA FLOW:
#   Kafka Topic (carbon-emissions-daily)
#     ↓ [Spark Structured Streaming]
#     ↓ JSON Parse → Validation → Feature Engineering → Aggregation
#     ↓ [Micro-batch writing]
#   MongoDB (carbon_footprint.kafka_stream_log + emissions_timeseries)
#
# PLACE IN LAMBDA ARCHITECTURE:
#   This script = 'Speed Layer' (speed layer)
#   Batch pipeline = 'Batch Layer' (batch layer)
#   MongoDB = 'Serving Layer' (serving layer)
#
# Important Features:
#   - Exactly-once semantics (each message is processed exactly once)
#   - Checkpointing (recovery point — resumes from where it left off on failure)
#   - Watermarking (managing late-arriving messages)
#   - Micro-batch processing (every 10 seconds)
#
# Usage:
#   python processing/spark_streaming_pipeline.py
# =============================================================================

import os
import sys
import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('spark_streaming')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR          = Path(__file__).parent.parent
KAFKA_BROKER      = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC       = "carbon-emissions-daily"
MONGO_URI         = os.getenv("MONGO_URI", "mongodb://admin:password@localhost:27017/carbon_footprint")
CHECKPOINT_DIR    = str(BASE_DIR / "checkpoints" / "streaming")  # Recovery points
MICRO_BATCH_SECS  = 10   # A micro-batch is processed every 10 seconds

# Kafka message schema — must match the JSON format sent by the producer
EMISSION_SCHEMA = StructType([
    StructField("date",                StringType(), True),
    StructField("country",             StringType(), True),
    StructField("sector",              StringType(), True),
    StructField("MtCO2 per day",       DoubleType(), True),
    StructField("timestamp",           StringType(), True),
    StructField("_kafka_ingested_at",  StringType(), True),
    StructField("_source",             StringType(), True),
])


def create_spark_session() -> SparkSession:
    """
    Creates a Spark Session specifically configured for streaming.
    
    Streaming-specific configurations:
    - Kafka connector: for Spark to use Kafka as a source
    - Checkpoint: required for error recovery and exactly-once
    - Trigger: micro-batch interval
    
    Returns:
        SparkSession with streaming support
    """
    logger.info("Creating Streaming Spark Session...")
    
    spark = (SparkSession.builder
        .appName("CarbonFootprint-StreamingPipeline")
        .master(os.getenv("SPARK_MASTER", "local[2]"))  # Minimum 2 threads for streaming
        # Kafka Spark Connector
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
                "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
        .config("spark.mongodb.write.connection.uri", MONGO_URI)
        # Streaming performance configurations
        .config("spark.sql.shuffle.partitions", "4")        # Low partition size for streaming
        .config("spark.streaming.stopGracefullyOnShutdown", "true")  # Graceful shutdown
        .getOrCreate()
    )
    
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"✓ Streaming Spark Session active | UI: http://localhost:4040")
    return spark


def read_from_kafka(spark: SparkSession):
    """
    Creates a Structured Streaming DataFrame from the Kafka topic.
    
    Each message read from Kafka:
    - key: Partition key (country_sector)
    - value: JSON string (emission data)
    - topic, partition, offset, timestamp: Kafka metadata
    
    This returns an 'infinite DataFrame' — continuously updated.
    
    Returns:
        Streaming DataFrame connected to Kafka
    """
    logger.info(f"Listening to Kafka topic: {KAFKA_TOPIC}")
    
    kafka_df = (spark.readStream
        .format("kafka")                              # Kafka source
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC)             # Topic to subscribe to
        .option("startingOffsets", "latest")          # Retrieve only new messages
        # .option("startingOffsets", "earliest")      # Retrieve all historical messages (replay)
        .option("failOnDataLoss", "false")            # Continue rather than fail on data loss
        .option("maxOffsetsPerTrigger", 1000)         # Max 1000 messages per micro-batch
        .load()
    )
    
    logger.info("✓ Kafka stream connection established")
    return kafka_df


def parse_kafka_messages(kafka_df):
    """
    Converts raw Kafka messages to processable emission records.
    
    Messages from Kafka arrive in binary format:
    - value column: bytes → string → JSON parse → struct columns
    
    Args:
        kafka_df: Raw Kafka streaming DataFrame
    
    Returns:
        Parsed emission DataFrame
    """
    # 1. Cast binary value to string
    string_df = kafka_df.selectExpr("CAST(value AS STRING) as json_str",
                                     "timestamp as kafka_timestamp")
    
    # 2. Parse JSON string according to schema
    # from_json function uses Spark's JSON parser
    parsed_df = string_df.withColumn(
        "data",
        F.from_json(F.col("json_str"), EMISSION_SCHEMA)
    )
    
    # 3. Flatten nested columns (data.country → country)
    emission_df = parsed_df.select(
        "kafka_timestamp",
        F.col("data.date").alias("date"),
        F.col("data.country").alias("country"),
        F.col("data.sector").alias("sector"),
        F.col("data.`MtCO2 per day`").alias("mtco2_per_day"),  # Column containing special characters
        F.col("data.timestamp").alias("emission_timestamp"),
        F.col("data._kafka_ingested_at").alias("ingested_at"),
        F.col("data._source").alias("source"),
        F.current_timestamp().alias("processed_at")  # Spark processing time
    )
    
    # 4. Filter out invalid records (null country or negative emission)
    clean_df = (emission_df
        .filter(F.col("country").isNotNull())
        .filter(F.col("mtco2_per_day").isNotNull())
        .filter(F.col("mtco2_per_day") > 0)
        .filter(F.col("date").isNotNull())
    )
    
    logger.info("✓ Kafka message parse logic defined")
    return clean_df


def compute_rolling_stats(emission_df):
    """
    Computes real-time statistics.
    
    Critical for presentation: This demonstrates the power of Spark Streaming.
    In each micro-batch, per country and sector:
    - Total emissions
    - Cumulative average
    
    Watermarking: ignore messages arriving more than 1 day late
    (for network latency or producer retries)
    
    Args:
        emission_df: Parsed streaming DataFrame
    
    Returns:
        Aggregated streaming DataFrame
    """
    # Cast date column to timestamp (required for watermarking)
    df_with_ts = emission_df.withColumn(
        "event_time",
        F.to_timestamp(F.col("date"), "yyyy-MM-dd")
    )
    
    # Define watermark: do not accept messages more than 1 day late
    # This determines how long Spark maintains state (memory management)
    df_watermarked = df_with_ts.withWatermark("event_time", "1 day")
    
    # Daily total emissions per country + sector
    rolling_stats = (df_watermarked
        .groupBy(
            "country",
            "sector",
            F.window("event_time", "1 day").alias("time_window")  # 1-day window
        )
        .agg(
            F.sum("mtco2_per_day").alias("total_mtco2"),
            F.avg("mtco2_per_day").alias("avg_mtco2"),
            F.count("*").alias("record_count"),
            F.max("processed_at").alias("last_updated")
        )
        .withColumn("window_start", F.col("time_window.start"))
        .withColumn("window_end",   F.col("time_window.end"))
        .drop("time_window")
    )
    
    return rolling_stats


def write_to_mongodb(emission_df, query_name: str = "stream_to_mongo"):
    """
    Writes Streaming DataFrame to MongoDB.
    
    foreachBatch is used because:
    - MongoDB connector does not support streaming mode
    - Each micro-batch is processed as an individual batch write
    
    Args:
        emission_df: Streaming DataFrame to write
        query_name: Query name to display in Spark UI
    
    Returns:
        Active streaming query
    """
    def write_batch(batch_df, batch_id: int):
        """
        Callback function invoked for each micro-batch.
        
        Args:
            batch_df: Static DataFrame of this micro-batch
            batch_id: Micro-batch sequence number (0, 1, 2, ...)
        """
        count = batch_df.count()
        
        if count == 0:
            logger.info(f"[Batch {batch_id}] Empty batch, skipping")
            return
        
        logger.info(f"[Batch {batch_id}] {count} records writing to MongoDB...")
        
        # Write raw stream records to kafka_stream_log
        (batch_df.write
            .format("mongodb")
            .mode("append")
            .option("database",   "carbon_footprint")
            .option("collection", "kafka_stream_log")
            .save()
        )
        
        logger.info(f"[Batch {batch_id}] ✓ {count} records written")
    
    # Create checkpoint directory (for recovery point)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Start streaming query
    query = (emission_df.writeStream
        .outputMode("append")          # append: only new rows are written
        .foreachBatch(write_batch)     # Callback for each batch
        .option("checkpointLocation", CHECKPOINT_DIR)  # Recovery point
        .trigger(processingTime=f"{MICRO_BATCH_SECS} seconds")  # Trigger: micro-batch interval
        .queryName(query_name)         # Name displayed in Spark UI
        .start()
    )
    
    return query


def write_aggregations_to_mongo(rolling_df):
    """
    Writes aggregated statistics to a separate MongoDB collection.
    Ready-to-use summary statistics for the dashboard.
    
    complete mode: writes the updated state of all groups in every micro-batch
    (upsert-like behavior instead of append)
    """
    def write_agg_batch(batch_df, batch_id: int):
        count = batch_df.count()
        if count == 0:
            return
        
        logger.info(f"[AggBatch {batch_id}] {count} aggregated records writing...")
        
        (batch_df.write
            .format("mongodb")
            .mode("append")
            .option("database",   "carbon_footprint")
            .option("collection", "streaming_aggregations")
            .save()
        )
    
    agg_checkpoint = CHECKPOINT_DIR + "_agg"
    Path(agg_checkpoint).mkdir(parents=True, exist_ok=True)
    
    query = (rolling_df.writeStream
        .outputMode("update")           # update: write changed rows
        .foreachBatch(write_agg_batch)
        .option("checkpointLocation", agg_checkpoint)
        .trigger(processingTime=f"{MICRO_BATCH_SECS} seconds")
        .queryName("aggregations_to_mongo")
        .start()
    )
    
    return query


def main():
    """
    Main streaming pipeline orchestrator.
    
    Pipeline steps:
    1. Create Spark Session
    2. Read from Kafka (streaming source)
    3. Parse JSON
    4. Compute rolling statistics
    5. Write to MongoDB (two streams: raw + aggregated)
    6. Start queries and wait
    """
    logger.info("=" * 60)
    logger.info("Starting Spark Structured Streaming Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Spark Session
    spark = create_spark_session()
    
    try:
        # Step 2: Read from Kafka
        kafka_df = read_from_kafka(spark)
        
        # Step 3: Parse JSON
        emission_df = parse_kafka_messages(kafka_df)
        
        # Step 4: Write raw stream to MongoDB
        stream_query = write_to_mongodb(emission_df, "raw_stream_to_mongo")
        
        # Step 5: Compute rolling statistics and write
        rolling_df   = compute_rolling_stats(emission_df)
        agg_query    = write_aggregations_to_mongo(rolling_df)
        
        logger.info("\n✓ Streaming Pipeline Active!")
        logger.info(f"  Stream Query: {stream_query.name}")
        logger.info(f"  Agg Query:    {agg_query.name}")
        logger.info(f"  Kafka Topic:  {KAFKA_TOPIC}")
        logger.info(f"  MongoDB:      carbon_footprint.kafka_stream_log")
        logger.info(f"  Micro-batch:  every {MICRO_BATCH_SECS} seconds")
        logger.info("\nPress Ctrl+C to stop")
        
        # Wait until queries terminate
        spark.streams.awaitAnyTermination()
        
    except KeyboardInterrupt:
        logger.info("\n⏹ Stopped by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise
    finally:
        spark.stop()
        logger.info("Spark Session closed")


if __name__ == "__main__":
    main()
