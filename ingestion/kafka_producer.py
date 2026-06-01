#!/usr/bin/env python3
# =============================================================================
# ingestion/kafka_producer.py
# Kafka Producer — Real-Time Emission Stream Simulator
# =============================================================================
# This script streams carbon emission records from Carbon Monitor CSV to Apache Kafka 
# as a simulated real-time feed.
#
# HOW IT WORKS:
#   1. Reads the Carbon Monitor CSV (sorted chronologically)
#   2. Publishes each row as a Kafka message to 'carbon-emissions-daily' topic
#   3. Pauses briefly between messages for real-time simulation
#   4. Allows the Spark Structured Streaming consumer to ingest real-time data
#
# PLACE IN BIG DATA ARCHITECTURE:
#   Data Sources → [KAFKA PRODUCER] → Kafka Topic → Spark Streaming → MongoDB
#
# Usage:
#   python ingestion/kafka_producer.py
#   python ingestion/kafka_producer.py --speed 0.5  # 2x speed
#   python ingestion/kafka_producer.py --batch       # Fast batch upload
# =============================================================================

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# Logging configuration — to monitor the stream in the terminal during presentation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('kafka_producer')

# =============================================================================
# Configuration Constants
# =============================================================================
BASE_DIR     = Path(__file__).parent.parent
DATA_FILE    = BASE_DIR / "data" / "carbon_monitor" / "carbon_monitor_global.csv"
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")   # Get from environment variable
KAFKA_TOPIC  = "carbon-emissions-daily"                      # Topic that the Spark consumer listens to
DEFAULT_DELAY = 1.0   # Delay between messages (seconds) — provides real-time simulation feel


def create_producer(max_retries: int = 10) -> KafkaProducer:
    """
    Creates a Kafka producer instance.
    Retries max_retries times if Kafka is not ready.
    
    Args:
        max_retries: Maximum connection retries
    
    Returns:
        Connected KafkaProducer instance
    
    Raises:
        SystemExit: Exits if all attempts fail
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connecting to Kafka ({KAFKA_BROKER})... Attempt {attempt}/{max_retries}")
            
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                # Serializer to convert message to JSON string → bytes.
                # Each emission record is sent in JSON format
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                # Serializer for message key (country+sector = partition key)
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                # Reliability configuration:
                acks='all',           # Wait for all replicas to acknowledge
                retries=3,            # Retry 3 times on send failure
                max_block_ms=10000,   # Maximum wait block time (10 seconds)
            )
            
            logger.info(f"✓ Kafka connection successful! Topic: {KAFKA_TOPIC}")
            return producer
            
        except NoBrokersAvailable:
            if attempt < max_retries:
                wait = min(2 ** attempt, 30)  # Exponential backoff (max 30 seconds)
                logger.warning(f"Kafka not ready, waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Could not connect to Kafka after {max_retries} attempts!")
                logger.error("Make sure the Docker cluster is running: docker-compose up -d")
                sys.exit(1)


def load_emission_data() -> pd.DataFrame:
    """
    Loads the Carbon Monitor CSV file and prepares it for streaming.
    
    Data Preprocessing:
    - Converts date column to datetime
    - Sorts chronologically (for real-time streaming simulation)
    - Cleans missing values
    
    Returns:
        Sorted and cleaned DataFrame
    """
    if not DATA_FILE.exists():
        logger.error(f"Data file not found: {DATA_FILE}")
        logger.info("Run first: python data/download_data.py")
        sys.exit(1)
    
    logger.info(f"Loading data file: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    logger.info(f"Number of loaded records: {len(df):,}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Find and parse date column
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)  # Chronological sorting
        logger.info(f"Date range: {df[date_col].min().date()} → {df[date_col].max().date()}")
    
    # Drop rows with missing values
    original_len = len(df)
    df = df.dropna()
    if len(df) < original_len:
        logger.info(f"Cleaned: dropped {original_len - len(df)} rows with missing values")
    
    return df


def record_to_message(row: pd.Series) -> dict:
    """
    Converts DataFrame row to Kafka message (dict).
    Message structure fits the JSON schema expected by Spark Structured Streaming.
    
    Args:
        row: DataFrame row
    
    Returns:
        JSON-serializable dict
    """
    # Convert Pandas timestamp to Python datetime (for JSON serialization)
    message = {}
    for col, val in row.items():
        if hasattr(val, 'isoformat'):
            message[col] = val.isoformat()
        elif pd.isna(val):
            message[col] = None
        else:
            message[col] = val
    
    # Add ingestion timestamp to Kafka message. Used to calculate latency on the Spark side.
    message['_kafka_ingested_at'] = datetime.utcnow().isoformat()
    message['_source'] = 'carbon_monitor_daily'
    
    return message


def stream_data(producer: KafkaProducer, df: pd.DataFrame, delay: float, batch_mode: bool = False):
    """
    Sends records in DataFrame to Kafka topic.
    Key for presentation: This function simulates a real-time data stream.
    In production, real-time data is fetched from the Carbon Monitor API.
    
    Args:
        producer: Active KafkaProducer instance
        df: Data to be sent
        delay: Delay between messages (seconds)
        batch_mode: If True, send all data rapidly (test mode)
    """
    total = len(df)
    sent = 0
    errors = 0
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Streaming started: {total:,} records")
    logger.info(f"Topic: {KAFKA_TOPIC}")
    logger.info(f"Delay: {delay}s/message {'(batch mode)' if batch_mode else ''}")
    logger.info(f"{'='*50}\n")
    
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            # Convert row to message
            message = record_to_message(row)
            
            # Partition key: country + sector combination.
            # The same country-sector pair always routes to the same partition.
            # This ensures records remain ordered per partition.
            partition_key = f"{message.get('country', 'UNKNOWN')}_{message.get('sector', 'ALL')}"
            
            # Send to Kafka (asynchronously — with callback)
            future = producer.send(
                topic=KAFKA_TOPIC,
                key=partition_key,
                value=message
            )
            
            sent += 1
            
            # Progress report every 100 messages (for the presentation terminal)
            if sent % 100 == 0:
                progress = (idx + 1) / total * 100
                logger.info(
                    f"[{progress:5.1f}%] {sent:,}/{total:,} messages sent | "
                    f"Last: {message.get('country','?')} | {message.get('sector','?')} | "
                    f"{message.get('MtCO2 per day', message.get('emission', '?'))} MtCO2"
                )
            
            # Delay between messages (real-time simulation)
            if not batch_mode and delay > 0:
                time.sleep(delay)
                
        except Exception as e:
            errors += 1
            logger.warning(f"Could not send message (row {idx}): {e}")
            
            # Stop if more than 10 errors occur
            if errors > 10:
                logger.error("Too many errors! Stopping the stream.")
                break
    
    # Flush all pending messages (ensure they are sent)
    producer.flush(timeout=30)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"✓ Streaming completed!")
    logger.info(f"  Sent: {sent:,} messages")
    logger.info(f"  Errors: {errors} messages")
    logger.info(f"  Topic: {KAFKA_TOPIC}")
    logger.info(f"{'='*50}")


def main():
    """
    Main function — argument parsing and orchestration.
    
    Command line arguments:
    --speed FLOAT  : Delay between messages in seconds (default: 1.0)
    --batch        : Fast batch sending mode (for testing)
    --rows INT     : Maximum rows to send (for testing)
    """
    parser = argparse.ArgumentParser(
        description='Carbon Monitor Kafka Producer — Real-Time Emission Stream'
    )
    parser.add_argument('--speed', type=float, default=DEFAULT_DELAY,
                        help=f'Delay between messages (seconds, default: {DEFAULT_DELAY})')
    parser.add_argument('--batch', action='store_true',
                        help='Batch sending mode — no delay')
    parser.add_argument('--rows', type=int, default=None,
                        help='Maximum rows to send (for testing)')
    args = parser.parse_args()
    
    # 1. Create Kafka producer
    producer = create_producer()
    
    # 2. Load data
    df = load_emission_data()
    
    # Limit rows in testing mode
    if args.rows:
        df = df.head(args.rows)
        logger.info(f"Test mode: First {args.rows} rows will be sent")
    
    # 3. Start stream
    try:
        stream_data(
            producer=producer,
            df=df,
            delay=args.speed,
            batch_mode=args.batch
        )
    except KeyboardInterrupt:
        logger.info("\n⏹ Stopped by user (Ctrl+C)")
    finally:
        producer.close()
        logger.info("Kafka producer closed.")


if __name__ == "__main__":
    main()
