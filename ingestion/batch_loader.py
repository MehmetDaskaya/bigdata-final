#!/usr/bin/env python3
# =============================================================================
# ingestion/batch_loader.py
# Batch Data Loader — EDGAR, Kaggle, Vehicle Datasets
# =============================================================================
# This script loads large-scale datasets using Apache Spark, 
# preprocesses them, and stores them in MongoDB.
#
# HOW IT WORKS:
#   1. Loads CSV files into Spark DataFrames
#   2. Performs basic data cleaning and validation
#   3. Writes processed data to MongoDB
#   4. You can monitor the execution in Spark Web UI (localhost:4040)
#
# PLACE IN BIG DATA ARCHITECTURE:
#   Data Sources → [BATCH LOADER] → Spark → MongoDB
#                                    ↘ HDFS/Parquet (raw data archive)
#
# Usage:
#   python ingestion/batch_loader.py --dataset all
#   python ingestion/batch_loader.py --dataset edgar
#   python ingestion/batch_loader.py --dataset individual
#   python ingestion/batch_loader.py --dataset vehicles
# =============================================================================

import os
import sys
import argparse
import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('batch_loader')

# =============================================================================
# Directory Structure
# =============================================================================
BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:password@localhost:27017/carbon_footprint")


def create_spark_session() -> SparkSession:
    """
    Creates an Apache Spark session.
    
    Important configurations:
    - MongoDB Spark Connector: Required for Spark to write to MongoDB
    - Spark UI: http://localhost:4040 (for monitoring during execution)
    - Driver memory: Increased (4GB) for large CSV files
    
    NOTE: Spark starts a JVM in the background.
    In cluster mode, spark://spark-master:7077 is used.
    In local mode, local[*] utilizes all CPU cores.
    
    Returns:
        Configured SparkSession
    """
    logger.info("Creating Spark Session...")
    
    spark = (SparkSession.builder
        .appName("CarbonFootprint-BatchLoader")          # Name to display in Spark UI
        .master(os.getenv("SPARK_MASTER", "local[*]"))   # local[*] = all CPU cores
        # MongoDB Connector JAR — allows Spark to write to MongoDB
        .config("spark.jars.packages",
                "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
        # MongoDB connection configuration
        .config("spark.mongodb.write.connection.uri", MONGO_URI)
        .config("spark.mongodb.read.connection.uri", MONGO_URI)
        # Memory settings (for large CSV files)
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "2g")
        # Number of shuffle partitions (optimized for small data)
        .config("spark.sql.shuffle.partitions", "8")
        # Adaptive Query Execution (Spark 3.x feature — auto-optimization)
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    
    # Set log level to WARN (INFO is too noisy)
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"✓ Spark Session active | Version: {spark.version}")
    logger.info(f"  Web UI: http://localhost:4040")
    logger.info(f"  Master: {spark.sparkContext.master}")
    
    return spark


def load_edgar_dataset(spark: SparkSession):
    """
    Loads the EDGAR Country-Based Emission Dataset
    
    Data Schema:
    - country_code: ISO 3166-1 alpha-2 (e.g. CN, US, TR)
    - country_name: Full country name
    - year: Year (1990-2023)
    - sector: Emission sector (ENERGY INDUSTRIES, TRANSPORT, etc.)
    - emission_mtco2: Annual emission (MtCO2)
    - per_capita_tco2: Per capita emission (tCO2)
    
    Steps:
    1. Load CSV
    2. Data type conversions
    3. Filter out negative values
    4. Write to MongoDB
    """
    edgar_file = DATA_DIR / "edgar" / "edgar_country_sector_1990_2023.csv"
    
    if not edgar_file.exists():
        logger.error(f"EDGAR file not found: {edgar_file}")
        logger.info("Resolution: python data/download_data.py")
        return
    
    logger.info(f"Loading EDGAR dataset: {edgar_file}")
    
    # DataFrame schema (more reliable than auto-detection)
    schema = StructType([
        StructField("country_code",    StringType(),  True),
        StructField("country_name",    StringType(),  True),
        StructField("year",            IntegerType(), True),
        StructField("sector",          StringType(),  True),
        StructField("emission_mtco2",  DoubleType(),  True),
        StructField("per_capita_tco2", DoubleType(),  True)
    ])
    
    df = (spark.read
          .option("header", "true")
          .option("encoding", "UTF-8")
          .schema(schema)
          .csv(str(edgar_file))
    )
    
    logger.info(f"Raw records loaded: {df.count():,}")
    
    # === DATA CLEANING ===
    # Drop null and negative emission values
    df_clean = (df
        .dropna(subset=["country_code", "year", "emission_mtco2"])
        .filter(F.col("emission_mtco2") > 0)
        .filter(F.col("year").between(1990, 2024))
        # Add timestamp (for MongoDB time-series)
        .withColumn("timestamp", F.to_timestamp(
            F.concat(F.col("year"), F.lit("-07-01")),  # Mid-year — for annual data
            "yyyy-MM-dd"
        ))
        # MongoDB collection metadata
        .withColumn("metadata", F.struct(
            F.col("country_code").alias("country"),
            F.col("sector")
        ))
        # Data source tag
        .withColumn("data_source", F.lit("EDGAR"))
        .withColumn("loaded_at", F.current_timestamp())
    )
    
    clean_count = df_clean.count()
    logger.info(f"Cleaned records: {clean_count:,}")
    
    # Summary statistics (useful for presentation)
    logger.info("Emission summary by sector (top 5):")
    (df_clean
        .groupBy("sector")
        .agg(F.round(F.avg("emission_mtco2"), 2).alias("avg_emission_mt"))
        .orderBy(F.desc("avg_emission_mt"))
        .show(5, truncate=False)
    )
    
    # === WRITE TO MONGODB ===
    # append mode: does not overwrite existing data, appends instead
    logger.info("Writing to MongoDB: carbon_footprint.edgar_data")
    (df_clean.write
        .format("mongodb")
        .mode("append")
        .option("database",   "carbon_footprint")
        .option("collection", "edgar_data")
        .save()
    )
    
    # === WRITE TO PARQUET ARCHIVE (HDFS simulation) ===
    parquet_path = str(DATA_DIR / "edgar" / "parquet")
    logger.info(f"Writing to Parquet archive: {parquet_path}")
    (df_clean
        .write
        .mode("overwrite")
        .partitionBy("country_code", "year")  # Partition by country+year (fast reads)
        .parquet(parquet_path)
    )
    
    logger.info(f"✓ EDGAR loading completed: {clean_count:,} records")


def load_individual_dataset(spark: SparkSession):
    """
    Loads the Individual Carbon Footprint Dataset
    
    Kaggle dataset — personal lifestyle features and carbon emission
    
    Steps:
    1. Load CSV
    2. Transform categorical columns
    3. Create class labels (Low/Medium/High)
    4. Write to MongoDB individual_records collection
    """
    individual_file = DATA_DIR / "kaggle_individual" / "individual_carbon.csv"
    
    if not individual_file.exists():
        logger.error(f"Individual data not found: {individual_file}")
        return
    
    logger.info(f"Loading individual dataset: {individual_file}")
    
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "true")   # Auto-detect schema
          .csv(str(individual_file))
    )
    
    logger.info(f"Loaded records: {df.count():,}")
    logger.info(f"Columns: {df.columns}")
    
    # === CREATE CLASS LABELS ===
    # Split CarbonEmission value into 3 categories:
    # Low: < 2500 kg CO2e/year (eco-friendly)
    # Medium: 2500-5000 kg CO2e/year (average)
    # High: > 5000 kg CO2e/year (high emission)
    df_labeled = df.withColumn("emission_class",
        F.when(F.col("CarbonEmission") < 2500, "Low")
         .when(F.col("CarbonEmission") < 5000, "Medium")
         .otherwise("High")
    ).withColumn("emission_label",  # Numerical label for machine learning
        F.when(F.col("CarbonEmission") < 2500, 0)
         .when(F.col("CarbonEmission") < 5000, 1)
         .otherwise(2)
    ).withColumn("data_source", F.lit("Kaggle_Individual")
    ).withColumn("loaded_at", F.current_timestamp())
    
    # Show class distribution (imbalance analysis)
    logger.info("Class distribution:")
    df_labeled.groupBy("emission_class").count().orderBy("emission_class").show()
    
    # Write to MongoDB
    (df_labeled.write
        .format("mongodb")
        .mode("append")
        .option("database",   "carbon_footprint")
        .option("collection", "individual_records")
        .save()
    )
    
    logger.info(f"✓ Individual dataset loaded: {df_labeled.count():,} records")


def load_vehicles_dataset(spark: SparkSession):
    """
    Loads the Vehicle CO2 Emission Dataset
    
    Canadian government vehicle fuel consumption and CO2 emission data
    
    Steps:
    1. Load CSV
    2. Prepare target variable for regression over CO2 emissions
    3. Clean categorical columns
    4. Write to MongoDB vehicle_records collection
    """
    vehicles_file = DATA_DIR / "vehicles_co2" / "vehicles_co2.csv"
    
    if not vehicles_file.exists():
        logger.error(f"Vehicle data not found: {vehicles_file}")
        return
    
    logger.info(f"Loading vehicle dataset: {vehicles_file}")
    
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv(str(vehicles_file))
    )
    
    logger.info(f"Loaded records: {df.count():,}")
    
    # Normalize column names (replace spaces with underscores)
    for col_name in df.columns:
        clean_name = col_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_per_")
        df = df.withColumnRenamed(col_name, clean_name)
    
    # Clean and validate CO2 g/km values
    co2_col = next((c for c in df.columns if 'CO2' in c.upper() or 'co2' in c.lower()), None)
    if co2_col:
        df = df.filter(F.col(co2_col) > 0).filter(F.col(co2_col) < 1000)
    
    df = (df
        .withColumn("data_source", F.lit("Canada_Vehicles_CO2"))
        .withColumn("loaded_at", F.current_timestamp())
    )
    
    # Average emissions by fuel type (EDA result)
    logger.info("Average CO2 emissions by fuel type:")
    fuel_col = next((c for c in df.columns if 'Fuel_Type' in c or 'fuel_type' in c.lower()), None)
    if fuel_col and co2_col:
        df.groupBy(fuel_col).agg(
            F.round(F.avg(co2_col), 1).alias("avg_co2_g_per_km"),
            F.count("*").alias("vehicle_count")
        ).orderBy("avg_co2_g_per_km").show()
    
    # Write to MongoDB
    (df.write
        .format("mongodb")
        .mode("append")
        .option("database",   "carbon_footprint")
        .option("collection", "vehicle_records")
        .save()
    )
    
    logger.info(f"✓ Vehicle dataset loaded: {df.count():,} records")


def main():
    """
    Main function — loads the selected datasets.
    """
    parser = argparse.ArgumentParser(
        description='Carbon Footprint Batch Data Loader'
    )
    parser.add_argument('--dataset', 
                        choices=['all', 'edgar', 'individual', 'vehicles'],
                        default='all',
                        help='Dataset to load (default: all)')
    args = parser.parse_args()
    
    # Create Spark Session
    spark = create_spark_session()
    
    try:
        if args.dataset in ['all', 'edgar']:
            logger.info("\n" + "="*50)
            logger.info("Loading EDGAR Dataset...")
            logger.info("="*50)
            load_edgar_dataset(spark)
        
        if args.dataset in ['all', 'individual']:
            logger.info("\n" + "="*50)
            logger.info("Loading Individual Dataset...")
            logger.info("="*50)
            load_individual_dataset(spark)
        
        if args.dataset in ['all', 'vehicles']:
            logger.info("\n" + "="*50)
            logger.info("Loading Vehicle CO2 Dataset...")
            logger.info("="*50)
            load_vehicles_dataset(spark)
        
        logger.info("\n✓ Batch loading completed!")
        logger.info("To view data in MongoDB: http://localhost:8081")
        
    except Exception as e:
        logger.error(f"Loading error: {e}", exc_info=True)
        raise
    finally:
        spark.stop()
        logger.info("Spark Session closed.")


if __name__ == "__main__":
    main()
