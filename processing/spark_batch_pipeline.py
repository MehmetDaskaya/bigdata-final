#!/usr/bin/env python3
# =============================================================================
# processing/spark_batch_pipeline.py
# Spark Batch Processing Pipeline — EDA + Feature Engineering + Aggregation
# =============================================================================
# This script performs batch processing on EDGAR and Carbon Monitor data using Spark.
# Crucial for presentation: represents the 'Batch Layer' of the Lambda architecture.
#
# PROCESSING STEPS:
#   1. Load raw CSV data into Spark
#   2. Data cleaning and schema normalization
#   3. Feature engineering (derived features)
#   4. Compute EDA statistics
#   5. Construct feature matrix for ML training
#   6. Write results to MongoDB and Parquet
#
# Usage:
#   python processing/spark_batch_pipeline.py
# =============================================================================

import os
import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('spark_batch')

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:password@localhost:27017/carbon_footprint")


def create_spark_session() -> SparkSession:
    """
    Creates a Spark Session optimized for batch processing.
    
    Key settings:
    - spark.sql.adaptive.enabled: Spark 3.x's Adaptive Query Execution (AQE, auto-optimizes based on data size)
    - spark.serializer: Kryo serializer is faster and uses less memory
    """
    spark = (SparkSession.builder
        .appName("CarbonFootprint-BatchPipeline")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.jars.packages",
                "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
        .config("spark.mongodb.write.connection.uri", MONGO_URI)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"✓ Batch Spark Session | Version: {spark.version} | Master: {spark.sparkContext.master}")
    return spark


def process_carbon_monitor(spark: SparkSession):
    """
    Carbon Monitor Data Processing Pipeline
    
    Steps:
    1. CSV → Spark DataFrame
    2. Parse date columns
    3. Calculate 7-day and 30-day rolling averages (Window function)
    4. Add seasonal indicators (month, quarter, holiday flag)
    5. Generate lag features (prior day/week/month emissions)
    6. Write results to MongoDB emissions_timeseries
    """
    carbon_file = DATA_DIR / "carbon_monitor" / "carbon_monitor_global.csv"
    
    if not carbon_file.exists():
        logger.warning(f"Carbon Monitor file not found: {carbon_file}")
        return None
    
    logger.info("Processing Carbon Monitor data...")
    
    # Load CSV
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv(str(carbon_file))
    )
    
    # Normalize column name
    emission_col = next(
        (c for c in df.columns if 'MtCO2' in c or 'emission' in c.lower()),
        df.columns[-1]
    )
    df = df.withColumnRenamed(emission_col, "mtco2_per_day")
    
    # Parse date column
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col and date_col != 'timestamp':
        df = df.withColumn("emission_date", F.to_date(F.col(date_col), "yyyy-MM-dd"))
    else:
        df = df.withColumn("emission_date", F.to_date("timestamp"))
    
    # === FEATURE ENGINEERING — Time-Based Features ===
    df = (df
        # Basic time components
        .withColumn("year",         F.year("emission_date"))
        .withColumn("month",        F.month("emission_date"))
        .withColumn("day_of_week",  F.dayofweek("emission_date"))  # 1=Sunday, 7=Saturday
        .withColumn("day_of_year",  F.dayofyear("emission_date"))
        .withColumn("quarter",      F.quarter("emission_date"))
        # Seasonal indicator (Northern hemisphere basis)
        .withColumn("season",
            F.when(F.col("month").isin(12, 1, 2), "Winter")
             .when(F.col("month").isin(3, 4, 5),  "Spring")
             .when(F.col("month").isin(6, 7, 8),  "Summer")
             .otherwise("Autumn")
        )
        # Weekend flag (important for ground transport sector)
        .withColumn("is_weekend", F.col("day_of_week").isin(1, 7).cast("int"))
        # Seasonal sine/cosine encoding (better suited for ML cyclical data)
        .withColumn("month_sin", F.sin(2 * 3.14159 * F.col("month") / 12))
        .withColumn("month_cos", F.cos(2 * 3.14159 * F.col("month") / 12))
    )
    
    # === WINDOW FUNCTIONS — Rolling Averages ===
    # Time series window per country + sector
    window_spec = (Window
        .partitionBy("country", "sector")         # Each country-sector group is independent
        .orderBy("emission_date")
        .rowsBetween(-6, 0)                        # Last 7 days (including current)
    )
    
    window_30 = (Window
        .partitionBy("country", "sector")
        .orderBy("emission_date")
        .rowsBetween(-29, 0)                       # Last 30 days
    )
    
    lag_window = (Window
        .partitionBy("country", "sector")
        .orderBy("emission_date")
    )
    
    df = (df
        # 7-day rolling average (weekly trend)
        .withColumn("rolling_avg_7d",  F.avg("mtco2_per_day").over(window_spec))
        # 30-day rolling average (monthly trend)
        .withColumn("rolling_avg_30d", F.avg("mtco2_per_day").over(window_30))
        # Lag features (critical for time-series models)
        .withColumn("lag_1d",  F.lag("mtco2_per_day", 1).over(lag_window))   # Yesterday
        .withColumn("lag_7d",  F.lag("mtco2_per_day", 7).over(lag_window))   # Last week
        .withColumn("lag_30d", F.lag("mtco2_per_day", 30).over(lag_window))  # Last month
        # Rate of change (percentage change relative to prior day)
        .withColumn("pct_change",
            ((F.col("mtco2_per_day") - F.col("lag_1d")) / F.col("lag_1d") * 100)
        )
        # Timestamp for MongoDB
        .withColumn("timestamp", F.to_timestamp("emission_date"))
        .withColumn("metadata", F.struct(
            F.col("country"),
            F.col("sector")
        ))
        .withColumn("data_source", F.lit("Carbon_Monitor"))
        .withColumn("processed_at", F.current_timestamp())
    )
    
    # Drop first rows with null lag values (insufficient history)
    df_clean = df.dropna(subset=["lag_7d", "lag_30d"])
    
    count = df_clean.count()
    logger.info(f"Carbon Monitor: {count:,} processed records")
    
    # EDA: Countries with highest emissions
    logger.info("\nCountries with highest average daily emissions (Top 10):")
    (df_clean
        .groupBy("country")
        .agg(F.round(F.avg("mtco2_per_day"), 3).alias("avg_daily_mtco2"))
        .orderBy(F.desc("avg_daily_mtco2"))
        .show(10)
    )
    
    # Write to MongoDB
    (df_clean.write
        .format("mongodb")
        .mode("append")
        .option("database",   "carbon_footprint")
        .option("collection", "emissions_timeseries")
        .save()
    )
    
    # Write to Parquet as well (fast access for ML training)
    parquet_path = str(DATA_DIR / "carbon_monitor" / "processed_parquet")
    (df_clean.write
        .mode("overwrite")
        .partitionBy("country", "year")
        .parquet(parquet_path)
    )
    
    logger.info(f"✓ Carbon Monitor processing completed → {count:,} records")
    return df_clean


def compute_eda_statistics(spark: SparkSession, df):
    """
    Exploratory Data Analysis (EDA) Statistics
    
    For presentation: This function computes the EDA statistics expected by the instructor.
    These stats are also visualized in the Jupyter notebook for charts.
    
    Computed metrics:
    - Annual total emission trend by country
    - Emission share by sector
    - Pre/post COVID-19 comparison
    - Seasonal pattern analysis
    """
    if df is None:
        logger.warning("No data for EDA")
        return
    
    logger.info("\n" + "="*60)
    logger.info("EDA — Exploratory Data Analysis Statistics")
    logger.info("="*60)
    
    # 1. Annual total emission trend (global)
    logger.info("\n[1] Annual Global CO2 Emission Trend:")
    (df.groupBy("year")
       .agg(F.round(F.sum("mtco2_per_day"), 1).alias("total_daily_mtco2"))
       .orderBy("year")
       .show(20)
    )
    
    # 2. Emission share by sector
    logger.info("\n[2] Average Daily Emission by Sector:")
    (df.groupBy("sector")
       .agg(
           F.round(F.avg("mtco2_per_day"), 3).alias("avg_daily_mtco2"),
           F.round(F.sum("mtco2_per_day"), 1).alias("total_mtco2")
       )
       .orderBy(F.desc("avg_daily_mtco2"))
       .show()
    )
    
    # 3. COVID-19 impact analysis (2019 vs 2020)
    logger.info("\n[3] COVID-19 Impact — 2019 vs 2020 Comparison:")
    (df.filter(F.col("year").isin(2019, 2020))
       .groupBy("year", "sector")
       .agg(F.round(F.avg("mtco2_per_day"), 3).alias("avg_daily_mtco2"))
       .orderBy("sector", "year")
       .show(20)
    )
    
    # 4. Seasonal pattern analysis
    logger.info("\n[4] Seasonal Emission Pattern:")
    (df.groupBy("season")
       .agg(F.round(F.avg("mtco2_per_day"), 3).alias("avg_emission"))
       .orderBy("season")
       .show()
    )
    
    # Save statistics as JSON (for dashboard)
    stats_path = BASE_DIR / "data" / "eda_stats.json"
    
    country_stats = (df
        .groupBy("country")
        .agg(
            F.round(F.avg("mtco2_per_day"), 4).alias("avg_emission"),
            F.round(F.sum("mtco2_per_day"), 2).alias("total_emission"),
            F.count("*").alias("record_count")
        )
        .orderBy(F.desc("avg_emission"))
        .limit(20)
        .toPandas()
    )
    
    country_stats.to_json(str(stats_path), orient='records', indent=2)
    logger.info(f"\n✓ EDA statistics saved: {stats_path}")


def main():
    """
    Batch pipeline main orchestrator.
    """
    logger.info("=" * 60)
    logger.info("Starting Spark Batch Processing Pipeline")
    logger.info("=" * 60)
    
    spark = create_spark_session()
    
    try:
        # Process Carbon Monitor
        cm_df = process_carbon_monitor(spark)
        
        # Compute EDA statistics
        compute_eda_statistics(spark, cm_df)
        
        logger.info("\n✓ Batch Pipeline completed!")
        logger.info("Next step: python ml/train_all.py")
        
    except Exception as e:
        logger.error(f"Batch pipeline error: {e}", exc_info=True)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
