# Territorial CO₂ Emissions Prediction System
### BDA5011 Big Data Analytics — Final Project
**Bahçeşehir University, Department of Big Data Analytics and Management**

> **A Scalable Big Data Architecture for Territorial CO₂ Emissions Prediction and Personal Carbon Footprint Downstream Classification using Apache Kafka, Apache Spark, MongoDB, PyTorch LSTM, and XGBoost.**

---

## 📋 Project Overview

This project implements a dual-path **Lambda Architecture** designed to ingest, process, and analyze greenhouse gas emissions at scale. The system integrates macroeconomic territorial CO₂ emissions (daily estimates from Carbon Monitor and annual historical inventories from EDGAR) with microeconomic lifestyle indicators (Kaggle lifestyle dataset) to enable multi-scale sustainability monitoring.

```
[EDGAR / Carbon Monitor CSV]
        │
        ▼ (Simulated Daily Stream)
[Kafka Producer] ──► kafka:9092 ──► Topic: carbon-emissions-daily
        │
        ▼ (10s Micro-Batches)
[Spark Structured Streaming] ◄───► [Spark Batch Pipeline] (Historical)
        │                                  │
        ▼                                  ▼
[MongoDB Time-Series]              [Parquet Archive]
  (emissions_timeseries)
        │
        ▼ (Model Training & Inference)
[ML Models: PyTorch LSTM + XGBoost + Spark MLlib RF]
        │
        ▼ (API Serving Layer)
[Next.js Interactive Dashboard: localhost:3000]
```

---

## 📊 Core Research Findings & ML Evaluation

All machine learning models are trained strictly under a leak-free evaluation protocol. Target leakage is prevented by shifting rolling windows by 1 day before feature engineering, and classification SMOTE resampling is strictly confined to the training folds.

### Machine Learning & Baseline Comparisons
Daily emissions time series exhibit high autocorrelation. Including naive persistence baselines is critical to establishing rigorous scientific validity:

| Model / Baseline | Task / Resolution | MAE | RMSE | $R^2$ / F1-Score | Data Scope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **7-Day Moving Average Baseline** | Daily Baseline Forecasting | **0.0189** | **0.0454** | **`0.9941`** | Carbon Monitor (Global) |
| **Naive Yesterday Value Baseline** | Daily Baseline Forecasting | **0.0216** | **0.0546** | **`0.9915`** | Carbon Monitor (Global) |
| **XGBoost Regression** | Daily Emission Prediction | 0.0404 | 0.1099 | **`0.9654`** | Carbon Monitor (Global) |
| **LSTM (PyTorch)** | Sequence-to-Sequence (7d ahead) | 0.1000 | 0.1231 | **`0.8533`** | Carbon Monitor (CN Power) |
| **XGBoost Classification** | Downstream Footprint Class | — | — | **`0.8555`** *(F1)* | Kaggle Individual |
| **Spark MLlib Random Forest** | Distributed Batch Regression | 104.57 | 237.27 | **`0.6644`** | EDGAR (Annual) |

*Note: Autoregressive baselines naturally dominate daily one-step forecasting due to strong temporal persistence. XGBoost remains highly valuable for its cross-country generalization, multivariate explanatory power, SHAP-based interpretability, and live dashboard integration—features that simple persistence cannot provide.*

---

## 🚀 Startup & Execution Guide

### Prerequisites
* **Docker Desktop** (with Compose enabled)
* **Python 3.10+**
* **Node.js 18+**

### Step 1: Orchestrate the Docker Cluster
Spin up Zookeeper, Kafka, Spark Master, two Spark Workers, MongoDB, and Mongo Express:
```bash
docker-compose up -d
```
Verify that all containers are healthy:
```bash
docker-compose ps
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Ingestion & Processing Layer
1. **Batch Pipeline**: Execute the batch processing script to load EDGAR inventories into MongoDB and archive Parquet stores:
   ```bash
   python processing/spark_batch_pipeline.py
   ```
2. **Real-time Kafka Stream**: Launch the Spark Streaming consumer to listen to Kafka and populate the MongoDB time-series collection:
   ```bash
   python processing/spark_streaming_pipeline.py
   ```
3. **Kafka Producer**: Start the Kafka producer to simulate real-time daily streaming by replaying CSV rows:
   ```bash
   python ingestion/kafka_producer.py --speed 0.5
   ```

### Step 4: Model Training & Evaluation
Train the regression, sequence, and classification models and output updated visual assets to the `results/` folder:
```bash
# Trains XGBoost, LSTM, and runs diagnostic evaluations
python ml/run_training.py

# Generates comparison reports and JSON artifacts for the dashboard
python ml/evaluate.py
```

### Step 5: Start the Presentation Dashboard
Run the Next.js frontend application to visualize the findings:
```bash
cd dashboard
npm run dev
# Open: http://localhost:3000
```

---

## 🌐 Component Endpoint Map

| Service | Endpoint / Port | Purpose |
| :--- | :--- | :--- |
| **Presentation Dashboard** | `http://localhost:3000` | Main frontend interface and charts |
| **Mongo Express GUI** | `http://localhost:8081` | Database UI (view collections & records live) |
| **Spark Master UI** | `http://localhost:8080` | Distributed cluster status tracking |
| **Spark Active Application** | `http://localhost:4040` | Real-time Spark job DAG monitoring |
| **MongoDB Database** | `localhost:27017` | Serving layer endpoint |
| **Apache Kafka Broker** | `localhost:9092` | Event streaming ingestion broker |

---

## 📁 Repository Structure

```
.
├── docker-compose.yml             # Zookeeper, Kafka, Spark x2, MongoDB, Mongo Express
├── mongo-init.js                  # Database collection schemas & indexes init
├── requirements.txt               # Pinned Python dependencies
├── README.md                      # This startup & findings guide
│
├── data/                          # Raw and processed datasets
│   ├── download_data.py           # Dataset downloader/generator utility
│   ├── carbon_monitor/            # Daily global CO₂ emissions CSVs
│   ├── kaggle_individual/         # Individual lifestyle survey CSV
│   └── edgar/                     # EDGAR country and sector inventories
│
├── ingestion/                     # Streaming ingestion scripts
│   ├── kafka_producer.py          # Publishes rows to carbon-emissions-daily
│   └── batch_loader.py            # Custom batch loader with retry policies
│
├── processing/                    # Spark ETL engine scripts
│   ├── spark_batch_pipeline.py    # Historical batch processing (writes to MongoDB)
│   └── spark_streaming_pipeline.py# Spark Structured Streaming from Kafka -> MongoDB
│
├── ml/                            # Machine learning & modeling codebase
│   ├── lstm_model.py              # PyTorch LSTM time-series implementation
│   ├── xgboost_model.py           # Gradient Boosting regression & classification
│   ├── spark_mllib_model.py       # Spark MLlib distributed Random Forest
│   ├── run_training.py            # Orchestrator training the ML suite
│   ├── evaluate.py                # Compiles metrics and saves evaluation charts
│   └── saved_models/              # Serialized trained model binaries
│
├── dashboard/                     # Next.js interactive frontend
│   ├── app/
│   │   ├── page.tsx               # Recharts-powered visualizer and KPI tabs
│   │   └── api/
│   │       ├── emissions/         # Serving route for database records
│   │       └── predictions/       # Serving route for predictions
│   └── package.json
│
├── notebooks/                     # Exploratory Data Analysis & Verification
│   ├── 01_EDA.ipynb               # Dataset visualization and profiles
│   ├── 02_Preprocessing.ipynb     # Outlier engineering & temporal encoding
│   └── 03_Model_Evaluation.ipynb  # Comparative metrics and SHAP value study
│
└── results/                       # Standardized evaluation outputs
    ├── regression_comparison.png  # Error metric bars
    ├── classification_metrics.png # Classifier performance bars
    ├── r2_comparison.png          # horizontal R² comparative chart
    └── model_evaluation_report.txt# Text execution report
```

---
*BDA5011 Big Data Analytics · Bahçeşehir University · 2026*
