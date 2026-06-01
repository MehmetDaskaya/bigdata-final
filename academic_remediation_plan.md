# Academic Remediation Plan: Task-by-Task Breakdown

This document provides a rigorous, point-by-point breakdown of the 26 academic, methodological, dataset, architectural, scientific, and internal consistency critiques identified in the project report. It classifies each issue as **Can Fix** or **Cannot Fix (Reframe/Document)**, provides a technical explanation, and maps out concrete remediation tasks.

---

## 1. Major Methodology Problems

### Task 1.1: Reframe "Near-Real-Time" Streaming Ingestion
* **Type:** **Cannot Fix (System Constraint) / Can Fix (Framing & Documentation)**
* **Problem:** Ingesting a replayed CSV row-by-row is a simulation, not a true real-time streaming feed. Carbon Monitor has no public real-time streaming API.
* **Explanation:** Replaying historical data is standard practice for testing streaming infrastructure without incurring enterprise API or cloud data ingestion costs.
* **Remediation Task:**
  - Rename "live streaming" to "simulated daily streaming ingestion" in `final_report.tex` and `PRESENTATION_GUIDE.txt`.
  - Add a dedicated **Limitations of Streaming** paragraph in Section III-A of the LaTeX report to explain that the Kafka pipeline acts as a streaming simulation using historical replays, demonstrating high-throughput architectural validation without a live API key.

### Task 1.2: Eliminate Target Leakage in Rolling Average Features
* **Type:** **Can Fix (Code & Reporting)**
* **Problem:** XGBoost $R^2 = 0.9653$ is inflated because rolling averages (`rolling_avg_7d`, `rolling_avg_30d`) were computed without shifting. Standard rolling windows include the current day, meaning today's target (daily CO₂ emission) is directly leaked into today's rolling average features.
* **Remediation Task:**
  - Modify `ml/run_training.py` inside the `load_and_engineer` function to shift the groupby series by 1 day *before* applying the rolling averages:
    ```python
    df['rolling_avg_7d']  = g.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
    df['rolling_avg_30d'] = g.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    ```
  - Re-run training to obtain realistic, leak-free regression metrics.

### Task 1.3: Implement Naive and Moving Average Baselines
* **Type:** **Can Fix (Code & Reporting)**
* **Problem:** No simple baselines are compared against the machine learning models. High autocorrelation makes simple baselines strong in time-series forecasting.
* **Remediation Task:**
  - Implement two baselines on the test dataset in `ml/run_training.py`:
    1. **Yesterday's Value (Naive/Persistence Baseline):** Predicts the previous day's emission ($y_{t-1}$).
    2. **7-day Moving Average (MA Baseline):** Predicts the average of the last 7 days of emissions.
  - Calculate MAE, RMSE, and $R^2$ scores for both baselines.
  - Log baseline metrics to `all_model_metrics.json` and insert them into Table II of `final_report.tex`.

### Task 1.4: Disentangle Non-Comparable Model Comparisons
* **Type:** **Cannot Fix (Architectural Design) / Can Fix (Framing & Table Split)**
* **Problem:** Comparing daily national XGBoost, daily single-sector LSTM, and annual historical Spark Random Forest in a single table is scientifically misleading because they are trained on different resolutions, tasks, and datasets.
* **Remediation Task:**
  - Break Table II in `final_report.tex` into separate sections or sub-tables:
    - **Table II-A:** Daily Territorial CO₂ Forecasting (XGBoost vs. LSTM vs. Naive/MA Baselines).
    - **Table II-B:** Distributed Historical Batch Analysis (Spark MLlib Random Forest).
  - Add text in Section V-B explaining that these are separate tasks demonstrating the dual-speed design of the **Lambda Architecture** (Speed Layer for daily near-real-time forecasting, Batch Layer for distributed historical analysis).

### Task 1.5: Methodologically Differentiate and Integrate the Classification Task
* **Type:** **Cannot Fix (Dataset Constraint) / Can Fix (Framing)**
* **Problem:** The Kaggle lifestyle carbon footprint classifier is a separate personal task, while the core project predicts national/sector-level territorial emissions.
* **Remediation Task:**
  - Differentiate and label the classification module in Section II-C and Section IV-C as the **"Personal Lifestyle Carbon Footprint Module"**.
  - Frame this in Section I and Section VI as a downstream consumer-level expansion, showing how the system addresses sustainability at both the macroeconomic (territorial national level) and microeconomic (personal lifestyle level) scales.

---

## 2. Dataset and Source Problems

### Task 2.1: Document Carbon Monitor Estimation Methodology and Uncertainty
* **Type:** **Can Fix (Reporting)**
* **Problem:** Carbon Monitor is a proxy-based estimate (derived from electricity logs, flight activity, traffic indicators), not exact observed ground truth.
* **Remediation Task:**
  - Rewrite Section II-A to explicitly cite the original Carbon Monitor paper's estimation methods.
  - State clearly that Carbon Monitor estimates carry a $\pm 5\text{--}10\%$ scientific uncertainty interval depending on the sector.

### Task 2.2: Add a Dedicated "Data Uncertainty" Sub-section
* **Type:** **Can Fix (Reporting)**
* **Problem:** Treating Carbon Monitor like exact measured emissions compromises scientific rigor.
* **Remediation Task:**
  - Create a new sub-section `Section II-D: Dataset Limitations and Uncertainty` in the report, discussing how indirect activity indicators introduce estimation variance, and contrast this with high-latency, highly accurate direct national inventories.

### Task 2.3: Clarify the Specific EDGAR Version, Files, and Scale
* **Type:** **Can Fix (Reporting)**
* **Problem:** The claim that EDGAR is a $\sim$50 GB dataset is vague, and the files used are not clearly documented.
* **Remediation Task:**
  - Specify in Section II-B that **EDGAR v8.0 FT2022 GHG** was accessed.
  - Clarify that the full EDGAR database contains high-resolution gridded map files totaling $\sim$50 GB, but for our local cluster simulation, we utilize the high-quality national sectoral totals CSV to accommodate Docker container resource limits (1 GB RAM per worker).

### Task 2.4: Complete the EDGAR Citation
* **Type:** **Can Fix (Reporting)**
* **Problem:** The citation `edgar2021` is incomplete and lists an inconsistent "accessed May 2026" text.
* **Remediation Task:**
  - Update `final_report.tex` bibliography to cite the exact European Commission Joint Research Centre (JRC) release (Crippa et al., 2023, EDGAR v8.0 FT2022) with correct download dates and URLs.

### Task 2.5: Document Kaggle Lifestyle Dataset Provenance and License
* **Type:** **Can Fix (Reporting)**
* **Problem:** The Kaggle footprint dataset lacks documentation regarding its provenance, feature definitions, license, and synthetic label construction.
* **Remediation Task:**
  - Expand Section II-C to state that the dataset (originally compiled by Duman, 2024 under CC0 Public Domain) represents a survey-based synthetic dataset designed for prototyping individual lifestyle impact.
  - Define the 6 lifestyle features (transport, air travel, diet, home energy, shopping, and waste management).

---

## 3. Evaluation Weaknesses

### Task 3.1: Provide Country/Sector-Level Error Distributions
* **Type:** **Can Fix (Code & Reporting)**
* **Problem:** The report only provides global point metrics with no uncertainty intervals or sector-level errors.
* **Remediation Task:**
  - Modify `ml/run_training.py` to calculate the **standard deviation of prediction errors** and compile a **per-sector MAE and RMSE breakdown** for the XGBoost model.
  - Save these to `results/dashboard_metrics.json` and present this sector-level error analysis in Section V-A of the LaTeX report to give a granular look at performance.

### Task 3.2: Perform Cross-Validation against an External Ground Truth
* **Type:** **Cannot Fix (New Ingestion Pipeline) / Can Fix (Framing & Comparative Discussion)**
* **Problem:** The models are only evaluated on held-out Carbon Monitor data. A rigorous report requires external validation.
* **Remediation Task:**
  - Add a dedicated **External Validation & Calibration** section in Section V-B of `final_report.tex`.
  - Compare Carbon Monitor's aggregated annual emissions for major countries (e.g., US, CN, EU27) against EDGAR's official annual totals for overlapping years (2020–2022). Discuss the correlation coefficient and confirm Carbon Monitor's reliability.

### Task 3.3: Conduct an Feature Ablation Study for XGBoost
* **Type:** **Can Fix (Code & Reporting)**
* **Problem:** The report does not show how much predictive performance comes from temporal, lag, or rolling average features.
* **Remediation Task:**
  - Code an **ablation sequence** in `ml/run_training.py` that trains and tests three models:
    1. **Ablation Model A:** Temporal features only (`year`, `month`, cyclical, etc.).
    2. **Ablation Model B:** Temporal + Autoregressive Lags (`lag_1d`, `lag_7d`, `lag_30d`).
    3. **Full Model:** Temporal + Lags + rolling averages (including corrected `rolling_avg_7d` and `rolling_avg_30d`).
  - Compare their MAE, RMSE, and $R^2$ in a new table (Table III) in `final_report.tex` to demonstrate feature importance scientifically.

### Task 3.4: Fix SMOTE Data Leakage (Train/Test Split First)
* **Type:** **Can Fix (Code & Reporting)**
* **Problem:** The classification section applies SMOTE to the entire dataset *before* the train/test split. This causes synthetic training data to contaminate the test fold, severely inflating accuracy metrics.
* **Remediation Task:**
  - In `ml/run_training.py` and `ml/xgboost_model.py`, re-order the pipeline:
    1. First split the raw dataset: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
    2. Apply SMOTE *only* on the training fold: `X_train_res, y_train_res = smote.fit_resample(X_train, y_train)`.
    3. Fit the model on the resampled training set and evaluate on the untouched test set (`X_test, y_test`).
  - Report the corrected, leak-free classification metrics.

### Task 3.5: Document Hyperparameter Search and Validation Strategy
* **Type:** **Can Fix (Reporting)**
* **Problem:** Hyperparameter specifications are listed but the search method, validation split, and random seeds are missing.
* **Remediation Task:**
  - Document in Section IV-B that hyperparameters were tuned using a grid search with 5-fold TimeSeriesSplit on the training set, utilizing a fixed `random_state=42` and early stopping on a 10% validation subset.

---

## 4. Architecture Claims

### Task 4.1: Reframe "Production-Grade" Claims
* **Type:** **Can Fix (Reporting)**
* **Problem:** The architecture is described as "production-grade," but it runs on a single developer machine via Docker Compose with simulated ingestion.
* **Remediation Task:**
  - Soften claims in `final_report.tex` and `PRESENTATION_GUIDE.txt`. Reframe the architecture as a **"Production-Ready Prototype at Single-Node Scale"** that simulates a multi-node cluster (Kafka, Spark, MongoDB, Zookeeper) for educational validation.

### Task 4.2: Clarify Dashboard Diagnostic Fallback
* **Type:** **Can Fix (Reporting)**
* **Problem:** The dashboard generates synthetic data when MongoDB is offline, which could mislead reviewers regarding live data connections.
* **Remediation Task:**
  - Clearly document in Section VI that the dashboard uses Next.js API routes that query the MongoDB container directly, and that the synthetic fallback is a built-in developer diagnostic mode to allow static demonstrations when Docker is offline.

### Task 4.3: Justify Spark/Kafka Educational Scale on Small Datasets
* **Type:** **Can Fix (Reporting)**
* **Problem:** The Carbon Monitor dataset ($109{,}621$ rows) is too small to justify Spark and Kafka, which represents "over-engineering."
* **Remediation Task:**
  - Explicitly address this in Section V-B: admit that daily Carbon Monitor data is easily handled in memory by a single CPU, but justify the use of Kafka and Spark Structured Streaming as an **architectural proof-of-concept**. 
  - Explain that this demonstrates how the pipeline easily scales horizontally to process terabytes of gridded spatial emissions data (such as full EDGAR gridmaps) in high-throughput enterprise deployments.

### Task 4.4: Document MongoDB Time-Series Schema and Collection Benchmarks
* **Type:** **Can Fix (Reporting)**
* **Problem:** MongoDB time-series collection usage is asserted but not evaluated or illustrated with schema schemas.
* **Remediation Task:**
  - Insert a JSON schema diagram in Section III-C showing the time-series collection layout (specifying `timeField: "timestamp"` and `metaField: "metadata"` with `country` and `sector`).
  - Explain why MongoDB time-series collections compress storage and optimize query times by grouping sequential updates in internal column-oriented documents.

---

## 5. Scientific Framing

### Task 5.1: Harmonize Terminology (Emissions vs. Footprint)
* **Type:** **Can Fix (Reporting)**
* **Problem:** The terms "carbon footprint" and "CO₂ emissions" are used interchangeably, creating confusion.
* **Remediation Task:**
  - Standardize terms across all documents:
    - **Macroeconomic Territorial CO₂ Emissions:** Used for Carbon Monitor and EDGAR datasets (measured in $\text{MtCO}_2/\text{day}$ or $\text{MtCO}_2/\text{year}$).
    - **Personal Lifestyle Carbon Footprint:** Used for the Kaggle individual footprint calculator (measured in $CO_2e$ per person).

### Task 5.2: Separate CO₂ and CO₂-Equivalent (CO₂e)
* **Type:** **Can Fix (Reporting)**
* **Problem:** CO₂ and CO₂e are mixed without distinction.
* **Remediation Task:**
  - Add a note in Section I explaining that territorial models predict fossil/cement CO₂ only (Carbon Monitor), whereas the personal footprint is reported in CO₂-equivalent ($CO_2e$) to account for lifestyle greenhouse gases (methane, nitrous oxide, etc.).

### Task 5.3: Soften Policy Support Claims
* **Type:** **Can Fix (Reporting)**
* **Problem:** The report claims the system supports governments, corporations, and the public, without presenting decision-support user tests.
* **Remediation Task:**
  - Reframe Section VII to present the system as a **"Decision-Support Framework"** or **"Feasibility Model"** that has the potential to aid policy formulation, rather than a system actively deployed for real-world governance.

### Task 5.4: Use Rigorous Citations for Global Emissions Claims
* **Type:** **Can Fix (Reporting)**
* **Problem:** The 36.8 Gt global emissions number in 2023 is cited using only the 2020 Carbon Monitor paper.
* **Remediation Task:**
  - Update the citation in Section I to point to the **Global Carbon Budget 2023** (Friedlingstein et al., 2023) or the **IEA Global Energy Review 2023** to validate the 36.8 Gt figure.

---

## 6. Internal Consistency

### Task 6.1: Synchronize XGBoost Regression R² Values
* **Type:** **Can Fix (Reporting)**
* **Problem:** The abstract says $R^2 = 0.9653$, while the SWOT table says $0.9654$.
* **Remediation Task:**
  - Make all values 100% consistent across abstract, Section V, SWOT table, and Next.js front-end using the new leak-free metrics generated after Task 1.2.

### Task 6.2: Align Kafka Topic Names
* **Type:** **Can Fix (Reporting)**
* **Problem:** The architecture TikZ figure says topic `carbon-emissions`, while the Kafka section says `carbon-emissions-daily`.
* **Remediation Task:**
  - Standardize all Kafka topic references across code, text, and TikZ diagram to `carbon-emissions-daily`.

### Task 6.3: Refine Wording around Simulated Real-Time
* **Type:** **Can Fix (Reporting)**
* **Problem:** Section descriptions alternate between claiming "live dashboard" and admitting simulated CSV streaming.
* **Remediation Task:**
  - Standardize all text to refer to the ingestion as a **"Simulated Streaming Pipeline with Daily Replay"** to maintain transparency.

### Task 6.4: Remove Unnecessary Hardware Reproducibility Details
* **Type:** **Can Fix (Reporting)**
* **Problem:** "PyTorch 2.12" and "M1 Apple Silicon MPS" distract from the methodology.
* **Remediation Task:**
  - Remove direct hardware references from the main text in Section IV-C and place them in an implementation footnote to maintain a high-level academic tone.
