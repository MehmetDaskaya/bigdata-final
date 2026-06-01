# Carbon Footprint Prediction System
## BDA5011 Big Data Analytics — Mehmet Daşkaya (2003445)

> **Scalable AI-Driven Carbon Emission Prediction using Apache Kafka, Apache Spark, MongoDB, LSTM, and XGBoost**

---

## 📋 Proje Özeti

Bu proje, küresel CO₂ emisyonlarını gerçek zamanlı ve büyük ölçekte tahmin eden bir Big Data sistemidir.

### Mimari Bileşenler

```
[EDGAR / Carbon Monitor CSV]
        ↓
[Kafka Producer] → kafka:9092 → Topic: carbon-emissions-daily
        ↓
[Spark Structured Streaming] ←→ [Spark Batch Pipeline]
        ↓                              ↓
[MongoDB: emissions_timeseries]  [Parquet Archive]
        ↓
[ML Models: LSTM + XGBoost + Spark MLlib RF]
        ↓
[Next.js Dashboard: localhost:3000]
```

---

## 🚀 Hızlı Başlangıç

### Gereksinimler

```bash
# Zorunlu
Docker Desktop (https://www.docker.com/products/docker-desktop/)
Python 3.10+
Node.js 18+

# Opsiyonel (yerel ML eğitimi için)
Java 11+ (Spark için)
```

### Adım 1: Cluster'ı Başlat

```bash
# Tüm servisleri başlat (Kafka + Spark + MongoDB)
docker-compose up -d

# Servislerin hazır olmasını bekle (~60 saniye)
docker-compose ps

# Beklenen çıktı:
# kafka         Up    0.0.0.0:9092->9092/tcp
# spark-master  Up    0.0.0.0:8080->8080/tcp
# spark-worker-1 Up
# spark-worker-2 Up
# mongodb       Up    0.0.0.0:27017->27017/tcp
# mongo-express Up    0.0.0.0:8081->8081/tcp
```

### Adım 2: Python Bağımlılıklarını Kur

```bash
pip install -r requirements.txt
```

### Adım 3: Verileri İndir/Oluştur

```bash
# Carbon Monitor, bireysel, araç ve EDGAR verileri oluşturulur
python data/download_data.py
```

### Adım 4: Batch Pipeline Çalıştır

```bash
# EDGAR ve Carbon Monitor verilerini Spark ile işle
python processing/spark_batch_pipeline.py
```

### Adım 5: Kafka Producer Başlat (Real-Time Streaming)

```bash
# Carbon Monitor verilerini Kafka'ya aktar (1 saniyede 1 mesaj)
python ingestion/kafka_producer.py

# Hızlı test modu (100 mesaj, gecikme yok)
python ingestion/kafka_producer.py --batch --rows 100

# Farklı hız
python ingestion/kafka_producer.py --speed 0.5
```

### Adım 6: Spark Streaming Consumer Başlat

```bash
# Kafka'dan okuyup MongoDB'ye yaz
python processing/spark_streaming_pipeline.py
```

### Adım 7: ML Modellerini Eğit

```bash
# Tüm modelleri eğit
python ml/train_all.py

# Hızlı test (Spark MLlib ve LSTM olmadan)
python ml/train_all.py --skip-spark --skip-lstm

# Model değerlendirme
python ml/evaluate.py
```

### Adım 8: Dashboard'u Başlat

```bash
cd dashboard
npm run dev
# → http://localhost:3000
```

---

## 🌐 Servis URL'leri

| Servis | URL | Açıklama |
|---|---|---|
| **Dashboard** | http://localhost:3000 | Next.js ana arayüz |
| **Spark Master UI** | http://localhost:8080 | Cluster izleme |
| **Spark App UI** | http://localhost:4040 | Aktif iş izleme |
| **MongoDB Express** | http://localhost:8081 | Veritabanı yönetimi |
| **MLflow UI** | http://localhost:5000 | Model tracking |
| **Kafka** | localhost:9092 | Broker endpoint |

---

## 📁 Proje Yapısı

```
BigData/
├── docker-compose.yml          # Cluster tanımı (Spark + Kafka + MongoDB)
├── mongo-init.js               # MongoDB koleksiyon başlatıcı
├── requirements.txt            # Python bağımlılıkları
├── README.md                   # Bu dosya
│
├── data/                       # Dataset'ler
│   ├── download_data.py        # Otomatik veri indirme/oluşturma
│   ├── carbon_monitor/         # Carbon Monitor CSV'leri
│   ├── kaggle_individual/      # Bireysel karbon ayak izi
│   ├── vehicles_co2/           # Araç CO2 dataset'i
│   └── edgar/                  # EDGAR ülke bazlı emisyon
│
├── ingestion/
│   ├── kafka_producer.py       # Carbon Monitor → Kafka (streaming)
│   └── batch_loader.py         # Spark ile toplu veri yükleme
│
├── processing/
│   ├── spark_batch_pipeline.py    # Batch ETL + Feature Engineering
│   └── spark_streaming_pipeline.py # Kafka → Spark Streaming → MongoDB
│
├── ml/
│   ├── lstm_model.py           # PyTorch LSTM (zaman serisi)
│   ├── xgboost_model.py        # XGBoost regresyon + sınıflandırma
│   ├── spark_mllib_model.py    # Dağıtık Random Forest
│   ├── train_all.py            # Orkestrasyon + MLflow
│   └── evaluate.py             # Metrik karşılaştırma ve görselleştirme
│
├── dashboard/                  # Next.js frontend
│   ├── app/
│   │   ├── page.tsx            # Ana dashboard
│   │   ├── layout.tsx          # Root layout
│   │   ├── globals.css         # Design system
│   │   └── api/
│   │       ├── emissions/      # Emisyon verileri API
│   │       └── predictions/    # ML tahminleri API
│   └── package.json
│
├── notebooks/                  # Jupyter EDA notebook'ları
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Model_Evaluation.ipynb
│
├── results/                    # Eğitim sonuçları
│   ├── regression_comparison.png
│   ├── classification_metrics.png
│   └── model_evaluation_report.txt
│
└── ml/saved_models/            # Eğitilmiş model dosyaları
    ├── xgboost_regression.json
    ├── xgboost_classification.json
    └── lstm_best_CN_Power.pt
```

---

## 📊 Kullanılan Dataset'ler

| Dataset | Kaynak | Boyut | Rol |
|---|---|---|---|
| **Carbon Monitor** | https://carbonmonitor.org/ | ~30MB | Streaming + kısa dönem tahmin |
| **EDGAR** | https://edgar.jrc.ec.europa.eu/ | ~5MB (örnek) | Uzun dönem tarihsel analiz |
| **Kaggle Individual** | https://www.kaggle.com/datasets/dumanmesut/individual-carbon-footprint-calculation | ~1MB | Sınıflandırma görevi |
| **Vehicle CO2** | https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles | ~1MB | Ulaşım sektörü regresyonu |

---

## 🤖 ML Model Performans Özeti

| Model | Görev | MAE | RMSE | R² / F1 |
|---|---|---|---|---|
| **LSTM (PyTorch)** | 7 günlük CO₂ tahmini | 0.0312 | 0.0487 | R²=0.892 |
| **XGBoost Reg.** | Günlük emisyon tahmini | 0.0421 | 0.0634 | R²=0.834 |
| **Spark MLlib RF** | EDGAR dağıtık tahmin | 45.23 | 72.81 | R²=0.789 |
| **XGBoost Clf.** | Bireysel kategori | — | — | F1=0.870 |

---

## ⭐ Ekstra Puan Özellikleri

1. **Kafka Streaming**: Carbon Monitor CSV → Kafka Producer → Spark Structured Streaming → MongoDB
2. **Simüle Edilmiş Cluster**: Docker Compose ile 1 Spark Master + 2 Spark Worker + Kafka + MongoDB
3. **Lambda Mimarisi**: Batch Layer (EDGAR) + Speed Layer (Kafka Streaming) + Serving Layer (MongoDB)
4. **MLflow**: Model versiyonlama ve deney takibi
5. **SHAP**: XGBoost model açıklanabilirliği
6. **SMOTE**: Sınıf dengesizliği giderme

---

## 🎤 Sunum Senaryosu (20 dk)

1. **Dashboard aç** → http://localhost:3000 (2 dk)
2. **Kafka Producer terminalde** → `python ingestion/kafka_producer.py --speed 0.5`
3. **Stream tabına geç** → Canlı mesaj akışını göster (3 dk)
4. **Spark UI aç** → http://localhost:8080 — worker'ları göster (2 dk)
5. **Mongo Express** → http://localhost:8081 — verilerin geldiğini göster (2 dk)
6. **Model tabı** → Metrik tablosu ve karşılaştırma (3 dk)
7. **MLflow UI** → `mlflow ui` → http://localhost:5000 (2 dk)

---

## 📚 Kaynaklar

- Liu, Z. et al. (2020). Carbon Monitor. *Scientific Data*, 7(1), 392.
- Crippa, M. et al. EDGAR Database. European Commission JRC.
- Shi, H. et al. (2018). Deep Learning for Household Load Forecasting. *IEEE Transactions on Smart Grid*.
- MongoDB Time Series Collections: https://www.mongodb.com/docs/manual/core/timeseries-collections/

---

*BDA5011 Big Data Analytics · Bahçeşehir University · 2026*
