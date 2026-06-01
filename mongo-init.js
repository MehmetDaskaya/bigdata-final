// =============================================================================
// mongo-init.js
// MongoDB Initialization Script — Collections and Indexes
// =============================================================================
// This script runs when MongoDB is first initialized.
// It creates collections and indexes in the carbon_footprint database.
// =============================================================================

// Switch to carbon_footprint database
db = db.getSiblingDB('carbon_footprint');

// Create a separate user for application access (for security)
db.createUser({
  user: 'appuser',
  pwd: 'apppassword',
  roles: [{ role: 'readWrite', db: 'carbon_footprint' }]
});

// =============================================================================
// 1. emissions_timeseries — Time-series collection
// Stores daily emission records by country and sector.
// MongoDB 5.0+'s time-series collection type is utilized:
// This type provides specialized optimization for time-series queries.
// =============================================================================
db.createCollection('emissions_timeseries', {
  timeseries: {
    timeField: 'timestamp',          // Timestamp field
    metaField: 'metadata',           // Segmentation metadata (country, sector)
    granularity: 'hours'             // 'hours' granularity for daily data
  },
  expireAfterSeconds: 0              // No automatic deletion (permanent storage)
});

// =============================================================================
// 2. individual_records — Individual carbon footprint collection
// Personal lifestyle records from the Kaggle dataset
// =============================================================================
db.createCollection('individual_records');

// Index for fast searching: query by predicted class and country
db.individual_records.createIndex({ predicted_class: 1 });
db.individual_records.createIndex({ country: 1 });

// =============================================================================
// 3. vehicle_records — Vehicle CO2 emission collection
// Canadian government open data — vehicle class and fuel consumption
// =============================================================================
db.createCollection('vehicle_records');

// Query index by vehicle class and fuel type
db.vehicle_records.createIndex({ vehicle_class: 1, fuel_type: 1 });
db.vehicle_records.createIndex({ co2_emissions_g_per_km: 1 });

// =============================================================================
// 4. ml_predictions — ML model prediction results
// Outputs of LSTM, XGBoost, and MLlib models are stored here
// =============================================================================
db.createCollection('ml_predictions');

// Index for querying based on model type and date
db.ml_predictions.createIndex({ model_type: 1, prediction_date: -1 });
db.ml_predictions.createIndex({ country: 1, sector: 1 });

// =============================================================================
// 5. kafka_stream_log — Real-time records from Kafka
// Temporary storage for raw data from the streaming pipeline
// =============================================================================
db.createCollection('kafka_stream_log');
db.kafka_stream_log.createIndex({ ingested_at: -1 });  // Newest records first

print('MongoDB initialization complete. Collections and indexes created.');
