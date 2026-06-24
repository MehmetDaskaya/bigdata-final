# =============================================================================
# ml/training_logger.py
# Real-Time ML Training Progress Logger for MongoDB
# =============================================================================

import os
import pymongo
from datetime import datetime

MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:password@localhost:27017/carbon_footprint?authSource=admin")
DB_NAME   = "carbon_footprint"
COLLECTION_NAME = "training_status"

def update_status(
    model_type: str,
    status: str,
    epoch: int = 0,
    total_epochs: int = 0,
    train_losses: list = None,
    val_losses: list = None,
    current_loss: float = None,
    current_val_loss: float = None,
    message: str = "",
    error: str = None
):
    """
    Updates the active training status in MongoDB.
    This enables real-time visual progress monitoring on the dashboard.
    
    Args:
        model_type: Type of model training (e.g. LSTM, XGBoost Regression)
        status: status string ('idle', 'running', 'completed', 'failed')
        epoch: Current training epoch
        total_epochs: Total number of epochs
        train_losses: List of training losses from start
        val_losses: List of validation losses from start
        current_loss: Latest training loss
        current_val_loss: Latest validation loss
        message: Log message / output line
        error: Error message if failed
    """
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Calculate progress percentage
        progress_percentage = 0.0
        if total_epochs > 0:
            progress_percentage = min(float(epoch) / float(total_epochs) * 100.0, 100.0)
        elif status == "completed":
            progress_percentage = 100.0
            
        update_doc = {
            "model_type": model_type,
            "status": status,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "progress_percentage": progress_percentage,
            "message": message,
            "updated_at": datetime.utcnow()
        }
        
        if train_losses is not None:
            update_doc["train_losses"] = train_losses
        if val_losses is not None:
            update_doc["val_losses"] = val_losses
        if current_loss is not None:
            update_doc["current_loss"] = current_loss
        if current_val_loss is not None:
            update_doc["current_val_loss"] = current_val_loss
        if error is not None:
            update_doc["error"] = error
            
        collection.update_one(
            {"_id": "status"},
            {"$set": update_doc},
            upsert=True
        )
        client.close()
    except Exception as e:
        print(f"[Training Logger] Error updating status in MongoDB: {e}")
