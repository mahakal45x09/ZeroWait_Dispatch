import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

import config
from business_rules import apply_business_rules

# ── Logging Setup ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zerowait")

# ── 1. Initialize the FastAPI App ───────────────────────────────────
app = FastAPI(
    title="KPT & JIT Dispatch API",
    version=config.MODEL_VERSION,
    description="AI-powered Just-In-Time delivery dispatch engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. Load the ML Model and Encoders ──────────────────────────────
logger.info("Loading AI Model and Encoders...")
model = None
cuisine_encoder = None
city_encoder = None

try:
    model = joblib.load(config.MODEL_PATH)
    cuisine_encoder = joblib.load(config.CUISINE_ENCODER_PATH)
    city_encoder = joblib.load(config.CITY_ENCODER_PATH)
    logger.info("Model loaded successfully from %s", config.MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model files: %s", e)


# ── 3. Data Schema ─────────────────────────────────────────────────
class OrderRequest(BaseModel):
    items_count: int = Field(ge=1, le=50, description="Number of items in the order")
    order_complexity_score: int = Field(ge=1, le=5, description="Order complexity from 1-5")
    peak_hour_flag: int = Field(ge=0, le=1)
    order_hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: int = Field(ge=0, le=1)
    avg_base_prep_time_min: float = Field(gt=0, le=120)
    kitchen_capacity: int = Field(ge=1, le=100)
    current_active_orders: int = Field(ge=0, le=200)
    rush_multiplier: float = Field(ge=0.5, le=3.0)
    is_cloud_kitchen: int = Field(ge=0, le=1)
    rating: float = Field(ge=1.0, le=5.0)
    reliability_score: float = Field(ge=0.0, le=1.0)
    historical_accuracy_score: float = Field(ge=0.0, le=1.0)
    cancellation_bias: float = Field(ge=0.0, le=5.0)
    cuisine_type: str
    city: str
    rider_distance_to_rest_km: float = Field(ge=0.0, le=50.0)
    rider_avg_speed_kmph: float = Field(gt=0, le=120, description="Must be > 0 to prevent division by zero")
    total_pos_kitchen_load: int = Field(ge=0, le=100)
    merchant_bias_score: str
    used_iot_button: bool
    live_weather_condition: str


# ── 4. Health & Info Endpoints ──────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "version": config.MODEL_VERSION,
    }


@app.get("/model/info")
def model_info():
    """Returns metadata about the loaded ML model."""
    return {
        "model_type": config.MODEL_DESCRIPTION,
        "version": config.MODEL_VERSION,
        "features": config.FEATURE_COUNT,
        "training_samples": config.TRAINING_DATASET_SIZE,
        "status": "loaded" if model is not None else "not_loaded",
    }


# ── 5. Prediction Endpoint ─────────────────────────────────────────
@app.post("/predict_dispatch")
def predict_dispatch(order: OrderRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="ML model is not loaded")

    try:
        # Convert incoming JSON into a dictionary
        order_data = order.model_dump()
        
        # Extract features that the ML model DOES NOT need (used by business rules & JIT)
        current_active = order_data.pop("current_active_orders", 0)
        rider_dist = order_data.pop("rider_distance_to_rest_km", 0.0)
        rider_speed = order_data.pop("rider_avg_speed_kmph", 1.0)
        
        pos_load = order_data.pop("total_pos_kitchen_load", 0)
        bias_score = order_data.pop("merchant_bias_score", "Medium (Standard)")
        iot_button = order_data.pop("used_iot_button", False)
        weather = order_data.pop("live_weather_condition", "Clear")

        # Now order_data ONLY contains features the XGBoost model expects
        df = pd.DataFrame([order_data])
        
        # Encode text data
        df['cuisine_encoded'] = cuisine_encoder.transform(df['cuisine_type'])
        df['city_encoded'] = city_encoder.transform(df['city'])
        df = df.drop(columns=['cuisine_type', 'city'])
        
        # ── AI Base Prediction ──────────────────────────────────────
        base_kpt = float(model.predict(df)[0])
        
        # ── Business Rules Engine ───────────────────────────────────
        rules_context = {
            "reliability_score": order_data.get("reliability_score", 1.0),
            "current_active_orders": current_active,
            "kitchen_capacity": order_data.get("kitchen_capacity", 999),
            "total_pos_kitchen_load": pos_load,
            "merchant_bias_score": bias_score,
            "used_iot_button": iot_button,
            "live_weather_condition": weather,
        }
        adjusted_kpt, applied_rules = apply_business_rules(base_kpt, rules_context)
            
        # ── JIT Dispatch Calculation ────────────────────────────────
        rider_travel_time = (rider_dist / rider_speed) * 60
        dispatch_delay_min = max(0.0, adjusted_kpt - rider_travel_time)

        action = "HOLD" if dispatch_delay_min > 0 else "DISPATCH_NOW"
        
        logger.info(
            "Prediction: base=%.1f adj=%.1f travel=%.1f delay=%.1f action=%s rules=%s",
            base_kpt, adjusted_kpt, rider_travel_time, dispatch_delay_min, action, applied_rules,
        )
        
        return {
            "status": "success",
            "base_predicted_kpt_min": round(base_kpt, 1),
            "final_adjusted_kpt_min": round(adjusted_kpt, 1),
            "rider_travel_time_min": round(rider_travel_time, 1),
            "recommended_dispatch_delay_min": round(dispatch_delay_min, 1),
            "business_rules_applied": applied_rules,
            "action": action,
        }
        
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))