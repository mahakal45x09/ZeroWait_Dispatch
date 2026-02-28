from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. Initialize the FastAPI App
app = FastAPI(title="KPT & JIT Dispatch API", version="1.0")

# ALLOW THE FRONTEND TO CONNECT 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the ML Model and Encoders
print("Loading AI Model and Encoders...")
try:
    model = joblib.load('kpt_xgboost_model.pkl')
    cuisine_encoder = joblib.load('cuisine_encoder.pkl')
    city_encoder = joblib.load('city_encoder.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")

# 3. Define the Data Schema
class OrderRequest(BaseModel):
    items_count: int
    order_complexity_score: int
    peak_hour_flag: int
    order_hour: int
    day_of_week: int
    is_weekend: int
    avg_base_prep_time_min: float
    kitchen_capacity: int
    current_active_orders: int
    rush_multiplier: float
    is_cloud_kitchen: int
    rating: float
    reliability_score: float
    historical_accuracy_score: float
    cancellation_bias: float
    cuisine_type: str
    city: str
    rider_distance_to_rest_km: float
    rider_avg_speed_kmph: float
    total_pos_kitchen_load: int
    merchant_bias_score: str
    used_iot_button: bool

# 4. Create the Prediction Endpoint
@app.post("/predict_dispatch")
def predict_dispatch(order: OrderRequest):
    try:
        # Convert incoming JSON into a dictionary
        order_data = order.dict()
        
        # PREPARATION FOR ML MODEL 
        # Extract features that the ML model DOES NOT need
        current_active = order_data.pop("current_active_orders", 0)
        rider_dist = order_data.pop("rider_distance_to_rest_km", 0.0)
        rider_speed = order_data.pop("rider_avg_speed_kmph", 1.0) # avoid division by zero
        
        pos_load = order_data.pop("total_pos_kitchen_load", 0)
        bias_score = order_data.pop("merchant_bias_score", "Medium (Standard)")
        iot_button = order_data.pop("used_iot_button", False)
        
        # Now order_data ONLY contains features the XGBoost model expects
        df = pd.DataFrame([order_data])
        
        # Encode text data
        df['cuisine_encoded'] = cuisine_encoder.transform(df['cuisine_type'])
        df['city_encoded'] = city_encoder.transform(df['city'])
        df = df.drop(columns=['cuisine_type', 'city'])
        
        #  1. AI BASE PREDICTION 
        base_kpt = float(model.predict(df)[0])
        adjusted_kpt = base_kpt
        
        # 2. BUSINESS LOGIC ENGINE 
        applied_rules = []
        
        # Rule A: Reliability Check
        if order_data['reliability_score'] < 0.75:
            adjusted_kpt += 4.0
            applied_rules.append("Low Reliability Buffer (+4.0m)")
            
        # Rule B: Kitchen Capacity Check
        if current_active >= order_data['kitchen_capacity']:
            surge = adjusted_kpt * 0.20
            adjusted_kpt += surge
            applied_rules.append(f"Capacity Surge Penalty (+{surge:.1f}m)")
            
        # Rule C:  POS Kitchen Load
        if pos_load > 15:
            penalty = (pos_load - 15) * 0.5
            adjusted_kpt += penalty
            applied_rules.append(f"POS Dine-in Load Surge (+{penalty:.1f}m)")
            
        # Rule D:  Merchant Geo-FOR Bias
        if bias_score == "High (Marks Early)":
            adjusted_kpt += 5.0
            applied_rules.append("High Merchant Bias Buffer (+5.0m)")
        elif bias_score == "Low (Trustworthy)":
            adjusted_kpt -= 2.0
            applied_rules.append("Trustworthy Merchant Reduction (-2.0m)")
            
        # Rule E: ZeroTap IoT Button
        if iot_button:
            adjusted_kpt -= 3.0
            applied_rules.append("ZeroTap IoT Button Used (-3.0m)")
            
        # Ensure KPT never drops below a realistic 5 minutes
        adjusted_kpt = max(5.0, adjusted_kpt)
            
        # --- 3. JUST-IN-TIME (JIT) DISPATCH CALCULATION ---
        rider_travel_time = (rider_dist / rider_speed) * 60
        dispatch_delay_min = adjusted_kpt - rider_travel_time
        
        # If the delay is negative, the rider is very far away, dispatch immediately
        dispatch_delay_min = max(0.0, dispatch_delay_min)
        
        # --- 4. RETURN JSON RESPONSE ---
        return {
            "status": "success",
            "base_predicted_kpt_min": round(base_kpt, 1),
            "final_adjusted_kpt_min": round(adjusted_kpt, 1),
            "rider_travel_time_min": round(rider_travel_time, 1),
            "recommended_dispatch_delay_min": round(dispatch_delay_min, 1),
            "business_rules_applied": applied_rules,
            "action": "HOLD" if dispatch_delay_min > 0 else "DISPATCH_NOW"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))