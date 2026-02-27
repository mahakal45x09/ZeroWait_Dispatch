from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Add this line
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="KPT & JIT Dispatch API", version="1.0")

# ---  ALLOW THE FRONTEND TO CONNECT ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows any frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 1. Initialize the FastAPI App
app = FastAPI(title="KPT & JIT Dispatch API", version="1.0")

# 2. Load the ML Model and Encoders into memory when the server starts
print("Loading AI Model and Encoders...")
try:
    model = joblib.load('kpt_xgboost_model.pkl')
    cuisine_encoder = joblib.load('cuisine_encoder.pkl')
    city_encoder = joblib.load('city_encoder.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")

# 3. Define the Data Schema (What the incoming JSON should look like)
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

# 4. Create the Prediction Endpoint
@app.post("/predict_dispatch")
def predict_dispatch(order: OrderRequest):
    try:
        # Convert incoming JSON into a dictionary, then a Pandas DataFrame
        order_data = order.dict()
        
        # --- PREPARATION FOR ML MODEL ---
        # The ML model doesn't need rider data or live active orders
        features_for_model = order_data.copy()
        for key in ['current_active_orders', 'rider_distance_to_rest_km', 'rider_avg_speed_kmph']:
            features_for_model.pop(key, None)
            
        df = pd.DataFrame([features_for_model])
        
        # Encode text data
        df['cuisine_encoded'] = cuisine_encoder.transform(df['cuisine_type'])
        df['city_encoded'] = city_encoder.transform(df['city'])
        df = df.drop(columns=['cuisine_type', 'city'])
        
        # --- 1. AI PREDICTION ---
        base_kpt = float(model.predict(df)[0])
        adjusted_kpt = base_kpt
        
        # --- 2. BUSINESS LOGIC ENGINE ---
        applied_rules = []
        
        # Reliability Check
        if order_data['reliability_score'] < 0.75:
            adjusted_kpt += 4.0
            applied_rules.append("Low Reliability Buffer (+4.0m)")
            
        # Kitchen Capacity Check
        if order_data['current_active_orders'] >= order_data['kitchen_capacity']:
            surge = adjusted_kpt * 0.20
            adjusted_kpt += surge
            applied_rules.append(f"Capacity Surge Penalty (+{surge:.1f}m)")
            
        # --- 3. JUST-IN-TIME (JIT) DISPATCH CALCULATION ---
        rider_travel_time = (order_data['rider_distance_to_rest_km'] / order_data['rider_avg_speed_kmph']) * 60
        dispatch_delay_min = adjusted_kpt - rider_travel_time
        
        # If the delay is negative, the rider is very far away, dispatch immediately (0 delay)
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