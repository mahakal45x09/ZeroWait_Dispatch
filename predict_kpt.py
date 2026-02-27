import joblib
import pandas as pd

# ==========================================
# 1. Load the Saved AI Model
# ==========================================
print("Loading the saved KPT model and encoders...")
loaded_model = joblib.load('kpt_xgboost_model.pkl')
loaded_cuisine_encoder = joblib.load('cuisine_encoder.pkl')
loaded_city_encoder = joblib.load('city_encoder.pkl')

# ==========================================
# 2. Simulate a Live Order Request
# ==========================================
# This dictionary represents what the live app sends to your server
live_order_data = {
    'items_count': 5,
    'order_complexity_score': 4,
    'peak_hour_flag': 1,               
    'order_hour': 20,                  # 8 PM Dinner Rush
    'day_of_week': 6,                  # Sunday
    'is_weekend': 1,                   
    'avg_base_prep_time_min': 18.0,
    'kitchen_capacity': 8,             # Max orders they can handle at once
    'current_active_orders': 9,        # LIVE DATA: The kitchen is currently overwhelmed!
    'rush_multiplier': 1.6,
    'is_cloud_kitchen': 0,             
    'rating': 3.9,
    'reliability_score': 0.65,         # Not very reliable (tends to run late)
    'historical_accuracy_score': 0.70,
    'cancellation_bias': 1.2,
    'cuisine_type': 'Chinese',    
    'city': 'Ahmedabad',
    
    # Rider Real-Time Data
    'rider_distance_to_rest_km': 3.5,
    'rider_avg_speed_kmph': 30.0
}

# ==========================================
# 3. Base AI Prediction
# ==========================================
# Format for the ML model (ignoring the rider/live capacity data for now)
features_for_model = live_order_data.copy()
# Remove keys the ML model wasn't trained on
for key in ['current_active_orders', 'rider_distance_to_rest_km', 'rider_avg_speed_kmph']:
    features_for_model.pop(key, None)

df_pred = pd.DataFrame([features_for_model])
df_pred['cuisine_encoded'] = loaded_cuisine_encoder.transform(df_pred['cuisine_type'])
df_pred['city_encoded'] = loaded_city_encoder.transform(df_pred['city'])
df_pred = df_pred.drop(columns=['cuisine_type', 'city'])

# Get the raw AI prediction
base_kpt = loaded_model.predict(df_pred)[0]

# ==========================================
# 4. Business Logic Engine (Reducing Rider Wait Time)
# ==========================================
adjusted_kpt = base_kpt

print(f"\n--- Order Evaluation ---")
print(f"1. AI Base KPT Prediction: {base_kpt:.1f} minutes")

# Strategy A: Reliability Buffer
# If they are historically chaotic (score < 0.75), add a 4-minute buffer to prevent under-prediction
if live_order_data['reliability_score'] < 0.75:
    adjusted_kpt += 4.0
    print("2. Reliability Check: LOW. Added 4.0 min buffer.")

# Strategy B: Kitchen Capacity Surge
# If they have more active orders than their capacity, they are bottlenecked. Add 20% penalty.
if live_order_data['current_active_orders'] >= live_order_data['kitchen_capacity']:
    surge_penalty = adjusted_kpt * 0.20
    adjusted_kpt += surge_penalty
    print(f"3. Capacity Check: OVERWHELMED. Added {surge_penalty:.1f} min surge penalty.")

print(f"-> FINAL Adjusted KPT: {adjusted_kpt:.1f} minutes")

# ==========================================
# 5. Just-In-Time (JIT) Dispatch Calculation
# ==========================================
# Calculate how long it takes the rider to drive to the restaurant
rider_travel_time_min = (live_order_data['rider_distance_to_rest_km'] / live_order_data['rider_avg_speed_kmph']) * 60

print(f"\n--- Dispatch Decision ---")
print(f"Rider Travel Time to Restaurant: {rider_travel_time_min:.1f} minutes")

# Dispatch Delay = Food Prep Time - Rider Travel Time
dispatch_delay = adjusted_kpt - rider_travel_time_min

if dispatch_delay > 0:
    print(f"🚨 ACTION: HOLD DISPATCH. Wait {dispatch_delay:.1f} minutes before assigning the rider.")
else:
    print(f"⚡ ACTION: DISPATCH IMMEDIATELY. The rider is far away and the food will be ready when they arrive.")