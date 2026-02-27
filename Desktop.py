
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# --- Parameters ---
NUM_RESTAURANTS = 300
NUM_ORDERS = 10000
NUM_RIDERS = 500
CITIES = ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot']
CUISINES = ['North Indian', 'Chinese', 'Fast Food', 'South Indian', 'Continental']

# ==========================================
# 1. RESTAURANTS DATASET
# ==========================================
restaurants = pd.DataFrame({
    'restaurant_id': range(1, NUM_RESTAURANTS + 1),
    'restaurant_name': [f"Restaurant_{i}" for i in range(1, NUM_RESTAURANTS + 1)],
    'city': np.random.choice(CITIES, NUM_RESTAURANTS),
    'cuisine_type': np.random.choice(CUISINES, NUM_RESTAURANTS),
    'avg_base_prep_time_min': np.random.uniform(8, 45, NUM_RESTAURANTS).round(1),
    'kitchen_capacity': np.random.randint(5, 26, NUM_RESTAURANTS),
    'rush_multiplier': np.random.uniform(1.0, 1.8, NUM_RESTAURANTS).round(2),
    'is_cloud_kitchen': np.random.choice([0, 1], NUM_RESTAURANTS, p=[0.8, 0.2]),
    'weekday_load_factor': np.random.uniform(0.8, 1.3, NUM_RESTAURANTS).round(2),
    'weekend_load_factor': np.random.uniform(1.0, 1.6, NUM_RESTAURANTS).round(2),
    'cancel_rate_percent': np.random.uniform(1, 15, NUM_RESTAURANTS).round(1),
    'avg_order_value': np.random.uniform(150, 800, NUM_RESTAURANTS).round(2),
    'rating': np.random.uniform(3.0, 5.0, NUM_RESTAURANTS).round(1)
})
restaurants['seating_capacity'] = restaurants['is_cloud_kitchen'].apply(
    lambda x: 0 if x == 1 else np.random.randint(10, 151)
)
restaurants['avg_daily_orders'] = (
    np.random.randint(50, 300, NUM_RESTAURANTS) * (restaurants['rating'] / 3.0)
).astype(int).clip(50, 500)

restaurants.to_csv('restaurants.csv', index=False)

# ==========================================
# 2. MERCHANT BEHAVIOR DATASET
# ==========================================
merchant_behavior = pd.DataFrame({
    'restaurant_id': restaurants['restaurant_id'],
    'marking_bias_mean_min': np.random.uniform(-5, 10, NUM_RESTAURANTS).round(1),
    'marking_bias_std_min': np.random.uniform(0.5, 4.0, NUM_RESTAURANTS).round(1),
    'on_time_marking_rate': np.random.uniform(0.5, 0.95, NUM_RESTAURANTS).round(2),
    'reliability_score': (restaurants['rating'] / 5.0 * np.random.uniform(0.8, 1.0, NUM_RESTAURANTS)).round(2).clip(0.4, 0.95),
    'avg_confirmation_delay_min': np.random.uniform(0, 5, NUM_RESTAURANTS).round(1),
    'last_30_day_variance': np.random.uniform(1.0, 5.0, NUM_RESTAURANTS).round(2),
    'peak_hour_delay_factor': np.random.uniform(1.1, 2.0, NUM_RESTAURANTS).round(2),
    'cancellation_bias': np.random.uniform(0.5, 2.0, NUM_RESTAURANTS).round(2),
    'historical_accuracy_score': np.random.uniform(0.6, 0.99, NUM_RESTAURANTS).round(2)
})

merchant_behavior.to_csv('merchant_behavior.csv', index=False)

# ==========================================
# 3. RIDERS DATASET
# ==========================================
riders = pd.DataFrame({
    'rider_id': range(1, NUM_RIDERS + 1),
    'city': np.random.choice(CITIES, NUM_RIDERS),
    'vehicle_type': np.random.choice(['Bike', 'Scooter'], NUM_RIDERS),
    'experience_years': np.random.randint(1, 7, NUM_RIDERS),
    'avg_speed_kmph': np.random.uniform(20, 45, NUM_RIDERS).round(1),
    'current_distance_km': np.random.uniform(0, 5, NUM_RIDERS).round(1),
    'active_orders': np.random.randint(0, 4, NUM_RIDERS),
    'availability_status': np.random.choice([0, 1], NUM_RIDERS, p=[0.2, 0.8]),
    'rider_rating': np.random.uniform(3.5, 5.0, NUM_RIDERS).round(1),
    'fatigue_index': np.random.uniform(0, 1, NUM_RIDERS).round(2),
    'shift_hours_today': np.random.uniform(0, 10, NUM_RIDERS).round(1),
    'acceptance_rate_percent': np.random.uniform(70, 100, NUM_RIDERS).round(1)
})

riders.to_csv('riders.csv', index=False)

# ==========================================
# 4. ORDERS DATASET (UPDATED)
# ==========================================
base_date = datetime(2023, 10, 1, 12, 0, 0)
timestamps = [base_date + timedelta(minutes=np.random.randint(0, 43200)) for _ in range(NUM_ORDERS)]

orders = pd.DataFrame({
    'order_id': range(1, NUM_ORDERS + 1),
    'restaurant_id': np.random.choice(restaurants['restaurant_id'], NUM_ORDERS),
    'rider_id': np.random.choice(riders['rider_id'], NUM_ORDERS),
    'order_timestamp': timestamps,
    'items_count': np.random.randint(1, 9, NUM_ORDERS),
    'order_complexity_score': np.random.randint(1, 6, NUM_ORDERS),
    'distance_km': np.random.uniform(1, 12, NUM_ORDERS).round(1),
    'traffic_level': np.random.choice(['Low', 'Medium', 'High'], NUM_ORDERS, p=[0.3, 0.5, 0.2]),
    'weather_condition': np.random.choice(['Clear', 'Rain', 'Heavy Rain'], NUM_ORDERS, p=[0.8, 0.15, 0.05]),
    'order_status': np.random.choice(['Delivered', 'Cancelled'], NUM_ORDERS, p=[0.95, 0.05])
})

orders['peak_hour_flag'] = orders['order_timestamp'].apply(
    lambda x: 1 if (12 <= x.hour <= 14) or (19 <= x.hour <= 21) else 0
)

# Merge necessary data
orders = orders.merge(restaurants[['restaurant_id', 'avg_base_prep_time_min', 'rush_multiplier', 'kitchen_capacity']], on='restaurant_id')
orders = orders.merge(merchant_behavior[['restaurant_id', 'reliability_score']], on='restaurant_id')
orders = orders.merge(riders[['rider_id', 'avg_speed_kmph']], on='rider_id')

# --- NEW FEATURES: Map Routing & Traffic Signals ---

# Map routing efficiency (0.7 to 1.0, where 1.0 is perfect GPS/routing alignment)
orders['map_routing_efficiency'] = np.random.uniform(0.7, 1.0, NUM_ORDERS).round(2)

# Number of traffic signals scales with distance and urban density
orders['num_traffic_signals'] = (orders['distance_km'] * np.random.uniform(0.5, 3.0, NUM_ORDERS)).astype(int)

# Signal wait time depends on traffic level and number of signals
traffic_signal_multiplier = orders['traffic_level'].map({'Low': 0.3, 'Medium': 0.8, 'High': 1.5})
orders['signal_wait_time_min'] = (orders['num_traffic_signals'] * np.random.uniform(0.2, 0.6, NUM_ORDERS) * traffic_signal_multiplier).round(1)

# --------------------------------------------------

prep_noise = np.random.normal(0, 2, NUM_ORDERS) / orders['reliability_score'] 
orders['actual_prep_time_min'] = (
    orders['avg_base_prep_time_min'] + 
    (orders['order_complexity_score'] * 1.5) + 
    (orders['peak_hour_flag'] * orders['rush_multiplier'] * 4.0) - 
    (orders['kitchen_capacity'] * 0.3) + 
    prep_noise
).clip(lower=5.0).round(1)

# Updated Delivery Time Logic to include map detours and signal waiting
traffic_penalty = orders['traffic_level'].map({'Low': 1.0, 'Medium': 1.3, 'High': 1.8})
weather_penalty = orders['weather_condition'].map({'Clear': 1.0, 'Rain': 1.4, 'Heavy Rain': 1.8})

orders['actual_delivery_time_min'] = (
    # Base travel time inflated by map inefficiency
    ((orders['distance_km'] / orders['avg_speed_kmph'] * 60) / orders['map_routing_efficiency']) * traffic_penalty * weather_penalty + 
    # Add time spent physically waiting at red lights
    orders['signal_wait_time_min']
).round(1)

actual_rider_wait = np.random.uniform(1, 10, NUM_ORDERS)
orders['actual_total_time_min'] = (orders['actual_prep_time_min'] + orders['actual_delivery_time_min'] + actual_rider_wait).round(1)
orders['customer_eta_given_min'] = (orders['actual_total_time_min'] * np.random.uniform(0.9, 1.2, NUM_ORDERS)).round(0)

# Clean up order columns
orders = orders[['order_id', 'restaurant_id', 'rider_id', 'order_timestamp', 'items_count', 'order_complexity_score', 
                 'map_routing_efficiency', 'num_traffic_signals', 'signal_wait_time_min', # Added new columns here
                 'actual_prep_time_min', 'actual_delivery_time_min', 'distance_km', 'traffic_level', 
                 'weather_condition', 'peak_hour_flag', 'order_status', 'customer_eta_given_min', 'actual_total_time_min']]

orders.sort_values('order_id', inplace=True)
orders.to_csv('orders.csv', index=False)

# ==========================================
# 5. DISPATCH LOG DATASET
# ==========================================
dispatch_log = pd.DataFrame({'order_id': orders['order_id']})

dispatch_log['predicted_kpt_old'] = (orders['actual_prep_time_min'] * np.random.uniform(0.6, 1.5, NUM_ORDERS)).round(1)
dispatch_log['rider_wait_old'] = np.where(dispatch_log['predicted_kpt_old'] < orders['actual_prep_time_min'], 
                                          orders['actual_prep_time_min'] - dispatch_log['predicted_kpt_old'] + np.random.uniform(2,5), 
                                          np.random.uniform(0, 3, NUM_ORDERS)).round(1)
dispatch_log['predicted_eta_old'] = (dispatch_log['predicted_kpt_old'] + orders['actual_delivery_time_min'] + dispatch_log['rider_wait_old']).round(1)

dispatch_log['predicted_kpt_new'] = (orders['actual_prep_time_min'] + np.random.normal(0, 2, NUM_ORDERS)).clip(lower=5).round(1)
dispatch_log['rider_wait_new'] = np.random.exponential(2, NUM_ORDERS).round(1).clip(0, 8)
dispatch_log['predicted_eta_new'] = (dispatch_log['predicted_kpt_new'] + orders['actual_delivery_time_min'] + dispatch_log['rider_wait_new']).round(1)

dispatch_log['eta_error_old'] = abs(dispatch_log['predicted_eta_old'] - orders['actual_total_time_min']).round(1)
dispatch_log['eta_error_new'] = abs(dispatch_log['predicted_eta_new'] - orders['actual_total_time_min']).round(1)

dispatch_log['kpt_improvement_min'] = (abs(dispatch_log['predicted_kpt_old'] - orders['actual_prep_time_min']) - 
                                       abs(dispatch_log['predicted_kpt_new'] - orders['actual_prep_time_min'])).round(1)
dispatch_log['eta_accuracy_improvement_min'] = (dispatch_log['eta_error_old'] - dispatch_log['eta_error_new']).round(1)
dispatch_log['rider_idle_time_reduction_min'] = (dispatch_log['rider_wait_old'] - dispatch_log['rider_wait_new']).round(1)

dispatch_log['dispatch_strategy'] = np.random.choice(['Old_Static', 'AI_Dynamic'], NUM_ORDERS, p=[0.2, 0.8])
dispatch_log['system_load_score'] = np.random.uniform(0.1, 1.0, NUM_ORDERS).round(2)
dispatch_log['fairness_index'] = np.random.uniform(0.7, 1.0, NUM_ORDERS).round(2)

dispatch_log.to_csv('dispatch_log.csv', index=False)

print("Updated datasets with Routing & Traffic Signals generated successfully!")