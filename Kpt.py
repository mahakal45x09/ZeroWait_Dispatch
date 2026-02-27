import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Step 1: Load and Merge Data
# ==========================================
print("Loading datasets...")
orders = pd.read_csv('orders.csv')
restaurants = pd.read_csv('restaurants.csv')
merchant_behavior = pd.read_csv('merchant_behavior.csv')

print("Merging datasets...")
df = orders.merge(restaurants, on='restaurant_id', how='left')
df = df.merge(merchant_behavior, on='restaurant_id', how='left')

# ==========================================
# Step 2: Feature Engineering & Preprocessing
# ==========================================
print("Engineering features...")
df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
df['order_hour'] = df['order_timestamp'].dt.hour
df['day_of_week'] = df['order_timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Encode categorical text variables into numbers
le_cuisine = LabelEncoder()
le_city = LabelEncoder()
df['cuisine_encoded'] = le_cuisine.fit_transform(df['cuisine_type'])
df['city_encoded'] = le_city.fit_transform(df['city'])

# ==========================================
# Step 3: Define Features (X) and Target (y)
# ==========================================
target = 'actual_prep_time_min'

# ONLY use data known at the exact moment the order is placed!
features = [
    'items_count', 'order_complexity_score', 'peak_hour_flag', 
    'order_hour', 'day_of_week', 'is_weekend',
    'avg_base_prep_time_min', 'kitchen_capacity', 'rush_multiplier', 
    'is_cloud_kitchen', 'rating', 'reliability_score', 
    'historical_accuracy_score', 'cancellation_bias',
    'cuisine_encoded', 'city_encoded'
]

X = df[features]
y = df[target]

# ==========================================
# Step 4: Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {len(X_train)} orders, Testing on {len(X_test)} orders.")

# ==========================================
# Step 5: Model Training
# ==========================================
print("Training XGBoost Regressor...")
model = xgb.XGBRegressor(
    n_estimators=200,      
    max_depth=6,           
    learning_rate=0.05,    
    random_state=42,
    n_jobs=-1              
)

model.fit(X_train, y_train)

# ==========================================
# Step 6: Evaluation & Metrics
# ==========================================
print("Evaluating model...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Under-prediction happens when the model predicts KPT will be faster than it actually is.
under_predictions = np.sum(y_pred < y_test)
under_prediction_rate = (under_predictions / len(y_test)) * 100

print("\n--- Model Performance ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
print(f"Under-prediction Rate: {under_prediction_rate:.1f}%")

# ==========================================
# Step 7: Save the Model & Encoders
# ==========================================
print("\nSaving model and encoders to disk...")
joblib.dump(model, 'kpt_xgboost_model.pkl')
joblib.dump(le_cuisine, 'cuisine_encoder.pkl')
joblib.dump(le_city, 'city_encoder.pkl')
print("Saved successfully! You can now deploy 'kpt_xgboost_model.pkl'.")

# ==========================================
# Step 8: Visualizing Feature Importance
# ==========================================
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Top Drivers of Kitchen Prep Time')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()