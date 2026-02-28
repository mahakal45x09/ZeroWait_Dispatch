import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="ZeroWait Dispatch", page_icon="images.jpeg", layout="wide")
st.title("ZeroWait Dispatch Dashboard")

# 2. Setup the Two-Column Layout
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Live Order Details")
    
    # Create a form so the page doesn't reload on every single keystroke
    with st.form("order_form"):
        row1_col1, row1_col2 = st.columns(2)
        cuisine_type = row1_col1.selectbox("Cuisine Type", ["Fast Food", "North Indian", "South Indian", "Chinese", "Continental"])
        city = row1_col2.selectbox("City", ["Ahmedabad", "Vadodra", "Rajkot", "Surat"])

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        items_count = row2_col1.number_input("Items Count", min_value=1, value=5)
        order_complexity_score = row2_col2.number_input("Complexity (1-5)", min_value=1, max_value=5, value=3)
        avg_base_prep_time_min = row2_col3.number_input("Base Prep (Min)", min_value=1.0, value=20.0)
        
        row3_col1, row3_col2 = st.columns(2)
        kitchen_capacity = row3_col1.number_input("Kitchen Capacity", min_value=1, value=10)
        current_active_orders = row3_col2.number_input("Current Active Orders", min_value=0, value=12)
        
        row4_col1, row4_col2 = st.columns(2)
        rider_distance_to_rest_km = row4_col1.number_input("Rider Distance (km)", min_value=0.1, value=3.5)
        rider_avg_speed_kmph = row4_col2.number_input("Rider Speed (km/h)", min_value=1.0, value=30.0)
        
        #  ZOMATO SIGNAL ENRICHMENT 
        st.markdown("---")
        st.markdown("#### 📡 Zomato Signal Enrichment")
        row5_col1, row5_col2 = st.columns(2)
        
        total_pos_kitchen_load = row5_col1.slider("Total POS Active Tickets (Dine-in + Apps)", min_value=0, max_value=50, value=10)
        merchant_bias_score = row5_col2.selectbox(
            "Historical Geo-FOR Bias Score",
            options=["Low (Trustworthy)", "Medium (Standard)", "High (Marks Early)"],
            index=1
        )
        used_iot_button = st.checkbox("🛎️ Merchant used 'ZeroTap' IoT Button", value=False)
        
        #  REAL WORLD CHAOS FACTORS
        st.markdown("---")
        st.markdown("#### 🌧️ Real-World Chaos Factors")
        live_weather_condition = st.selectbox(
            "Live Local Weather (API Simulation)", 
            ["Clear", "Light Rain", "Heavy Rain / Waterlogging"],
            help="Heavy rain slows down kitchen operations and rider transit."
        ) 
        st.markdown("---")
        
        # The submit button
        submit_button = st.form_submit_button(label="🚀 Calculate AI Dispatch Time", use_container_width=True)

with col2:
    st.subheader("AI Decision Engine")
    
    # 3. What happens when the user clicks the button
    if submit_button:
        # Build the exact dictionary your FastAPI expects
        order_data = {
            "cuisine_type": cuisine_type,
            "city": city,
            "items_count": items_count,
            "order_complexity_score": order_complexity_score,
            "avg_base_prep_time_min": avg_base_prep_time_min,
            "kitchen_capacity": kitchen_capacity,
            "current_active_orders": current_active_orders,
            "rider_distance_to_rest_km": rider_distance_to_rest_km,
            "rider_avg_speed_kmph": rider_avg_speed_kmph,
            
            "total_pos_kitchen_load": total_pos_kitchen_load,
            "merchant_bias_score": merchant_bias_score,
            "used_iot_button": used_iot_button,
            "live_weather_condition": live_weather_condition, # moved this here where it belongs!
            
            # Hidden fields (hardcoded defaults)
            "peak_hour_flag": 1,
            "order_hour": 19,
            "day_of_week": 5,
            "is_weekend": 1,
            "rush_multiplier": 1.5,
            "is_cloud_kitchen": 0,
            "rating": 4.2,
            "reliability_score": 0.80,
            "historical_accuracy_score": 0.85,
            "cancellation_bias": 1.0
        }
        
        with st.spinner("Connecting to AI Model..."):
            try:
                # Send the data to your FastAPI backend
                response = requests.post('https://zerowait-api-5xfg.onrender.com/predict_dispatch', json=order_data)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.write(f"**Raw AI Prep Time:** {data['base_predicted_kpt_min']} min")
                    st.write(f"**Adjusted KPT:** {data['final_adjusted_kpt_min']} min")
                    
                    if data.get('business_rules_applied'):
                        rules = "\n".join([f"- {rule}" for rule in data['business_rules_applied']])
                        st.info(f"**System Adjustments Applied:**\n{rules}")
                    
                    st.divider()
                    st.write(f"**Rider Travel Time:** {data['rider_travel_time_min']} min")
                    
                    st.markdown("<h5 style='text-align: center; color: gray;'>System Recommendation:</h5>", unsafe_allow_html=True)
                    
                    # Check action to color code the output
                    if data['action'] == "HOLD":
                        st.markdown(f"<h3 style='text-align: center; color: #dc3545;'>🛑 HOLD DISPATCH</h3>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='text-align: center;'>Wait <span style='color: #0d6efd;'>{data['recommended_dispatch_delay_min']}</span> minutes to dispatch</h4>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='text-align: center; color: #198754;'>⚡ DISPATCH NOW</h3>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='text-align: center;'>Wait <span style='color: #0d6efd;'>0</span> minutes to dispatch</h4>", unsafe_allow_html=True)
                        
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to API. Is your Uvicorn server running on port 8000?")