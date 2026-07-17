import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path

from business_rules import apply_business_rules

# ── Load Lottie Animation ────────────────────────────────────────────
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_url = "https://assets8.lottiefiles.com/packages/lf20_xrmpegx2.json"
lottie_animation = load_lottieurl(lottie_url)

# ── Load ML Model (cached so it only runs once) ─────────────────────
@st.cache_resource
def load_model():
    """Load the XGBoost model and label encoders from disk."""
    base = Path(__file__).resolve().parent
    model = joblib.load(base / "kpt_xgboost_model.pkl")
    cuisine_enc = joblib.load(base / "cuisine_encoder.pkl")
    city_enc = joblib.load(base / "city_encoder.pkl")
    return model, cuisine_enc, city_enc

model, cuisine_encoder, city_encoder = load_model()

# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(page_title="ZeroWait Dispatch", page_icon="🚀", layout="wide")

# ── Custom CSS for Premium Styling ───────────────────────────────────
st.markdown("""
<style>
    /* Dark theme adjustments */
    .stApp { background-color: #0f1117; }
    
    /* Metric cards */
    .metric-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(6,182,212,0.05));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-box .value {
        font-size: 32px;
        font-weight: 700;
        color: #818cf8;
    }
    .metric-box .label {
        font-size: 12px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    /* Action banners */
    .hold-banner {
        background: linear-gradient(135deg, #f43f5e, #e11d48);
        color: white;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        margin: 16px 0;
    }
    .dispatch-banner {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        margin: 16px 0;
    }
    .action-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }
    .action-detail {
        font-size: 18px;
        opacity: 0.9;
    }
    .action-detail strong {
        font-size: 24px;
    }
    
    /* Rule tags */
    .rule-tag {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        color: #818cf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        margin: 4px 4px 4px 0;
        font-weight: 500;
    }
    
    /* Header gradient */
    .header-gradient {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 11px;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 8px 0;
        border-bottom: 1px solid rgba(75,85,99,0.3);
        margin: 16px 0 12px;
    }
    
    /* Form submit button override */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 12px !important;
        border-radius: 12px !important;
    }
    .stFormSubmitButton > button:hover {
        box-shadow: 0 8px 25px rgba(99,102,241,0.3) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────
st.markdown('<div class="header-gradient">⚡ ZeroWait Dispatch</div>', unsafe_allow_html=True)
st.caption("AI-Powered Just-In-Time Delivery Engine  •  ML Model + Business Rules + JIT Dispatch")

st.markdown("---")

# ── Two-Column Layout ────────────────────────────────────────────────
col1, col2 = st.columns([1.4, 1])

with col1:
    st.subheader("📋 Live Order Details")
    
    with st.form("order_form"):
        # ── Order Basics ─────────────────────────────────────────
        row1_col1, row1_col2 = st.columns(2)
        cuisine_type = row1_col1.selectbox("Cuisine Type", ["Fast Food", "North Indian", "South Indian", "Chinese", "Continental"])
        city = row1_col2.selectbox("City", ["Ahmedabad", "Vadodara", "Rajkot", "Surat"])

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        items_count = row2_col1.number_input("Items Count", min_value=1, max_value=50, value=5)
        order_complexity_score = row2_col2.number_input("Complexity (1-5)", min_value=1, max_value=5, value=3)
        avg_base_prep_time_min = row2_col3.number_input("Base Prep (Min)", min_value=1.0, max_value=120.0, value=20.0)
        
        row3_col1, row3_col2 = st.columns(2)
        kitchen_capacity = row3_col1.number_input("Kitchen Capacity", min_value=1, max_value=100, value=10)
        current_active_orders = row3_col2.number_input("Current Active Orders", min_value=0, max_value=200, value=12)
        
        # ── Rider Info ───────────────────────────────────────────
        row4_col1, row4_col2 = st.columns(2)
        rider_distance_to_rest_km = row4_col1.number_input("Rider Distance (km)", min_value=0.1, max_value=50.0, value=3.5)
        rider_avg_speed_kmph = row4_col2.number_input("Rider Speed (km/h)", min_value=1.0, max_value=120.0, value=30.0)
        
        # ── Signal Enrichment ────────────────────────────────────
        st.markdown('<div class="section-header">📡 Signal Enrichment</div>', unsafe_allow_html=True)
        row5_col1, row5_col2 = st.columns(2)
        
        total_pos_kitchen_load = row5_col1.slider("POS Active Tickets (Dine-in + Apps)", min_value=0, max_value=50, value=10)
        merchant_bias_score = row5_col2.selectbox(
            "Historical Geo-FOR Bias Score",
            options=["Low (Trustworthy)", "Medium (Standard)", "High (Marks Early)"],
            index=1
        )
        used_iot_button = st.checkbox("🛎️ Merchant used 'ZeroTap' IoT Button", value=False)
        
        # ── Chaos Factors ────────────────────────────────────────
        st.markdown('<div class="section-header">🌧️ Real-World Chaos Factors</div>', unsafe_allow_html=True)
        live_weather_condition = st.selectbox(
            "Live Local Weather", 
            ["Clear", "Light Rain", "Heavy Rain / Waterlogging"],
            help="Heavy rain slows down kitchen operations and rider transit."
        )
        
        st.markdown("---")
        submit_button = st.form_submit_button(label="⚡ Calculate AI Dispatch Time", use_container_width=True)

with col2:
    st.subheader("🤖 AI Decision Engine")
    
    # Show lottie or placeholder
    if not submit_button:
        try:
            from streamlit_lottie import st_lottie
            if lottie_animation is not None:
                st_lottie(lottie_animation, height=200, key="scooter_anim")
            else:
                st.info("🏍️ AI Dispatch Engine Ready — configure order details and click Calculate.")
        except ImportError:
            st.info("🏍️ AI Dispatch Engine Ready — configure order details and click Calculate.")
    
    # ── Run Prediction Locally ───────────────────────────────────
    if submit_button:
        with st.spinner("Running AI Prediction..."):
            try:
                # ── 1. Prepare features for the ML model ─────────
                model_features = {
                    "items_count": items_count,
                    "order_complexity_score": order_complexity_score,
                    "peak_hour_flag": 1,
                    "order_hour": 19,
                    "day_of_week": 5,
                    "is_weekend": 1,
                    "avg_base_prep_time_min": avg_base_prep_time_min,
                    "kitchen_capacity": kitchen_capacity,
                    "rush_multiplier": 1.5,
                    "is_cloud_kitchen": 0,
                    "rating": 4.2,
                    "reliability_score": 0.80,
                    "historical_accuracy_score": 0.85,
                    "cancellation_bias": 1.0,
                    "cuisine_type": cuisine_type,
                    "city": city,
                }
                
                df = pd.DataFrame([model_features])
                df['cuisine_encoded'] = cuisine_encoder.transform(df['cuisine_type'])
                df['city_encoded'] = city_encoder.transform(df['city'])
                df = df.drop(columns=['cuisine_type', 'city'])
                
                # ── 2. AI Base Prediction ────────────────────────
                base_kpt = float(model.predict(df)[0])
                
                # ── 3. Business Rules Engine ─────────────────────
                rules_context = {
                    "reliability_score": 0.80,
                    "current_active_orders": current_active_orders,
                    "kitchen_capacity": kitchen_capacity,
                    "total_pos_kitchen_load": total_pos_kitchen_load,
                    "merchant_bias_score": merchant_bias_score,
                    "used_iot_button": used_iot_button,
                    "live_weather_condition": live_weather_condition,
                }
                adjusted_kpt, applied_rules = apply_business_rules(base_kpt, rules_context)
                
                # ── 4. JIT Dispatch Calculation ──────────────────
                rider_travel_time = (rider_distance_to_rest_km / rider_avg_speed_kmph) * 60
                dispatch_delay = max(0.0, adjusted_kpt - rider_travel_time)
                action = "HOLD" if dispatch_delay > 0 else "DISPATCH_NOW"
                
                # ── 5. Display Results ───────────────────────────
                
                # Action Banner
                if action == "HOLD":
                    st.markdown(f'''
                    <div class="hold-banner">
                        <div style="font-size:40px">🛑</div>
                        <div class="action-title">HOLD DISPATCH</div>
                        <div class="action-detail">Wait <strong>{dispatch_delay:.1f}</strong> minutes to dispatch</div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="dispatch-banner">
                        <div style="font-size:40px">⚡</div>
                        <div class="action-title">DISPATCH NOW</div>
                        <div class="action-detail">Wait <strong>0</strong> minutes to dispatch</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Metric Cards
                m1, m2 = st.columns(2)
                m1.markdown(f'''
                <div class="metric-box">
                    <div class="value" style="color:#818cf8">{base_kpt:.1f}</div>
                    <div class="label">Raw AI Prediction (min)</div>
                </div>
                ''', unsafe_allow_html=True)
                m2.markdown(f'''
                <div class="metric-box">
                    <div class="value" style="color:#f59e0b">{adjusted_kpt:.1f}</div>
                    <div class="label">Adjusted KPT (min)</div>
                </div>
                ''', unsafe_allow_html=True)
                
                m3, m4 = st.columns(2)
                m3.markdown(f'''
                <div class="metric-box">
                    <div class="value" style="color:#06b6d4">{rider_travel_time:.1f}</div>
                    <div class="label">Rider Travel Time (min)</div>
                </div>
                ''', unsafe_allow_html=True)
                m4.markdown(f'''
                <div class="metric-box">
                    <div class="value" style="color:#10b981">{dispatch_delay:.1f}</div>
                    <div class="label">Dispatch Delay (min)</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Business Rules Applied
                if applied_rules:
                    rules_html = " ".join([f'<span class="rule-tag">{r}</span>' for r in applied_rules])
                    st.markdown(f'''
                    <div style="margin-top:12px; padding:16px; background:rgba(31,41,55,0.5); border-radius:12px; border:1px solid rgba(75,85,99,0.3);">
                        <div style="font-size:11px; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Business Rules Applied</div>
                        {rules_html}
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown('''
                    <div style="margin-top:12px; padding:16px; background:rgba(31,41,55,0.5); border-radius:12px; border:1px solid rgba(75,85,99,0.3);">
                        <div style="font-size:11px; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Business Rules Applied</div>
                        <span class="rule-tag" style="color:#6b7280; background:rgba(107,114,128,0.1); border-color:rgba(107,114,128,0.2);">No adjustments needed</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ using XGBoost, Streamlit & Python  •  ZeroWait Dispatch v1.0")