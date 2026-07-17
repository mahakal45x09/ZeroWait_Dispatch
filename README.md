# ⚡ ZeroWait Dispatch: AI-Powered JIT Delivery Engine

**🚀 Live Demo:** [**Click here to try the AI Dispatcher**](https://zerowaitdispatch.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)
![Tests](https://img.shields.io/badge/Tests-20%20Passing-brightgreen.svg)
![CI](https://github.com/mahakal45x09/ZeroWait_Dispatch/actions/workflows/ci.yml/badge.svg)

---

## 🎯 Project Overview

In the food delivery industry, dispatching a rider **too early** means they wait idle at the restaurant (wasted time & money), while dispatching **too late** means cold food and angry customers.

**ZeroWait Dispatch** solves this by predicting the exact **Kitchen Prep Time (KPT)** using an XGBoost ML model, applying 6 configurable business rules, and calculating a **Just-In-Time (JIT) dispatch delay** — ensuring riders arrive exactly when the food is ready.

```
Order Placed → XGBoost Predicts KPT → Business Rules Adjust → JIT Delay Calculated → HOLD or DISPATCH
```

---

## ✨ Core Features

| Feature | Description |
|---------|-------------|
| 🧠 **ML Prediction** | XGBoost regressor trained on 10,000 orders predicts base kitchen prep time |
| ⚙️ **Business Rules Engine** | 6 configurable rules (reliability, capacity, POS load, merchant bias, IoT, weather) |
| ⏱️ **JIT Dispatch** | Calculates optimal delay: `dispatch_delay = adjusted_KPT - rider_travel_time` |
| 📊 **Interactive Dashboard** | Streamlit UI with live prediction, metric cards, and action banners |
| 🌐 **REST API** | FastAPI backend with health checks, model info, and Swagger docs |
| 🧪 **Test Suite** | 20 unit tests covering all business rules and edge cases |
| 🐳 **Docker Ready** | Dockerfile + docker-compose for one-command deployment |
| 🔄 **CI/CD** | GitHub Actions pipeline runs tests on every push |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Input (Order)                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            XGBoost ML Model (16 features)                │
│         Predicts base_kpt from historical data           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Business Rules Engine (6 rules)                │
│                                                          │
│  A. Reliability Buffer .......... +4.0m if score < 0.75  │
│  B. Kitchen Capacity Surge ...... +20% if overwhelmed    │
│  C. POS Dine-in Load ............ +0.5m per extra ticket │
│  D. Merchant Geo-FOR Bias ....... +5m early / -2m trust  │
│  E. ZeroTap IoT Button .......... -3.0m when triggered   │
│  F. Weather Buffer .............. +3m rain / +8m storm   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              JIT Dispatch Calculation                     │
│     delay = adjusted_KPT − rider_travel_time             │
│     Action: HOLD (wait X min) or DISPATCH_NOW            │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ZeroWait_Dispatch/
│
├── app.py                     # FastAPI backend (API + health checks)
├── dashboard.py               # Streamlit interactive frontend (standalone)
├── index.html                 # Premium HTML frontend (dark glassmorphism UI)
│
├── business_rules.py          # Configurable business rules engine (6 rules)
├── config.py                  # Centralized env-based configuration
├── predict_kpt.py             # CLI prediction demo script
├── Kpt.py                     # Model training & evaluation script
│
├── kpt_xgboost_model.pkl      # Trained XGBoost regression model
├── city_encoder.pkl           # Label encoder for city data
├── cuisine_encoder.pkl        # Label encoder for cuisine types
│
├── orders.csv                 # 10,000 historical orders
├── restaurants.csv            # 300 restaurant profiles
├── riders.csv                 # 500 rider profiles
├── merchant_behavior.csv      # Merchant reliability metrics
├── dispatch_log.csv           # AI dispatch decision logs
├── Desktop.py                 # Synthetic data generator
├── Graph.py                   # Comparison visualization script
│
├── tests/
│   └── test_business_rules.py # 20 unit tests (all passing ✅)
│
├── .github/workflows/
│   └── ci.yml                 # GitHub Actions CI pipeline
│
├── Dockerfile                 # Container for FastAPI backend
├── docker-compose.yml         # Multi-service orchestration
├── requirements.txt           # Python dependencies
└── .gitignore                 # Git ignore rules
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Machine Learning** | Python, Pandas, Scikit-Learn, XGBoost, Joblib |
| **Backend API** | FastAPI, Uvicorn, Pydantic (with Field validators) |
| **Frontend** | Streamlit (primary), HTML/CSS/JS (premium alternative) |
| **Testing** | Pytest (20 tests) |
| **DevOps** | Docker, Docker Compose, GitHub Actions CI |
| **Deployment** | Streamlit Cloud |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/mahakal45x09/ZeroWait_Dispatch.git
cd ZeroWait_Dispatch
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard
```bash
streamlit run dashboard.py
```
The dashboard runs predictions **locally** — no backend needed!

### 3. (Optional) Run the FastAPI Backend
```bash
uvicorn app:app --reload
```
Then open `index.html` in your browser for the HTML frontend, or visit `http://127.0.0.1:8000/docs` for Swagger API docs.

### 4. (Optional) Run with Docker
```bash
docker-compose up --build
```
- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

### 5. Run Tests
```bash
python -m pytest tests/ -v
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict_dispatch` | Submit order → receive AI dispatch recommendation |
| `GET` | `/health` | Health check for monitoring |
| `GET` | `/model/info` | Model metadata and status |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## 🧪 Testing

20 unit tests covering all business rules:

```
tests/test_business_rules.py::TestReliabilityBufferRule       ✅ 3 tests
tests/test_business_rules.py::TestKitchenCapacitySurgeRule    ✅ 3 tests
tests/test_business_rules.py::TestPOSKitchenLoadRule          ✅ 2 tests
tests/test_business_rules.py::TestMerchantBiasRule            ✅ 3 tests
tests/test_business_rules.py::TestIoTButtonRule               ✅ 2 tests
tests/test_business_rules.py::TestWeatherBufferRule           ✅ 3 tests
tests/test_business_rules.py::TestApplyBusinessRules          ✅ 4 tests
──────────────────────────────────────────────────────────────
20 passed in 0.12s
```

---

## 📄 License

This project is open source and available for educational and portfolio purposes.
