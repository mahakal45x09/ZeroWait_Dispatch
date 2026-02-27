#  ZeroWait Dispatch: AI-Powered JIT Delivery Engine
** Live Demo:** [Click here to try the AI Dispatcher](https://zerowaitdispatch-e859rsjzqzbg9ruiggwdx2.streamlit.app/#zero-wait-dispatch-dashboard)



![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)

##  Project Overview
In the highly competitive food delivery industry, dispatching a rider too early results in high wait times at the restaurant, while dispatching too late results in cold food. **ZeroWait Dispatch** solves this by predicting the exact Kitchen Prep Time (KPT) and calculating a Just-In-Time (JIT) dispatch delay. 

This full-stack machine learning project uses an **XGBoost regression model** combined with a **FastAPI backend** and a **Streamlit frontend dashboard** to simulate a live delivery dispatch center, ensuring riders arrive exactly when the food is ready.

##  Core Features
* **Predictive ML Model:** Predicts base prep time using historical restaurant data, cuisine type, city, and order complexity.
* **Business Logic Engine:** Dynamically adjusts predicted times based on live kitchen capacity, weather/rush multipliers, and restaurant reliability scores.
* **JIT Dispatch Algorithm:** Calculates exactly how many minutes to delay a rider dispatch based on their current distance and speed.
* **Interactive Dashboards:** Includes a Streamlit Python UI (`dashboard.py`) for dispatchers to receive instant AI recommendations.

##  Project Structure
```text
ZEROWAIT-DISPATCH/
│
├── app.py                     # FastAPI backend server & API endpoints
├── dashboard.py               # Streamlit interactive frontend UI
├── index.html                 # Alternative lightweight HTML frontend
│
├── predict_kpt.py             # ML prediction logic & data preprocessing
├── Kpt.py                     # Model training and evaluation script
│
├── kpt_xgboost_model.pkl      # Trained XGBoost regression model
├── city_encoder.pkl           # Label encoder for geographic data
├── cuisine_encoder.pkl        # Label encoder for cuisine types
│
├── data/                      # (Included CSV Datasets)
│   ├── orders.csv             # Historical order data
│   ├── restaurants.csv        # Restaurant metrics and capacity
│   ├── riders.csv             # Rider speed and location logs
│   ├── merchant_behavior.csv  # Historical reliability metrics
│   └── dispatch_log.csv       # Output logs of AI dispatch decisions
│
├── requirements.txt           # Python dependencies for cloud deployment
└── .gitignore                 # Git ignore configuration

## Tech Stack
Machine Learning: Python, Pandas, Scikit-Learn, XGBoost, Joblib
Backend API: FastAPI, Uvicorn, Pydantic (RESTful Architecture, CORS Middleware)
Frontend UI: Streamlit, HTML/CSS/JavaScript

## install
pip install -r requirements.txt

##Start the Backend API (Terminal 1)
uvicorn app:app --reload

##Start the Frontend Dashboard (Terminal 2)
streamlit run dashboard.py





