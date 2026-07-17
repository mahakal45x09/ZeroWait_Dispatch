"""
Centralized configuration for ZeroWait Dispatch.

Reads from environment variables with sensible defaults for local development.
"""

import os
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("ZEROWAIT_MODEL_PATH", str(BASE_DIR / "kpt_xgboost_model.pkl"))
CUISINE_ENCODER_PATH = os.getenv("ZEROWAIT_CUISINE_ENCODER", str(BASE_DIR / "cuisine_encoder.pkl"))
CITY_ENCODER_PATH = os.getenv("ZEROWAIT_CITY_ENCODER", str(BASE_DIR / "city_encoder.pkl"))

# ── API Settings ────────────────────────────────────────────────────
API_HOST = os.getenv("ZEROWAIT_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("ZEROWAIT_API_PORT", "8000"))
API_URL = os.getenv("ZEROWAIT_API_URL", f"http://127.0.0.1:{API_PORT}")

# CORS — restrict in production, allow all in development
CORS_ORIGINS = os.getenv("ZEROWAIT_CORS_ORIGINS", "*").split(",")

# ── Logging ─────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("ZEROWAIT_LOG_LEVEL", "INFO").upper()

# ── Model Info (for /model/info endpoint) ───────────────────────────
MODEL_VERSION = os.getenv("ZEROWAIT_MODEL_VERSION", "1.0.0")
MODEL_DESCRIPTION = "XGBoost Regressor for Kitchen Prep Time prediction"
TRAINING_DATASET_SIZE = 10000
FEATURE_COUNT = 16
