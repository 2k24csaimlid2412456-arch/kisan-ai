"""
Kisan AI - FastAPI Backend
Serves the crop disease prediction model as a REST API.

Endpoints:
    GET  /           - Health check
    POST /predict    - Upload leaf image, get disease prediction
    GET  /classes    - List all 38 disease classes
    GET  /docs       - Swagger UI (automatic)
"""

import os
import sys
import time
import uuid
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kisan-ai")

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Kisan AI — Crop Disease Detection API",
    description=(
        "Upload a leaf photo to detect crop diseases. "
        "Trained on 54,000+ images across 38 disease classes using EfficientNetB0."
    ),
    version="1.0.0",
    contact={"name": "Kisan AI", "url": "https://github.com/YOUR_USERNAME/kisan-ai"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL = None

def get_model():
    """Lazy-load model on first request."""
    global MODEL
    if MODEL is None:
        logger.info("Loading model...")
        from model.predict import load_model
        MODEL = load_model()
        logger.info("Model loaded successfully.")
    return MODEL


# Optional: Download model from URL if not present (for Railway deployment)
MODEL_URL = os.getenv("MODEL_URL", "")

def download_model_if_needed():
    model_path = Path("model/best_model.pth")
    if not model_path.exists() and MODEL_URL:
        import requests
        logger.info(f"Downloading model from {MODEL_URL}...")
        r = requests.get(MODEL_URL, timeout=120)
        r.raise_for_status()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(r.content)
        logger.info("Model downloaded successfully.")

@app.on_event("startup")
async def startup_event():
    download_model_if_needed()
    logger.info("Kisan AI API started.")


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
TMP_DIR       = Path("/tmp/kisan-ai-uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ── Response models ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    class_key:    str
    display_name: str
    confidence:   float
    severity:     str
    treatment:    str
    prevention:   str
    uncertain:    bool
    inference_ms: float


class HealthResponse(BaseModel):
    status:  str
    version: str
    model_loaded: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status":       "Kisan AI is running",
        "version":      "1.0.0",
        "model_loaded": MODEL is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_disease(file: UploadFile = File(...)):
    """
    Upload a leaf image to detect crop disease.

    - Accepts JPEG, PNG, WEBP images up to 10MB
    - Returns disease name, confidence %, severity, treatment, and prevention advice
    - Confidence below 60% returns an 'uncertain' response
    """
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. "
                   f"Accepted: {', '.join(ALLOWED_TYPES)}"
        )

    # Read content
    contents = await file.read()

    # Validate file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents)//1024}KB). Maximum allowed: 10MB"
        )

    # Save to temp file
    tmp_filename = TMP_DIR / f"{uuid.uuid4().hex}.jpg"
    with open(tmp_filename, "wb") as f:
        f.write(contents)

    try:
        t_start = time.time()
        model  = get_model()

        from model.predict import predict
        result = predict(str(tmp_filename), model)

        inference_ms = round((time.time() - t_start) * 1000, 1)

        logger.info(
            f"Prediction: {result['display_name']} "
            f"({result['confidence']}%) in {inference_ms}ms"
        )

        return {**result, "inference_ms": inference_ms}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        # Clean up temp file
        if tmp_filename.exists():
            tmp_filename.unlink()


@app.post("/predict/top3", tags=["Prediction"])
async def predict_top3(file: UploadFile = File(...)):
    """
    Returns top-3 disease predictions with confidence scores.
    Useful for borderline cases where the model is less certain.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    tmp_filename = TMP_DIR / f"{uuid.uuid4().hex}.jpg"
    with open(tmp_filename, "wb") as f:
        f.write(contents)

    try:
        model = get_model()
        from model.predict import predict_top3
        results = predict_top3(str(tmp_filename), model)
        return {"top3": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_filename.exists():
            tmp_filename.unlink()


@app.get("/classes", tags=["Info"])
def get_classes():
    """List all 38 disease classes the model can detect."""
    import json
    treatments_path = Path(__file__).parent.parent / "treatments.json"
    with open(treatments_path) as f:
        treatments = json.load(f)

    classes = [
        {
            "class_key":    k,
            "display_name": v["display_name"],
            "severity":     v["severity"],
        }
        for k, v in treatments.items()
    ]
    return {"total": len(classes), "classes": classes}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
