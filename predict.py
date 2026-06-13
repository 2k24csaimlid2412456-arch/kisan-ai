"""
Kisan AI - Inference / Prediction Module
Loads the trained EfficientNetB0 model and runs predictions on leaf images.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
MODEL_PATH       = BASE_DIR / "best_model.pth"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
TREATMENTS_PATH  = BASE_DIR.parent / "treatments.json"

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE        = 224
NUM_CLASSES     = 38
CONF_THRESHOLD  = 60.0   # % — below this, return uncertain response
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# ── Transform ─────────────────────────────────────────────────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def load_model(model_path: str = None) -> nn.Module:
    """Load trained model weights. Returns model in eval mode."""
    path = Path(model_path) if model_path else MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. "
            "Run model/train.py first to generate best_model.pth"
        )

    # Load class names to get correct num_classes
    n_classes = NUM_CLASSES
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            class_names_list = json.load(f)
        n_classes = len(class_names_list)

    model = _build_model(num_classes=n_classes)
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"))
    )
    model.eval()
    return model


def _load_class_names() -> list:
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            return json.load(f)
    # Fallback: derive from treatments.json
    with open(TREATMENTS_PATH) as f:
        return list(json.load(f).keys())


def _load_treatments() -> dict:
    with open(TREATMENTS_PATH) as f:
        return json.load(f)


def predict(image_input, model: nn.Module) -> dict:
    """
    Run inference on a leaf image.

    Args:
        image_input: File path (str/Path) or PIL Image object.
        model: Loaded EfficientNetB0 model (from load_model()).

    Returns:
        dict with keys:
            class_key, display_name, confidence,
            severity, treatment, prevention
    """
    # Load image
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError("image_input must be a file path or PIL Image")

    # Run inference
    tensor = INFER_TRANSFORM(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)

    confidence_pct = round(conf.item() * 100, 1)

    # Load class names and treatments
    class_names = _load_class_names()
    treatments  = _load_treatments()
    class_key   = class_names[idx.item()]

    # Confidence guard — return uncertain if below threshold
    if confidence_pct < CONF_THRESHOLD:
        return {
            "class_key":    class_key,
            "display_name": "Uncertain prediction",
            "confidence":   confidence_pct,
            "severity":     "unknown",
            "treatment":    (
                "Image quality may be too low, or this may not be a plant leaf. "
                "Please upload a clear, well-lit, close-up photo of a single leaf."
            ),
            "prevention":   "",
            "uncertain":    True,
        }

    info = treatments.get(class_key, {
        "display_name": class_key.replace("___", " — ").replace("_", " "),
        "severity":     "unknown",
        "treatment":    "No treatment data available for this class.",
        "prevention":   "",
    })

    return {
        "class_key":    class_key,
        "display_name": info.get("display_name", class_key),
        "confidence":   confidence_pct,
        "severity":     info.get("severity", "unknown"),
        "treatment":    info.get("treatment", ""),
        "prevention":   info.get("prevention", ""),
        "uncertain":    False,
    }


def predict_top3(image_input, model: nn.Module) -> list:
    """
    Returns top-3 predictions with confidence scores.
    Useful for borderline cases.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError("image_input must be a file path or PIL Image")

    tensor = INFER_TRANSFORM(img).unsqueeze(0)
    class_names = _load_class_names()
    treatments  = _load_treatments()

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top3_conf, top3_idx = probs.topk(3)
    results = []
    for conf, idx in zip(top3_conf.tolist(), top3_idx.tolist()):
        key  = class_names[idx]
        info = treatments.get(key, {})
        results.append({
            "class_key":    key,
            "display_name": info.get("display_name", key),
            "confidence":   round(conf * 100, 1),
            "severity":     info.get("severity", "unknown"),
        })
    return results


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Loading model...")
    m = load_model()
    print(f"Running prediction on: {image_path}")
    result = predict(image_path, m)

    print("\n── Prediction Result ─────────────────────────")
    print(f"  Disease    : {result['display_name']}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"  Severity   : {result['severity']}")
    print(f"\n  Treatment  : {result['treatment']}")
    print(f"\n  Prevention : {result['prevention']}")

    print("\n── Top 3 Predictions ────────────────────────")
    top3 = predict_top3(image_path, m)
    for i, p in enumerate(top3, 1):
        print(f"  {i}. {p['display_name']} — {p['confidence']}%")
