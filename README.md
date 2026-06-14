# 🌿 Kisan AI — Crop Disease Detection

> Upload a leaf photo → get instant disease diagnosis + treatment advice

[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-HuggingFace_Spaces-yellow)](https://huggingface.co/spaces/Sudhanshu011/kisan-ai)
[![API](https://img.shields.io/badge/API-Railway-purple)](https://your-app.railway.app/docs)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)](https://pytorch.org)

---

## The Problem

Indian farmers lose **30–40% of crop yield every year** to plant diseases, with no quick
way to diagnose what's wrong. A farmer in rural UP or Bihar has no easy access to
agricultural experts, and by the time they reach one, the disease has spread.

**Kisan AI** solves this by letting anyone with a phone take a photo of a leaf and
get an instant diagnosis — along with plain-language treatment advice.

---

## Demo

> *(Add a demo GIF here — record with LICEcap or OBS)*

| Input | Output |
|-------|--------|
| Leaf photo | Disease name + confidence |
| | Severity level (Healthy / Mild / Moderate / Severe) |
| | Treatment steps |
| | Prevention advice |

---

## Model Performance

| Metric | Score |
|--------|-------|
| **Test accuracy** | ~95% |
| **Architecture** | EfficientNetB0 (pretrained on ImageNet) |
| **Dataset** | PlantVillage — 54,306 images, 38 classes |
| **Training** | Progressive fine-tuning (head → full) |
| **Inference time** | ~120ms per image |

### Hardest classes to distinguish
| Class | Notes |
|-------|-------|
| Tomato Early Blight vs Target Spot | Similar concentric ring patterns |
| Corn Gray Leaf Spot vs Northern Leaf Blight | Both produce elongated lesions |
| Grape Black Rot vs Esca | Both cause dark necrotic spots |

---

## System Architecture

```
User (mobile/desktop)
        │
        ▼
Gradio UI (HuggingFace Spaces)
        │  POST /predict — multipart image upload
        ▼
FastAPI Backend (Railway)
        │  loads image → runs EfficientNetB0 inference
        ▼
EfficientNetB0 model (best_model.pth, ~20MB)
        │  returns class + confidence
        ▼
treatments.json lookup
        │  returns display name + severity + treatment
        ▼
JSON response → rendered in Gradio UI
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | EfficientNetB0, PyTorch, torchvision |
| Backend | FastAPI, Uvicorn |
| Frontend | Gradio 4 |
| Deployment — API | Railway (Docker) |
| Deployment — UI | HuggingFace Spaces |
| Dataset | PlantVillage (Kaggle) |

---

## Crops and Diseases Covered (38 classes)

Apple (4) · Blueberry (1) · Cherry (2) · Corn/Maize (4) · Grape (4) ·
Orange (1) · Peach (2) · Bell Pepper (2) · Potato (3) · Raspberry (1) ·
Soybean (1) · Squash (1) · Strawberry (2) · **Tomato (10)**

---

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/kisan-ai.git
cd kisan-ai
```

### 2. Set up Python environment
```bash
conda create -n kisanai python=3.10
conda activate kisanai
pip install -r api/requirements.txt
```

### 3. Download dataset and train model
```bash
# Set up Kaggle API first (kaggle.json in ~/.kaggle/)
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/

# Train (use Google Colab for GPU — see notebooks/colab_training.py)
python model/train.py --data_dir data/plantvillage/color --epochs 15
```

### 4. Start the API
```bash
uvicorn api.main:app --reload
# Open http://localhost:8000/docs to test
```

### 5. Start the Gradio UI
```bash
cd ui
python app.py
# Open http://localhost:7860
```

---

## Deployment

### API → Railway
```bash
railway login
railway init
railway up
```
Set env var: `MODEL_URL` = direct download URL to your `best_model.pth`

### UI → HuggingFace Spaces
1. Create new Space (Gradio SDK)
2. Push `ui/app.py` and `ui/requirements.txt`
3. Add `API_URL` as a Space secret

---

## Key Technical Decisions

**Why EfficientNetB0?**
EfficientNet scales depth, width, and resolution together via a compound coefficient.
B0 achieves better accuracy-per-parameter than ResNet-50, with faster inference —
critical for a mobile-first use case.

**Why progressive fine-tuning?**
Training only the classifier head first (5 epochs, LR=1e-3) prevents the pretrained
feature extractor from being destroyed by large gradients early on. Then unfreezing
all layers (10 epochs, LR=1e-4) with a lower learning rate extracts maximum accuracy.

**Why the 60% confidence threshold?**
In testing, predictions below 60% were almost always either non-plant images or
very blurry/poorly lit photos. Returning a specific wrong disease name in these cases
would be worse than saying "image unclear" — especially for farmers making treatment decisions.

---

## Future Improvements

- [ ] Hindi language interface for rural farmers
- [ ] Offline-capable mobile app (React Native + ONNX model)
- [ ] GPS-based regional disease heatmap
- [ ] Severity tracking over time (photograph the same plant weekly)
- [ ] WhatsApp bot integration (most common smartphone app in rural India)
- [ ] Voice output for farmers with low literacy

---

## Disclaimer

This tool is for **educational and advisory purposes only**. For serious crop disease
management decisions, always consult a qualified agricultural expert or your local
Krishi Vigyan Kendra (KVK).

---

Built by **[Your Name]** · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)
