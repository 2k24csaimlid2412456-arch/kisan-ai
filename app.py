"""
Kisan AI - Gradio Frontend
Upload a leaf photo → get instant disease diagnosis + treatment advice.

Deploy to HuggingFace Spaces:
    1. Create a new Space (Gradio SDK)
    2. Push this file + requirements.txt
    3. Set API_URL as a Space secret
"""

import os
import json
import requests
import gradio as gr
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

SEVERITY_EMOJI = {
    "none":     "🟢",
    "mild":     "🟡",
    "moderate": "🟠",
    "severe":   "🔴",
    "unknown":  "⚪",
}

SEVERITY_LABEL = {
    "none":     "Healthy — No disease detected",
    "mild":     "Mild — Early stage, manageable",
    "moderate": "Moderate — Needs prompt attention",
    "severe":   "Severe — Urgent action required",
    "unknown":  "Unknown — Please re-upload a clearer image",
}


# ── Prediction function ───────────────────────────────────────────────────────
def predict_disease(image):
    if image is None:
        return (
            "⚠️ No image uploaded",
            "",
            "",
            "Please upload a photo of a plant leaf.",
            "",
        )

    try:
        with open(image, "rb") as f:
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": ("leaf.jpg", f, "image/jpeg")},
                timeout=30,
            )

        if response.status_code != 200:
            return (
                "❌ Server Error",
                "",
                "",
                f"API returned status {response.status_code}. Please try again.",
                "",
            )

        data = response.json()

    except requests.exceptions.ConnectionError:
        return (
            "❌ Cannot connect to API",
            "",
            "",
            f"Make sure the backend is running at {API_URL}",
            "",
        )
    except Exception as e:
        return ("❌ Error", "", "", str(e), "")

    # Format outputs
    disease_name = data["display_name"]
    confidence   = f"{data['confidence']}%"
    severity_key = data.get("severity", "unknown")
    severity_txt = (
        f"{SEVERITY_EMOJI.get(severity_key, '⚪')} "
        f"{SEVERITY_LABEL.get(severity_key, severity_key.title())}"
    )
    treatment = data.get("treatment", "No treatment data available.")
    prevention = data.get("prevention", "")

    if data.get("uncertain"):
        disease_name = "⚠️ Uncertain — Image unclear"

    treatment_full = f"**Treatment:**\n{treatment}"
    if prevention:
        treatment_full += f"\n\n**Prevention:**\n{prevention}"

    return disease_name, confidence, severity_txt, treatment_full, ""


# ── Top-3 function ────────────────────────────────────────────────────────────
def predict_top3(image):
    if image is None:
        return "Upload an image first."
    try:
        with open(image, "rb") as f:
            response = requests.post(
                f"{API_URL}/predict/top3",
                files={"file": ("leaf.jpg", f, "image/jpeg")},
                timeout=30,
            )
        data = response.json()
        lines = []
        for i, p in enumerate(data.get("top3", []), 1):
            emoji = SEVERITY_EMOJI.get(p["severity"], "⚪")
            lines.append(
                f"{i}. {p['display_name']}  —  {p['confidence']}%  {emoji}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
.gradio-container { max-width: 860px; margin: auto; }
.result-box textarea { font-size: 15px !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Kisan AI — Crop Disease Detector",
    theme=gr.themes.Soft(primary_hue="green"),
    css=css,
) as demo:

    gr.Markdown("""
# 🌿 Kisan AI — Crop Disease Detector
**Upload a clear, well-lit photo of a single plant leaf** to instantly identify
diseases and get actionable treatment advice.

Trained on 54,000+ images · 38 disease classes · EfficientNetB0 · ~95% accuracy
""")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                type="filepath",
                label="Upload leaf photo",
                height=300,
            )
            submit_btn = gr.Button(
                "🔍  Detect Disease",
                variant="primary",
                size="lg",
            )
            gr.Markdown(
                "_For best results: photograph a single leaf, "
                "with good lighting, against a plain background._"
            )

        with gr.Column(scale=1):
            disease_out   = gr.Textbox(label="Detected disease",     elem_classes="result-box")
            confidence_out = gr.Textbox(label="Confidence score",    elem_classes="result-box")
            severity_out  = gr.Textbox(label="Severity level",       elem_classes="result-box")
            treatment_out = gr.Textbox(
                label="Treatment & Prevention",
                lines=7,
                elem_classes="result-box",
            )

    with gr.Accordion("🔢 See top-3 predictions (for borderline cases)", open=False):
        top3_btn = gr.Button("Get top-3 predictions")
        top3_out = gr.Textbox(label="Top 3 results", lines=4)
        top3_btn.click(predict_top3, inputs=img_input, outputs=top3_out)

    gr.Examples(
        label="Try with example images",
        examples=[
            ["examples/tomato_blight.jpg"],
            ["examples/potato_lateblight.jpg"],
            ["examples/corn_rust.jpg"],
            ["examples/healthy_leaf.jpg"],
        ],
        inputs=img_input,
        outputs=[disease_out, confidence_out, severity_out, treatment_out],
        fn=predict_disease,
        cache_examples=False,
    )

    submit_btn.click(
        predict_disease,
        inputs=img_input,
        outputs=[disease_out, confidence_out, severity_out, treatment_out, gr.Textbox(visible=False)],
    )

    gr.Markdown("""
---
**Crops supported:** Apple · Blueberry · Cherry · Corn · Grape · Orange · Peach · Bell Pepper · Potato · Raspberry · Soybean · Squash · Strawberry · Tomato

**Note:** This tool is for educational and advisory purposes. Always consult a local agricultural expert for serious crop disease management.

🔗 [GitHub](https://github.com/YOUR_USERNAME/kisan-ai) · Built with PyTorch + FastAPI + Gradio
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
