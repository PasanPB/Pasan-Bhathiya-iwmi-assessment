import sys
import os

# 🔹 Fix import path (VERY IMPORTANT)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
sys.path.append(PROJECT_ROOT)

import streamlit as st
import torch
import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt

from src.model import ModelDevelopment

# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelDevelopment().get_model().to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model, device


st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=DM+Sans:wght@400;500;700&display=swap');

    :root {
        --bg-1: #f6fbfa;
        --bg-2: #e5f5f1;
        --ink: #14281d;
        --muted: #4c6358;
        --accent: #0f8b72;
        --accent-2: #f4a259;
        --ok: #2f9e44;
        --warn: #d9480f;
        --card: rgba(255, 255, 255, 0.82);
        --border: rgba(15, 139, 114, 0.18);
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at -10% -15%, #d4efe7 0%, transparent 60%),
            radial-gradient(1000px 500px at 120% -10%, #ffe6cc 0%, transparent 60%),
            linear-gradient(180deg, var(--bg-1), var(--bg-2));
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.02em;
    }

    p, div, label, span {
        font-family: 'DM Sans', sans-serif !important;
    }

    .hero {
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        background: var(--card);
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 26px rgba(19, 42, 33, 0.08);
        animation: fadeSlide 0.55s ease-out;
    }

    .hero p {
        color: var(--muted);
        margin-top: 0.45rem;
        margin-bottom: 0;
        font-size: 0.98rem;
    }

    .result-chip {
        margin-top: 0.65rem;
        display: inline-block;
        border-radius: 999px;
        padding: 0.3rem 0.9rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .chip-ok {
        background: rgba(47, 158, 68, 0.16);
        color: var(--ok);
    }

    .chip-warn {
        background: rgba(217, 72, 15, 0.14);
        color: var(--warn);
    }

    .panel {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.95rem;
        background: var(--card);
        box-shadow: 0 6px 18px rgba(19, 42, 33, 0.05);
        animation: fadeSlide 0.65s ease-out;
    }

    @keyframes fadeSlide {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff, #eef8f5);
        border-left: 1px solid rgba(15, 139, 114, 0.14);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    model, device = load_model()
except FileNotFoundError:
    st.title("Face Mask Detection App")
    st.error("Model file is missing in deployment.")
    st.info("Expected file path: models/best_model.pth")
    st.info("Add that file to your GitHub repository and redeploy the app.")
    st.stop()

classes = ["with_mask", "without_mask"]

# ------------------------
# Preprocess Image
# ------------------------
def preprocess_image(image):
    image = ImageOps.grayscale(image).convert("RGB") if image.mode != "RGB" else image
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    image = np.expand_dims(image, axis=0)

    tensor = torch.tensor(image, dtype=torch.float32).to(device)
    return tensor

# ------------------------
# Prediction Function
# ------------------------
def predict(image):
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    return probs


def draw_probability_chart(probs):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    colors = ["#0f8b72", "#f4a259"]

    bars = ax.bar(classes, probs, color=colors, width=0.56)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Class Confidence")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    for idx, bar in enumerate(bars):
        value = probs[idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1b4332",
            fontweight="bold",
        )

    return fig

st.markdown(
    """
    <div class="hero">
        <h1>Face Mask Detection</h1>
        <p>
            Upload a face image and get a fast classification with confidence scores for
            both classes.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
uploaded_file = st.file_uploader("Drop an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is None:
    st.info("Add a JPG or PNG image to start the prediction.")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        probs = predict(image)

        pred_class = classes[np.argmax(probs)]
        confidence = np.max(probs)

        label = "Mask detected" if pred_class == "with_mask" else "No mask detected"
        chip_class = "chip-ok" if pred_class == "with_mask" else "chip-warn"

        col_img, col_result = st.columns([1.08, 1], gap="large")

        with col_img:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_result:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Prediction")
            st.markdown(
                f'<span class="result-chip {chip_class}">{label}</span>',
                unsafe_allow_html=True,
            )
            st.metric("Model confidence", f"{confidence * 100:.2f}%")
            st.progress(float(confidence))
            st.caption(
                "Confidence indicates how strongly the model favors the selected class."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.subheader("Class Probabilities")
        st.caption("Distribution across with_mask and without_mask classes")
        st.pyplot(draw_probability_chart(probs), use_container_width=True)

    except Exception as e:
        st.error(f"Error processing image: {e}")

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("Model Snapshot")

st.sidebar.markdown("""
### Architecture
- 3 Convolutional Layers  
- Batch Normalization  
- Max Pooling  
- Dropout (0.5)  
- Fully Connected Layers  

### Training
- Optimizer: Adam  
- Learning Rate Scheduler: StepLR  

### Performance
- Validation Accuracy: **(update this with your result)**  

### Notes
- Custom CNN built from scratch  
- No pretrained models used  
""")