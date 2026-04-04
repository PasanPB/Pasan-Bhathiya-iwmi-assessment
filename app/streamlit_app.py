import sys
import os

# 🔹 Fix import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
sys.path.append(PROJECT_ROOT)

import streamlit as st
import torch
import numpy as np
from PIL import Image
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


# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="Face Mask Detection", layout="centered")

# ------------------------
# Load Model Safely
# ------------------------
try:
    model, device = load_model()
except Exception:
    st.title("Face Mask Detection App")
    st.error("Model file missing or failed to load.")
    st.info("Make sure models/best_model.pth exists.")
    st.stop()

classes = ["with_mask", "without_mask"]

# ------------------------
# Preprocess Image
# ------------------------
def preprocess_image(image):
    image = image.convert("RGB")  # ensure 3 channels
    image = image.resize((128, 128))

    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return torch.tensor(image, dtype=torch.float32).to(device)

# ------------------------
# Prediction
# ------------------------
def predict(image):
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    return probs

# ------------------------
# Chart
# ------------------------
def draw_probability_chart(probs):
    fig, ax = plt.subplots()

    ax.bar(classes, probs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")

    for i, v in enumerate(probs):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')

    return fig

# ------------------------
# UI
# ------------------------
st.title("😷 Face Mask Detection")
st.write("Upload an image to check if a person is wearing a mask.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is None:
    st.info("Please upload an image to begin.")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        probs = predict(image)

        pred_class = classes[np.argmax(probs)]
        confidence = float(np.max(probs))

        label = "Mask Detected" if pred_class == "with_mask" else "No Mask Detected"

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Prediction")
            st.success(label)

            st.metric("Confidence", f"{confidence * 100:.2f}%")

            st.progress(min(max(confidence, 0.0), 1.0))

        st.write("")
        st.subheader("Class Probabilities")
        st.pyplot(draw_probability_chart(probs))

    except Exception as e:
        st.error(f"Error processing image: {e}")

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("Model Info")

st.sidebar.markdown("""
### Architecture
- 3 Convolution Layers  
- Batch Normalization  
- Max Pooling  
- Dropout  
- Fully Connected Layers  

### Training
- Optimizer: Adam  
- Scheduler: StepLR  

### Performance
- Validation Accuracy: **(update this)**  

### Notes
- Custom CNN  
- No pretrained models  
""")