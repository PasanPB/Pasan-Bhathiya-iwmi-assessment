import streamlit as st
import torch
import numpy as np
import cv2
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
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

classes = ["with_mask", "without_mask"]

# ------------------------
# Preprocess Image
# ------------------------
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
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

# ------------------------
# UI Layout
# ------------------------
st.title("😷 Face Mask Detection App")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    probs = predict(image)

    pred_class = classes[np.argmax(probs)]
    confidence = np.max(probs) * 100

    st.subheader(f"Prediction: {pred_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # ------------------------
    # Bar Chart (Top 3)
    # ------------------------
    st.subheader("Prediction Probabilities")

    fig, ax = plt.subplots()
    ax.bar(classes, probs)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")

    st.pyplot(fig)

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("📊 Model Info")

st.sidebar.markdown("""
**Architecture:**
- 3 Convolution Layers  
- Batch Normalization  
- Max Pooling  
- Dropout  
- Fully Connected Layers  

**Training:**
- Optimizer: Adam  
- Scheduler: StepLR  

**Accuracy:**
- ~90% (example, update after training)
""")