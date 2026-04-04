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
from src.preprocessing import BasicPreprocessing

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


@st.cache_resource
def get_achieved_test_accuracy(_model, _device):
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    if not os.path.isdir(dataset_dir):
        return None

    prep = BasicPreprocessing(dataset_dir)
    image_paths, labels = prep.import_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(image_paths, labels)
    _, _, test_loader = prep.create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    _model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(_device)
            labels_batch = labels_batch.to(_device)
            outputs = _model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)

    if total == 0:
        return None

    return (correct / total) * 100


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
    st.error("❌ Model file missing or failed to load.")
    st.info("Make sure 'models/best_model.pth' exists.")
    st.stop()

achieved_accuracy = None
accuracy_note = ""
try:
    achieved_accuracy = get_achieved_test_accuracy(model, device)
except Exception:
    achieved_accuracy = None
    accuracy_note = "Test dataset not available in deployment, so sidebar accuracy is hidden."

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
def draw_probability_chart(probs, class_names, top_k=3):
    top_k = min(top_k, len(class_names))
    top_indices = np.argsort(probs)[::-1][:top_k]

    top_labels = [class_names[i].replace("_", " ").title() for i in top_indices]
    top_probs = probs[top_indices]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x_positions = np.arange(len(top_labels))
    bars = ax.bar(x_positions, top_probs, width=0.58, color=["#0f8b72", "#f4a259", "#457b9d"][:len(top_labels)])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(top_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(f"Top {top_k} Prediction Confidence")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, prob in zip(bars, top_probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            prob + 0.02,
            f"{prob * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    return fig

# ------------------------
# UI
# ------------------------
st.title("😷 Face Mask Detection")
st.write("Upload an image to check if a person is wearing a mask.")

# 🔥 Instruction (your requirement)
st.markdown("### 📤 Upload Image")
st.caption("Please upload a clear, zoomed image of a masked or unmasked person for accurate detection.")

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
        st.subheader("Top Predictions")
        st.pyplot(draw_probability_chart(probs, classes, top_k=3), use_container_width=True)
        if len(classes) < 3:
            st.caption("Top-3 view requested; only 2 classes are available in this model.")

    except Exception as e:
        st.error(f"Error processing image: {e}")

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("📊 Model Info")

st.sidebar.markdown(f"""
### 🧠 Architecture
- 3 Convolution Layers  
- Batch Normalization  
- Max Pooling  
- Dropout (0.5)  
- Fully Connected Layers  

### ⚙️ Training
- Optimizer: Adam  
- Learning Rate Scheduler: StepLR  

### 📈 Performance
- Test Accuracy: **{f"{achieved_accuracy:.2f}%" if achieved_accuracy is not None else "N/A"}**  

### 📌 Notes
- Custom CNN built from scratch  
- No pretrained models used  
""")

if accuracy_note:
    st.sidebar.caption(accuracy_note)