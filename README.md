# Face Mask Classification Project

End-to-end computer vision project for classifying faces as:

- with_mask
- without_mask

The repository includes data preprocessing, custom CNN model development, training and evaluation, and a Streamlit web application for interactive inference.

## Highlights

- Custom CNN architecture built from scratch (no pretrained backbone)
- Structured into dedicated classes for preprocessing, model development, training, and inference
- Data augmentation and normalization pipeline
- Training with Adam optimizer and StepLR scheduler
- Saved best checkpoint and evaluation artifacts
- Streamlit UI with image upload, prediction confidence, and top-k chart

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── models/
│   └── best_model.pth
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── metrics.json
├── src/
│   ├── inference.py
│   ├── model.py
│   ├── preprocessing.py
│   └── train.py
├── test/
│   ├── __init__.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── train_runner.py
├── requirements.txt
└── README.md
```

## Class Design

### 1) Preprocessing
- File: `src/preprocessing.py`
- Class: `BasicPreprocessing`
- Key methods:
  - `import_dataset`
  - `split_data`
  - `get_transforms`
  - `create_dataloaders`

### 2) Model Development
- File: `src/model.py`
- Class: `ModelDevelopment`
- CNN model: `MaskCNN` (compatibility alias `MaskClassifier` is also available)

### 3) Training
- File: `src/train.py`
- Class: `Trainer`
- Key methods:
  - `train_one_epoch`
  - `validate`
  - `train`
  - `plot_results`
  - `evaluate_and_save_confusion_matrix`

### 4) Inferencing
- File: `src/inference.py`
- Class: `BasicInference`
- Uses Haar cascade (`haarcascade_frontalface_default.xml`) to detect faces in static images.

## Installation

```powershell
pip install -r requirements.txt
```

Main dependencies:
- numpy
- pandas
- matplotlib
- opencv-python-headless
- torch
- torchvision
- scikit-learn
- streamlit
- Pillow

## Environment Setup

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
```

Linux or macOS:

```bash
source .venv/bin/activate
export PYTHONPATH=.
```

## Dataset Format

Place images in:

```text
dataset/
├── with_mask/
└── without_mask/
```

Supported file types: `.jpg`, `.jpeg`, `.png`.

## Run Commands

### 1) Test preprocessing

```powershell
$env:PYTHONPATH = "."
python test/test_preprocessing.py
```

### 2) Test model forward pass

```powershell
$env:PYTHONPATH = "."
python test/test_model.py
```

### 3) Train model

```powershell
$env:PYTHONPATH = "."
python train_runner.py
```

### 4) Run Streamlit app

```powershell
$env:PYTHONPATH = "."
streamlit run app/streamlit_app.py
```

## Training and Evaluation Outputs

After training, these artifacts are produced:

- `models/best_model.pth`
- `results/training_curves.png`
- `results/confusion_matrix.png`
- `results/metrics.json`

`results/metrics.json` stores persisted accuracy metrics used by Streamlit sidebar in deployment.

## Streamlit App Features

- Upload `.jpg`, `.jpeg`, `.png`
- Displays uploaded image and predicted class
- Shows confidence percentage and progress bar
- Top-k probability bar chart (`top_k=3` request with safe fallback for 2-class model)
- Sidebar with architecture summary and achieved accuracy

Deployment note:
- In cloud deployments where `dataset/` is unavailable, sidebar accuracy is read from `results/metrics.json`.
- If no metrics file exists, accuracy displays as `N/A`.

## Assessment Coverage Summary

- Task 1 (Preprocessing pipeline): Completed
- Task 2 (Custom architecture and training): Completed
- Task 3 (Evaluation and inferencing): Completed
- Task 4 (Streamlit app): Completed
- Version control and documentation requirements: Completed

## Model Behavior Summary

Model performs better on:
- front-facing faces
- well-lit images
- standard mask usage

Model is weaker on:
- low-light or shadowed images
- partial occlusions
- improper/non-standard mask wearing
- extreme head angles

## Known Notes

- `test/test_model.py` imports `MaskClassifier`; this is supported through compatibility aliasing in `src/model.py`.
- `split_data()` supports both `split_data(image_paths, labels)` and `split_data(df)` usage patterns.

## Future Improvements

- Add uncertainty thresholding in Streamlit for low-confidence predictions
- Expand dataset diversity for difficult real-world conditions
- Add CI pipeline for automated testing and linting