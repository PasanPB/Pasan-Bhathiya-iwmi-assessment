# Face Mask Classification Project

This project implements a complete end-to-end deep learning pipeline to classify images into:

- with_mask
- without_mask

It includes data preprocessing, custom CNN model development, training with validation and checkpointing, evaluation helpers, and a Streamlit web app for interactive prediction.

## Key Features

- Custom CNN built from scratch using PyTorch (no pretrained backbone)
- Class-based architecture for preprocessing, model development, and training
- Train/validation/test split with image augmentation
- Best model checkpoint saving during training
- Training curve plot generation
- Streamlit app for image upload and prediction
- Visualization of top-k class confidence (top-3 requested, auto-fallback for 2-class model)

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

## Class-Based Design (Assessment Requirement)

The project follows the required class-based structure with separate classes for core sections.

1. Preprocessing section
- Class: BasicPreprocessing
- File: src/preprocessing.py
- Responsibilities:
	- import_dataset
	- split_data
	- get_transforms
	- create_dataloaders

2. Model section
- Class: ModelDevelopment
- File: src/model.py
- Responsibilities:
	- build_model
	- get_model

3. Training section
- Class: Trainer
- File: src/train.py
- Responsibilities:
	- train_one_epoch
	- validate
	- train
	- plot_results

Additional class:
- BasicInference in src/inference.py for image-level face detection and inference utilities.

## Requirements

Install dependencies:

```powershell
pip install -r requirements.txt
```

requirements.txt includes:
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

## Dataset Setup

Place dataset images in:

```text
dataset/
├── with_mask/
└── without_mask/
```

Supported image formats: jpg, jpeg, png

## Run Instructions

### 1) Run preprocessing test

```powershell
$env:PYTHONPATH = "."
python test/test_preprocessing.py
```

### 2) Run model test

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

## Training Details

- Input size: 128 x 128
- Model: 3 convolution blocks + batch normalization + max pooling + dropout + FC layers
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Scheduler: StepLR (step_size=5, gamma=0.5)
- Checkpoint: best validation accuracy model saved to models/best_model.pth

## Streamlit App Details

The web app provides:

- Image upload interface
- Predicted class and confidence score
- Confidence progress bar
- Top predictions bar chart

Note on top-3 chart requirement:
- Current model has 2 classes, so the app renders top-k with safe fallback: k = min(3, number_of_classes).
- This means it still behaves correctly and displays all available classes.

## Outputs

After training, these artifacts are generated or updated:

- models/best_model.pth
- results/training_curves.png

Inference utilities can also save:

- results/confusion_matrix.png

## Common Issues and Fixes

1. Import errors for local modules
- Ensure PYTHONPATH is set to project root before running scripts.

2. Streamlit model load error
- Confirm models/best_model.pth exists.

3. OpenCV GUI errors in cloud/deployment
- Use opencv-python-headless (already in requirements).

## Model Behavior Notes

Model tends to perform better on:
- clear and front-facing faces
- proper lighting
- standard mask usage

Model may struggle with:
- low light or shadows
- partial face occlusion
- incorrect mask wearing styles
- non-standard masks (scarf, transparent mask, etc.)
- extreme side profile angles

## Future Improvements

- Add threshold-based uncertain label in app for low-confidence predictions
- Expand dataset for difficult real-world cases
- Add per-class metrics and confusion matrix to training report
- Add automated test pipeline in CI