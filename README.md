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

## Assessment Task Compliance

### Task 1 -> Data Preprocessing and Pipeline

#### 1) Load and Split
- [x] Dataset loading implemented using class-based preprocessing workflow.
- [x] Required libraries imported:
	- os
	- cv2
	- numpy
	- pandas
	- matplotlib
- [x] Additional libraries used where needed (for splitting and loaders):
	- sklearn.model_selection.train_test_split
	- torch.utils.data.DataLoader, Dataset
	- torchvision.transforms
	- PIL.Image
- [x] Training, validation, and test splits are created.
- [x] DataLoaders are created for train/validation/test datasets.

#### 2) Transformations
- [x] Resizing applied (128 x 128).
- [x] Normalization applied.
- [x] Data augmentation applied for training:
	- random horizontal flip
	- random rotation
- [x] Separate train and evaluation transforms are used.

#### 3) Required Structure
- [x] `BasicPreprocessing` class implemented.
- [x] `__init__` implemented.
- [x] `import_dataset` implemented.
- [x] Additional preprocessing methods implemented (split_data, get_transforms, create_dataloaders, any_name).

### Task 2 -> Custom Computer Vision Architecture and Training

#### 1) Design from Scratch
- [x] Custom CNN model built from scratch in `src/model.py`.
- [x] No pretrained backbones used (no ResNet/VGG/YOLO pretrained models).

#### 2) Required Layers
- [x] Convolutional blocks included.
- [x] Max pooling included.
- [x] Batch normalization included.
- [x] Dropout included.
- [x] Fully connected layers included.

#### 3) Training and Scheduling
- [x] Optimizer implemented (Adam).
- [x] Dynamic learning-rate scheduler implemented (StepLR).
- [x] Per-epoch training/validation loss and accuracy logging implemented.

#### 4) Deliverables
- [x] Model definition file available: `src/model.py`.
- [x] Training history plot generated: `results/training_curves.png`.

### Task 3 -> Model Evaluation and Basic Inferencing

#### 1) Metrics and Detection
- [x] Static-image face detection implemented in `src/inference.py`.
- [x] Haarcascade frontal-face detector used (`haarcascade_frontalface_default.xml`).
- [x] Classification report generation implemented.
- [x] Confusion matrix generation implemented and saved to `results/confusion_matrix.png`.

#### 2) Analysis
- [x] Brief model success/failure analysis documented in this README (Model Behavior Notes).

#### 3) Required Structure
- [x] `BasicInference` class implemented with `__init__`, `detect_images`, and helper methods.
- [x] `main()` implemented with required assessment print statement.

### Task 4 -> Streamlit Application

#### 1) Interface
- [x] Upload support for `.jpg`, `.png`, `.jpeg`.

#### 2) Inference Display
- [x] Uploaded image displayed.
- [x] Predicted class displayed.
- [x] Confidence percentage displayed.

#### 3) Visualization
- [x] Top-k prediction bar chart implemented with `top_k=3` request.
- [x] Safe fallback to available class count (`min(3, number_of_classes)`) for the current 2-class model.

#### 4) Info Sidebar
- [x] Architecture summary included.
- [x] Achieved accuracy shown as computed test accuracy from the saved model and current dataset split.

#### 5) Deliverable Status
- [x] Functional `app/streamlit_app.py` included.
- [ ] Deployed URL or screen recording must be attached at submission time.

### Version Control and Documentation (Mandatory)

- [x] Git repository used with multiple descriptive commits.
- [x] Comprehensive README included for reproducibility and maintenance.
- [x] Python/model checkpoint-aware `.gitignore` included.
- [x] Commit history available for reviewer inspection.

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
	- evaluate_and_save_confusion_matrix

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

This test imports `MaskClassifier` from `src/model.py` (kept for compatibility).

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
- results/confusion_matrix.png

`results/confusion_matrix.png` is now generated automatically at the end of training from test-set predictions.

## Common Issues and Fixes

1. Import errors for local modules
- Ensure PYTHONPATH is set to project root before running scripts.

2. Preprocessing split mismatch
- `import_dataset()` returns `image_paths, labels` and `split_data()` supports both:
	- `split_data(image_paths, labels)`
	- `split_data(df)`

3. Streamlit model load error
- Confirm models/best_model.pth exists.

4. OpenCV GUI errors in cloud/deployment
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