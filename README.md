# Project Repository Overview

This workspace contains multiple repositories focused on species identification, machine learning model development, and data aggregation. Below is a summary of each repository and its main features, along with additional context from the workspace structure and README files.

---

## 1. `docs`
**Description:** Contains documentation and guides for the workspace and its projects.  
**Main File:** `README.md` with workspace and project documentation.

---

## 2. `classifier-app`
**Description:** A web application (TypeScript) that provides an interface for users to interact with machine learning models for species classification.  
**Features:**  

- Static assets with mobile support through react native (we have tested iOS only)
- Connects to backend ML models  
- Responsive design with light/dark mode support

---

## 3. `CNN-test`
**Description:** Python project for training and evaluating Convolutional Neural Networks (CNNs) for species identification from melting curve data.

**Directory Structure & Workflow:**

- `cnn_model.py`: Defines CNN architecture and configuration.
- `cnn.py`: Main script for training/evaluating the CNN.
- `identify.py`, `batch_identify.py`: Tools for species identification and batch evaluation.
- `species-aggregator/`: Data preparation utilities and datasets.
- `model/`: Saved PyTorch model and label classes.
- `requirements.txt`: Python dependencies.

**Workflow:**

1. **Data Preparation:** Place raw CSVs in `species-aggregator/csv/raw/`, use `dataset_creator.py` for splits.
2. **Model Training:** Run `python cnn.py` to train and save the model.
3. **Hyperparameter Tuning:** Run `python hyperparam_tuner.py` for random search.
4. **Evaluation:** Use provided scripts for accuracy summaries and per-sample predictions.

---

## 4. `transfer-learning-vit`

**Description:** Python project focused on transfer learning using Vision Transformers (ViT) and VGG architectures for species identification.  
**Features:**  

- ViT model scripts and hyperparameter tuning  
- Data augmentation for time series  
- Image data and model export utilities  
- Requirements listed in `requirements.txt`

---

## 5. `VITid`

**Description:** Identification server for Vision Transformer (ViT) models.

**Setup:**

- Create a Python virtual environment and install requirements.
- For CUDA support:  
	`pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`
- Download model files from Google Drive and place in `MODEL/`.

**Deploying the Web Server:**
- Use Gunicorn for deployment:  
	`gunicorn --bind 0.0.0.0:8000 --worker-class gevent idServer:app`
- Nginx configuration provided (`nginx.conf`).

---

## 6. `test-validation-server`

**Description:** Python server for validating CNN models and running identification tests.  
**Features:**  

- Model and identification scripts  
- Client for testing server endpoints  
- Stores test data and requirements

---

## 7. `random-forest`

> [!WARNING]
> This was never worked on fully and thus is only kept for archival purposes.

**Description:** Python project implementing a Random Forest classifier for species identification.  
**Features:**  

- Main script for Random Forest classification
- Data aggregation utilities

---

## 8. `species-aggregator`
**Description:** Python scripts for reading raw CSV data files and outputting sorted CSV files for species datasets.  
**Features:**  

- Data aggregation and dataset creation  
- Utilities for combining and preparing species data  
- Data storage folders (`csv/`, `dataset/`)

---

# Additional Information

## Aggregation Utilities
The `species-aggregator/` directory appears in several projects, providing reusable scripts for dataset creation and aggregation. This is done with git submodules and if any issues arise one can run `git submodule init // git submodule update`

## Python Environment

Most machine learning projects use Python and require dependencies listed in their respective `requirements.txt` files. It is recommended to set up a virtual environment for each project and install dependencies before running scripts.

```bash
python -m venv ./.venv
source ./.venv/bin/activate
pip install -r requiremeents.txt
```

---

# Getting Started

1. **Install Dependencies:** For each Python project, run `pip install -r requirements.txt` in the respective directory.
2. **Prepare Data:** follow process listed in ##8.
3. **Run Model(s):** follow processes listed in each repository.

---
