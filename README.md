# VisionGuard - AI vs Real Image Classification

[![CI/CD](https://github.com/turab1020/VisionGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/turab1020/VisionGuard/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.18](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-blue.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.51-purple.svg)](https://dvc.org/)

A complete MLOps pipeline for classifying AI-generated images from real photographs using deep learning with transfer learning, experiment tracking, data versioning, and automated CI/CD deployment.

## Project Overview

VisionGuard addresses the growing challenge of distinguishing AI-generated images from authentic photographs. The project implements a full MLOps lifecycle:

- **Data Pipeline**: Automated dataset acquisition, auditing, EDA, and preprocessing
- **Model Training**: Custom CNN + Transfer Learning (MobileNetV2, EfficientNetB0, ResNet50)
- **Hyperparameter Tuning**: Bayesian optimization via Keras Tuner (10 trials)
- **Experiment Tracking**: MLflow on DagsHub for metric logging and model registry
- **Data Versioning**: DVC with DagsHub remote storage
- **Web Application**: Flask app with Grad-CAM visualizations and prediction history
- **Containerization**: Docker multi-stage build + Docker Compose (web + PostgreSQL)
- **CI/CD**: GitHub Actions (lint → test → Docker build → deploy)

## Results

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| Custom CNN | 0.6871 | 0.5758 | 0.7312 |
| MobileNetV2 | 0.8095 | 0.8169 | 0.9059 |
| EfficientNetB0 | 0.7619 | 0.7500 | 0.8338 |
| ResNet50 | 0.7347 | 0.7368 | 0.8061 |
| **Tuned MobileNetV2** | **0.8435** | **0.8369** | **0.8863** |

Best model: **Tuned MobileNetV2** (Bayesian optimization, 10 trials)

## Project Structure

```
VisionGuard/
├── src/
│   ├── app.py                    # Flask web application
│   ├── dataset_audit.py          # Data quality audit
│   ├── visual_eda.py             # Exploratory data analysis
│   ├── preprocess.py             # Data preprocessing pipeline
│   ├── augmentation.py           # Data augmentation layer
│   ├── models.py                 # CNN & transfer learning models
│   ├── train.py                  # Training pipeline with MLflow
│   ├── hyperparameter_search.py  # Keras Tuner optimization
│   └── gradcam.py                # Grad-CAM interpretability
├── templates/                    # Flask HTML templates
├── tests/
│   └── test_app.py               # 12 pytest test cases
├── models/                       # Saved model weights
├── reports/
│   ├── figures/                  # EDA plots & Grad-CAM heatmaps
│   └── screenshots/              # App & pipeline screenshots
├── data/                         # Dataset (DVC tracked)
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # Web + PostgreSQL services
├── .github/workflows/ci.yml      # CI/CD pipeline
├── requirements.txt              # Python dependencies
└── .dvc/                         # DVC configuration
```

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/turab1020/VisionGuard.git
cd VisionGuard
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the Flask App

```bash
python src/app.py
# Open http://localhost:5001
```

### Run Tests

```bash
pytest tests/test_app.py -v
```

### Docker

```bash
docker-compose up --build
# Open http://localhost:5000
```

## MLflow Experiment Tracking

Experiments are tracked on DagsHub:
- [DagsHub Repository](https://dagshub.com/turab1020/VisionGuard)
- MLflow UI available through DagsHub integration

## Dataset

- **Source**: Kaggle - AI vs Real Image Classification
- **Total Images**: 975 (539 AI-generated, 436 Real)
- **Split**: 70% train / 15% validation / 15% test (stratified)
- **Preprocessing**: Resize to 224x224, normalize [0,1]
- **Augmentation**: Random flip, rotation, zoom, brightness, contrast

## Tech Stack

- **Deep Learning**: TensorFlow 2.18, Keras
- **Web Framework**: Flask 3.0, SQLAlchemy
- **Experiment Tracking**: MLflow 2.16 on DagsHub
- **Data Versioning**: DVC 3.51
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **Testing**: pytest (12 tests)

## Author

**Turab Haider**
- GitHub: [@turab1020](https://github.com/turab1020)
- DagsHub: [@turab1020](https://dagshub.com/turab1020)
