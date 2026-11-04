# VisionGuard — AI vs Real Image Classification

A web service that classifies whether an uploaded image is AI-generated or real, built with TensorFlow/Keras, tracked with MLflow on DagsHub, served via Flask, and containerized with Docker Compose.

## Project Structure

```
VisionGuard/
├── src/                    # Source code
│   ├── dataset_audit.py    # Task 1.2 — dataset audit script
│   ├── visual_eda.py       # Task 1.3 — EDA figures
│   ├── data_pipeline.py    # Task 2.1 — tf.data pipeline
│   ├── baseline_cnn.py     # Task 2.2 — custom CNN
│   ├── transfer_model.py   # Task 2.3 — transfer learning
│   ├── train.py            # Task 2.4 — training with callbacks
│   ├── evaluate.py         # Task 2.5 — evaluation metrics
│   ├── error_analysis.py   # Task 2.6 — misclassified images
│   ├── keras_tuner_exp.py  # Task 2.7 — hyperparameter tuning
│   ├── gradcam.py          # Bonus  — Grad-CAM heat maps
│   ├── mlflow_toy.py       # Task 3.1 — MLflow local toy
│   ├── mlflow_train.py     # Task 3.3 — instrumented training
│   ├── register_model.py   # Task 3.6 — model registry
│   ├── load_champion.py    # Task 3.6 — load model by alias
│   └── inference.py        # Task 4.1 — inference module
├── templates/              # Flask HTML templates
├── static/                 # CSS and static assets
├── tests/                  # Unit tests
├── models/                 # Saved models (DVC-tracked)
├── data/                   # Dataset (DVC-tracked)
├── notebooks/              # Jupyter notebooks (optional)
├── .github/workflows/      # GitHub Actions CI
├── app.py                  # Task 4   — Flask application
├── Dockerfile              # Task 6.2 — container image
├── docker-compose.yml      # Task 6.3 — compose stack
├── requirements.txt        # Python dependencies
├── check_env.py            # Task 1.1 — environment check
├── .env.example            # Template for secrets
├── .gitignore
└── README.md
```

## Quick Start

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd VisionGuard

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify environment
python check_env.py

# 5. Copy env template and fill in your values
cp .env.example .env

# 6. Run with Docker Compose
docker compose up --build
```

## Links

- **GitHub**: [repo link]
- **DagsHub**: [project link]
- **Live API**: [deployment URL]
