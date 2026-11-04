#!/bin/bash
# ─────────────────────────────────────────────────────
# DagsHub + DVC Remote Setup Script
# Run AFTER creating your DagsHub repo
# Usage: bash setup_dagshub.sh <dagshub_username> <dagshub_repo_name>
# ─────────────────────────────────────────────────────
set -e

if [ $# -lt 2 ]; then
    echo "Usage: bash setup_dagshub.sh <dagshub_username> <dagshub_repo_name>"
    echo "Example: bash setup_dagshub.sh turabhaider VisionGuard"
    exit 1
fi

USERNAME="$1"
REPO="$2"

echo "Setting up DagsHub DVC remote..."
echo "  Username: $USERNAME"
echo "  Repo:     $REPO"

# Add DagsHub as DVC remote storage
dvc remote add -f dagshub "https://dagshub.com/${USERNAME}/${REPO}.dvc"
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user "$USERNAME"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Now enter your DagsHub ACCESS TOKEN when prompted"
echo "  (Get it from: https://dagshub.com/user/settings/tokens)"
echo "═══════════════════════════════════════════════════"
echo ""
read -sp "DagsHub Access Token: " TOKEN
echo ""

dvc remote modify dagshub --local password "$TOKEN"

# Also set up .env for MLflow tracking
cat > .env << ENVEOF
# DagsHub MLflow Tracking
MLFLOW_TRACKING_URI=https://dagshub.com/${USERNAME}/${REPO}.mlflow
MLFLOW_TRACKING_USERNAME=${USERNAME}
MLFLOW_TRACKING_PASSWORD=${TOKEN}

# Flask
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Database
DATABASE_URL=sqlite:///visionguard.db
ENVEOF

echo ""
echo "✅ DVC remote 'dagshub' configured"
echo "✅ .env file created with MLflow credentials"
echo ""
echo "Now push your data with:"
echo "  dvc push -r dagshub"
