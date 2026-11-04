"""
Task 5.4 — Flask Application Tests
Tests upload, prediction, history, error handling, and database writes.
"""
import os
import sys
import io
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def app():
    """Create test app with in-memory SQLite."""
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    # Import after setting env
    from app import app as flask_app, db
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False

    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def sample_image():
    """Create a small valid JPEG image in memory."""
    from PIL import Image
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


class TestIndexPage:
    def test_index_loads(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"VisionGuard" in resp.data

    def test_index_has_upload_form(self, client):
        resp = client.get("/")
        assert b"upload" in resp.data.lower()
        assert b'enctype="multipart/form-data"' in resp.data


class TestPrediction:
    @patch("app.get_model")
    def test_predict_returns_result(self, mock_get_model, client, sample_image):
        """Test that uploading an image returns a prediction page."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.85]])
        mock_model.layers = []  # Simplify Grad-CAM (will skip)
        mock_get_model.return_value = mock_model

        resp = client.post(
            "/predict",
            data={"file": (sample_image, "test.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert b"Real" in resp.data or b"AI-Generated" in resp.data

    @patch("app.get_model")
    def test_predict_ai_generated(self, mock_get_model, client, sample_image):
        """Test AI-Generated prediction (prob < 0.5)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.2]])
        mock_model.layers = []
        mock_get_model.return_value = mock_model

        resp = client.post(
            "/predict",
            data={"file": (sample_image, "fake.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert b"AI-Generated" in resp.data

    @patch("app.get_model")
    def test_predict_real(self, mock_get_model, client, sample_image):
        """Test Real prediction (prob >= 0.5)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.9]])
        mock_model.layers = []
        mock_get_model.return_value = mock_model

        resp = client.post(
            "/predict",
            data={"file": (sample_image, "real.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert b"Real" in resp.data

    def test_predict_no_file(self, client):
        """Test prediction without file redirects."""
        resp = client.post("/predict", content_type="multipart/form-data")
        assert resp.status_code == 302  # redirect

    def test_predict_empty_filename(self, client):
        """Test empty filename redirects."""
        resp = client.post(
            "/predict",
            data={"file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302

    def test_predict_invalid_extension(self, client):
        """Test invalid file extension is rejected."""
        resp = client.post(
            "/predict",
            data={"file": (io.BytesIO(b"not an image"), "test.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302  # redirect with flash


class TestHistory:
    def test_history_empty(self, client):
        """Test history page with no predictions."""
        resp = client.get("/history")
        assert resp.status_code == 200
        assert b"No predictions yet" in resp.data

    @patch("app.get_model")
    def test_history_after_prediction(self, mock_get_model, client, sample_image):
        """Test that predictions appear in history."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.7]])
        mock_model.layers = []
        mock_get_model.return_value = mock_model

        # Make a prediction
        client.post(
            "/predict",
            data={"file": (sample_image, "test_hist.jpg")},
            content_type="multipart/form-data",
        )

        # Check history
        resp = client.get("/history")
        assert resp.status_code == 200
        assert b"test_hist.jpg" in resp.data


class TestDatabase:
    @patch("app.get_model")
    def test_prediction_saved_to_db(self, mock_get_model, app, client, sample_image):
        """Test that predictions are saved to the database."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.75]])
        mock_model.layers = []
        mock_get_model.return_value = mock_model

        client.post(
            "/predict",
            data={"file": (sample_image, "db_test.jpg")},
            content_type="multipart/form-data",
        )

        from app import Prediction
        with app.app_context():
            count = Prediction.query.count()
            assert count == 1
            pred = Prediction.query.first()
            assert pred.filename == "db_test.jpg"
            assert pred.prediction == "Real"
            assert pred.confidence > 0


class TestHealth:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
