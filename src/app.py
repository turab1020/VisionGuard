"""
Task 5 — Flask Web Application
Upload an image → get AI vs Real prediction with confidence + Grad-CAM.
"""
import os
import io
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy

# ── App config ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "visionguard-dev-key-change-in-prod")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Database — SQLite for dev, Postgres for prod
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'visionguard.db'}")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

CLASS_NAMES = ["AI-Generated", "Real"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}


# ── Database model ──────────────────────────────────────────────
class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Prediction {self.filename}: {self.prediction} ({self.confidence:.2f})>"


# ── Load model at startup ──────────────────────────────────────
_model = None


def get_model():
    global _model
    if _model is None:
        model_path = BASE_DIR / "models" / "best_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train.py first."
            )
        _model = keras.models.load_model(str(model_path))
        print(f"  Model loaded: {model_path}")
    return _model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    """Preprocess uploaded image for prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def generate_gradcam(model, img_array):
    """Generate Grad-CAM heatmap for the prediction."""
    try:
        # Find base model layer (for transfer models)
        base_model = None
        last_conv_name = None

        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                base_model = layer
                break

        if base_model is not None:
            for sub in reversed(base_model.layers):
                if 'conv' in sub.name.lower():
                    last_conv_name = sub.name
                    break
            conv_output = base_model.get_layer(last_conv_name).output
        else:
            for layer in reversed(model.layers):
                if isinstance(layer, keras.layers.Conv2D):
                    last_conv_name = layer.name
                    break
            conv_output = model.get_layer(last_conv_name).output

        if last_conv_name is None:
            return None

        grad_model = keras.Model(
            inputs=model.input,
            outputs=[conv_output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, pred = grad_model(img_array)
            loss = pred[:, 0]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Resize to image size and apply colormap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        heatmap_resized = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_resized]
        jet_heatmap = tf.image.resize(
            jet_heatmap[np.newaxis, ...], (224, 224)
        ).numpy()[0]

        original = img_array[0]
        superimposed = jet_heatmap * 0.4 + original * 0.6
        superimposed = np.clip(superimposed * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(superimposed)
    except Exception as e:
        print(f"  Grad-CAM error: {e}")
        return None


# ── Routes ──────────────────────────────────────────────────────
@app.route("/")
def index():
    """Home page with upload form."""
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and run prediction."""
    if "file" not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}", "error")
        return redirect(url_for("index"))

    try:
        # Read and preprocess
        image_bytes = file.read()
        img_array = preprocess_image(image_bytes)

        # Predict
        model = get_model()
        prob = float(model.predict(img_array, verbose=0).flatten()[0])
        pred_class = CLASS_NAMES[1] if prob >= 0.5 else CLASS_NAMES[0]
        confidence = prob if prob >= 0.5 else 1 - prob

        # Save uploaded image
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex[:12]}.{ext}"
        save_path = UPLOAD_FOLDER / unique_name
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        # Generate Grad-CAM
        gradcam_name = None
        gradcam_img = generate_gradcam(model, img_array)
        if gradcam_img is not None:
            gradcam_name = f"gradcam_{unique_name.rsplit('.', 1)[0]}.png"
            gradcam_img.save(UPLOAD_FOLDER / gradcam_name)

        # Save to database
        prediction = Prediction(
            filename=file.filename,
            prediction=pred_class,
            confidence=round(confidence, 4),
            image_path=unique_name,
            timestamp=datetime.utcnow(),
        )
        db.session.add(prediction)
        db.session.commit()

        return render_template(
            "result.html",
            filename=file.filename,
            prediction=pred_class,
            confidence=round(confidence * 100, 1),
            image_url=url_for("static", filename=f"uploads/{unique_name}"),
            gradcam_url=url_for("static", filename=f"uploads/{gradcam_name}") if gradcam_name else None,
            raw_prob=round(prob, 4),
        )

    except Exception as e:
        flash(f"Prediction error: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/history")
def history():
    """Show all past predictions."""
    page = request.args.get("page", 1, type=int)
    per_page = 20
    pagination = Prediction.query.order_by(
        Prediction.timestamp.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    return render_template("history.html", predictions=pagination)


@app.route("/health")
def health():
    """Health check endpoint for Docker/deployment."""
    return {"status": "healthy", "model_loaded": _model is not None}, 200


# ── Create tables ───────────────────────────────────────────────
with app.app_context():
    db.create_all()


if __name__ == "__main__":
    print("VisionGuard — Flask App")
    print("=" * 40)
    # Pre-load model
    with app.app_context():
        try:
            get_model()
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
