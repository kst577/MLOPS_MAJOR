import os
import numpy as np
import joblib
from flask import Flask, request, render_template_string
from PIL import Image

app = Flask(__name__)

MODEL_PATH = "savedmodel.pth"
model = joblib.load(MODEL_PATH)

PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 500px; margin: 40px auto; padding: 20px; }
        h1 { color: #333; }
        .result { margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>Olivetti Face Predictor</h1>
    <p>Upload a grayscale face image. The model predicts which person (0-39) it belongs to.</p>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
    <div class="result">
        <strong>Predicted person:</strong> {{ prediction }}
    </div>
    {% endif %}
</body>
</html>
"""


def prepare_image(image_file):
    """Convert uploaded image to the same 4096-feature format used during training."""
    img = Image.open(image_file).convert("L")
    img = img.resize((64, 64))
    pixels = np.array(img, dtype=np.float64) / 255.0
    return pixels.flatten().reshape(1, -1)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        uploaded = request.files.get("image")
        if uploaded and uploaded.filename:
            features = prepare_image(uploaded)
            prediction = int(model.predict(features)[0])

    return render_template_string(PAGE, prediction=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
