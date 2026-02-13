from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from src.verify import evaluate

from src.verify import evaluate

app = Flask(__name__)
CORS(app)

@app.route("/api/evaluate", methods=["POST"])
def evaluate_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    prompt = request.form.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    image_file = request.files["image"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image_path = tmp.name
        image_file.save(image_path)

    try:
        result = evaluate(image_path, prompt)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
