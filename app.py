"""REST API that serves a trained NeuroSense model.

    python app.py
    curl -F "file=@scan.jpg" http://127.0.0.1:5000/predict

Endpoints:
    GET  /health   - service status and the classes the model predicts
    POST /predict  - classify one image sent as multipart form field "file"

Research and education only: predictions are not a medical diagnosis.
"""

import argparse
import json
import os

import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow import keras
from werkzeug.exceptions import HTTPException

from train import CLASS_NAMES_FILENAME, MODEL_FILENAME

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def load_model_and_classes(model_dir):
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    class_names_path = os.path.join(model_dir, CLASS_NAMES_FILENAME)
    for path in (model_path, class_names_path):
        if not os.path.isfile(path):
            raise SystemExit(f"Not found: {path}\nRun train.py first, or pass --model-dir.")

    model = keras.models.load_model(model_path)
    with open(class_names_path) as f:
        class_names = json.load(f)
    if model.output_shape[-1] != len(class_names):
        raise SystemExit(
            f"{model_path} has {model.output_shape[-1]} outputs but {class_names_path} "
            f"lists {len(class_names)} classes. Re-run train.py."
        )
    return model, class_names


def create_app(model_dir='models'):
    model, class_names = load_model_and_classes(model_dir)
    img_size = model.input_shape[1:3]

    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

    def error(message, status):
        return jsonify(error=message), status

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if exc.code == 413:
            return error(f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)
        return error(exc.description, exc.code)

    @app.get('/health')
    def health():
        return jsonify(status='ok', classes=class_names, image_size=list(img_size))

    @app.post('/predict')
    def predict():
        upload = request.files.get('file')
        if upload is None or not upload.filename:
            return error('Send the image as multipart form field "file".', 400)
        data = upload.read()
        if not data:
            return error('The uploaded file is empty.', 400)

        # Same decoding and resizing as training (image_dataset_from_directory).
        # Uploads are handled in memory and never written to disk.
        try:
            image = tf.io.decode_image(data, channels=3, expand_animations=False)
        except tf.errors.InvalidArgumentError:
            return error('Could not read the file as a JPEG, PNG, BMP or GIF image.', 400)
        image = tf.image.resize(image, img_size)

        probabilities = model(image[tf.newaxis], training=False).numpy()[0]
        best = int(probabilities.argmax())
        return jsonify(
            prediction=class_names[best],
            confidence=float(probabilities[best]),
            probabilities={name: float(p) for name, p in zip(class_names, probabilities)},
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model-dir', default='models',
                        help='folder written by train.py (default: %(default)s)')
    parser.add_argument('--host', default='127.0.0.1',
                        help='use 0.0.0.0 to accept connections from other machines '
                             '(default: %(default)s)')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    create_app(args.model_dir).run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
