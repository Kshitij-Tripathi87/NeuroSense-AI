# Installation Guide

## Prerequisites

- Python 3.11, 3.12 or 3.13
- pip

## Steps

1. Clone the repository and open a terminal in its folder.

2. Install dependencies (a virtual environment is recommended):

   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset. Put the images in one sub-folder per class:

   ```
   dataset/
     Training/
       glioma/  meningioma/  notumor/  pituitary/
     Testing/
       glioma/  meningioma/  notumor/  pituitary/
   ```

   The data can live anywhere; pass its location with `--data-dir` in the next steps.
   Supported image formats are JPEG, PNG, BMP and GIF.

4. Train the model:

   ```
   python train.py
   ```

   or, with the data somewhere else:

   ```
   python train.py --data-dir "C:\path\to\Dataset\Training"
   ```

   20% of the training images are held out for validation. Training stops early once the
   validation loss stops improving. Two files are written to `models/`:

   - `brain_tumor_model.keras` – the model from the best epoch
   - `class_names.json` – the class names in the order of the model's outputs

5. Evaluate the model on the test images:

   ```
   python evaluate.py
   ```

   or:

   ```
   python evaluate.py --data-dir "C:\path\to\Dataset\Testing"
   ```

   This prints accuracy, per-class precision, recall and F1 score, and the confusion matrix.

6. Start the API (optional):

   ```
   python app.py
   ```

   Then send it an image from another terminal:

   ```
   curl -F "file=@scan.jpg" http://127.0.0.1:5000/predict
   ```

   (In Windows PowerShell, type `curl.exe` instead of `curl`.)

   The reply is JSON with the predicted class, its confidence and the probability of every
   class. `GET /health` returns the service status and class names.

   By default the API only accepts connections from the same machine. Use `--host 0.0.0.0` to
   open it to your network, and `--port` to change the port. It runs on Flask's built-in
   server, which is meant for development; put it behind a production WSGI server such as
   Waitress or Gunicorn before exposing it more widely.

## Options

Run any script with `--help` for its full list. The most useful:

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | `dataset/Training` (train), `dataset/Testing` (evaluate) | Folder with one sub-folder per class |
| `--model-dir` | `models` | Where the model and class names are written and read (all three scripts) |
| `--epochs` | `50` | Maximum number of epochs |
| `--patience` | `5` | Epochs without improvement before training stops |
| `--img-size` | `128` | Images are resized to this size |
| `--batch-size` | `32` | Batch size |
| `--seed` | `42` | Seed for the train/validation split and weight initialisation |
