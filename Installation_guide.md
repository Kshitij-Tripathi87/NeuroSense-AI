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

## Options

Run `python train.py --help` or `python evaluate.py --help` for the full list. The most useful:

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | `dataset/Training` (train), `dataset/Testing` (evaluate) | Folder with one sub-folder per class |
| `--model-dir` | `models` | Where the model and class names are written and read |
| `--epochs` | `50` | Maximum number of epochs |
| `--patience` | `5` | Epochs without improvement before training stops |
| `--img-size` | `128` | Images are resized to this size |
| `--batch-size` | `32` | Batch size |
| `--seed` | `42` | Seed for the train/validation split and weight initialisation |
