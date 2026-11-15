# NeuroSense-AI

NeuroSense AI is a deep learning project that classifies brain MRI images into four categories:

- **Glioma** – tumours arising from glial cells
- **Meningioma** – tumours of the meninges
- **Pituitary** – pituitary gland tumours
- **No tumour** – healthy brain tissue

> **Research and education only.** This is not a medical device. It has not been clinically
> validated and must not be used to diagnose or treat anyone.

## What's in the repo

| File | Purpose |
|---|---|
| `train.py` | Trains a small convolutional neural network (CNN) and saves the best model |
| `evaluate.py` | Scores a trained model on a held-out test set |
| `app.py` | REST API that classifies uploaded images with the trained model |
| `Installation_guide.md` | Setup and usage steps |

## Model

- Custom CNN: three convolution + max-pooling blocks, a 128-unit dense layer, softmax output
- Images are resized to 128×128 RGB; pixel rescaling is built into the model, so it takes raw 0–255 images
- Data augmentation during training: horizontal flip, rotation, zoom, shift
- Dropout for regularisation
- Early stopping on validation loss; the saved model is the epoch with the lowest validation loss
- The class-name order is saved next to the model as `class_names.json`

## Evaluation

`evaluate.py` reports, on a test set the model has not seen:

- Overall accuracy
- Precision, recall and F1 score per class
- Confusion matrix

No benchmark results are published here yet.

## REST API

`python app.py` serves the trained model on `http://127.0.0.1:5000`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status and the classes the model predicts |
| `POST /predict` | Classify one image sent as multipart form field `file` |

```
curl -F "file=@scan.jpg" http://127.0.0.1:5000/predict
```

```json
{
  "prediction": "glioma",
  "confidence": 0.91,
  "probabilities": {"glioma": 0.91, "meningioma": 0.05, "notumor": 0.01, "pituitary": 0.03}
}
```

The numbers above only show the response format. Errors come back as `{"error": "..."}` with a
4xx status. Uploads are limited to 10 MB and are never written to disk.

## Dataset

The dataset is not included. Any image set works as long as it has one sub-folder per class:

```
dataset/
  Training/
    glioma/  meningioma/  notumor/  pituitary/
  Testing/
    glioma/  meningioma/  notumor/  pituitary/
```

This is the layout of the "Brain Tumor MRI Dataset" by Masoud Nickparvar on Kaggle.

## Quick start

```
pip install -r requirements.txt
python train.py
python evaluate.py
python app.py
```

See [Installation_guide.md](Installation_guide.md) for details and options.

## License

MIT – see [LICENSE](LICENSE).
