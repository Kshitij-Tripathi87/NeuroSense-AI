"""Evaluate a trained NeuroSense model on a held-out test set.

Expects the same folder layout as training (one sub-folder per class), e.g.
dataset/Testing/glioma/*.jpg. Use images the model never saw during training.
"""

import argparse
import json
import os

import numpy as np
from tensorflow import keras

from train import CLASS_NAMES_FILENAME, MODEL_FILENAME


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', default=os.path.join('dataset', 'Testing'),
                        help='folder with one sub-folder per class (default: %(default)s)')
    parser.add_argument('--model-dir', default='models',
                        help='folder written by train.py (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


def confusion_matrix(y_true, y_pred, num_classes):
    """Rows are the true class, columns the predicted class."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def per_class_metrics(matrix):
    """Precision, recall and F1 per class; 0 where a class has no samples."""
    true_pos = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0)
    support = matrix.sum(axis=1)
    precision = np.divide(true_pos, predicted, out=np.zeros_like(true_pos), where=predicted > 0)
    recall = np.divide(true_pos, support, out=np.zeros_like(true_pos), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(true_pos), where=denom > 0)
    return precision, recall, f1, support


def main():
    args = parse_args()
    model_path = os.path.join(args.model_dir, MODEL_FILENAME)
    class_names_path = os.path.join(args.model_dir, CLASS_NAMES_FILENAME)

    for path in (model_path, class_names_path):
        if not os.path.isfile(path):
            raise SystemExit(f"Not found: {path}\nRun train.py first, or pass --model-dir.")
    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"Test folder not found: {args.data_dir}\nPass its location with --data-dir.")

    with open(class_names_path) as f:
        class_names = json.load(f)

    found = sorted(d for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d)))
    if found != sorted(class_names):
        raise SystemExit(
            f"Class folders in {args.data_dir} do not match the trained model.\n"
            f"  model was trained on: {sorted(class_names)}\n"
            f"  test folder contains: {found}"
        )

    model = keras.models.load_model(model_path)
    img_size = model.input_shape[1:3]

    # class_names fixes the label order to the one used in training.
    test_ds = keras.utils.image_dataset_from_directory(
        args.data_dir,
        class_names=class_names,
        image_size=img_size,
        batch_size=args.batch_size,
        label_mode='int',
        shuffle=False,
    )

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    y_pred = model.predict(test_ds).argmax(axis=1)

    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    precision, recall, f1, support = per_class_metrics(matrix)
    accuracy = np.trace(matrix) / matrix.sum()

    width = max(len(name) for name in class_names + ['macro average'])
    print(f"\nTest images: {matrix.sum()}")
    print(f"Accuracy:    {accuracy:.4f}\n")
    print(f"{'':<{width}}  precision  recall  f1-score  support")
    for i, name in enumerate(class_names):
        print(f"{name:<{width}}  {precision[i]:9.4f}  {recall[i]:6.4f}  {f1[i]:8.4f}  {support[i]:7d}")
    print(f"{'macro average':<{width}}  {precision.mean():9.4f}  {recall.mean():6.4f}  "
          f"{f1.mean():8.4f}  {support.sum():7d}")

    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    col = max(width, 6)
    print(f"{'':<{width}}  " + '  '.join(f"{name:>{col}}" for name in class_names))
    for i, name in enumerate(class_names):
        print(f"{name:<{width}}  " + '  '.join(f"{count:>{col}d}" for count in matrix[i]))


if __name__ == '__main__':
    main()
