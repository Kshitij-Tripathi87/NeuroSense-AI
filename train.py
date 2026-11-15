"""Train the NeuroSense brain-tumour MRI classifier.

Expects one sub-folder per class inside the data directory, e.g.

    dataset/Training/glioma/*.jpg
    dataset/Training/meningioma/*.jpg
    dataset/Training/notumor/*.jpg
    dataset/Training/pituitary/*.jpg

Writes the best model (lowest validation loss) and the class-name mapping to
the model directory.
"""

import argparse
import json
import os

from tensorflow import keras
from tensorflow.keras import layers

MODEL_FILENAME = 'brain_tumor_model.keras'
CLASS_NAMES_FILENAME = 'class_names.json'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', default=os.path.join('dataset', 'Training'),
                        help='folder with one sub-folder per class (default: %(default)s)')
    parser.add_argument('--model-dir', default='models',
                        help='where to write the model and class names (default: %(default)s)')
    parser.add_argument('--img-size', type=int, default=128,
                        help='images are resized to this many pixels square (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50,
                        help='maximum epochs; early stopping usually ends sooner (default: %(default)s)')
    parser.add_argument('--patience', type=int, default=5,
                        help='stop after this many epochs without a better validation loss '
                             '(default: %(default)s)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='fraction of the data held out for validation (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def build_model(img_size, num_classes):
    """Small CNN. Rescaling and augmentation are part of the model, so it takes
    raw 0-255 RGB images; augmentation is only active during training."""
    return keras.Sequential([
        keras.Input(shape=(img_size, img_size, 3)),
        layers.Rescaling(1. / 255),
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.05, fill_mode='constant'),
        layers.RandomZoom(0.1, fill_mode='constant'),
        layers.RandomTranslation(0.05, 0.05, fill_mode='constant'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])


def main():
    args = parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit(
            f"Dataset folder not found: {args.data_dir}\n"
            "Pass its location with --data-dir (see Installation_guide.md)."
        )

    keras.utils.set_random_seed(args.seed)

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=args.val_split,
        subset='both',
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode='categorical',
    )
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    # Saved before training so the mapping exists even if the run is interrupted.
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, MODEL_FILENAME)
    class_names_path = os.path.join(args.model_dir, CLASS_NAMES_FILENAME)
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)

    model = build_model(args.img_size, len(class_names))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        # The file on disk is always the epoch with the lowest validation loss.
        keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience),
    ]

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    val_losses = history.history['val_loss']
    best_epoch = val_losses.index(min(val_losses))
    print(f"Best epoch: {best_epoch + 1} of {len(val_losses)} "
          f"(val_loss {val_losses[best_epoch]:.4f}, "
          f"val_accuracy {history.history['val_accuracy'][best_epoch]:.4f})")
    print(f"Model saved at {model_path}")
    print(f"Class names saved at {class_names_path}")


if __name__ == '__main__':
    main()
