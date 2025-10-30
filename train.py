import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 4
DATASET_PATH = r'C:\Users\kshitij\Downloads\Dataset\Training'
MODEL_DIR = './model'
MODEL_PATH = os.path.join(MODEL_DIR, 'brain_tumor_model.h5')
os.makedirs(MODEL_DIR, exist_ok=True)

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# CNN Model WITHOUT transfer learning
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training (will take a LONG time; try smaller epochs for testing/debugging)
model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# Save model
model.save(MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")
