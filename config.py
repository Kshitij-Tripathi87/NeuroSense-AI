import os

# ====================================
# NeuroScan AI - Configuration File
# ====================================

# Model Configuration
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Classes - Brain Tumor Types
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'brain_tumor_model.h5')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'Training')
TEST_DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'Testing')

# Training Configuration
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# Image dimensions will be determined automatically from dataset
# Target size for resizing will be set in preprocessing

# Server Configuration
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

# File Upload Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
