# Installation Guide

## Prerequisites
- Python 3.11+
- pip

## Steps

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt

3. Prepare the dataset:
- Place MRI images in dataset/images/classname and dataset/test/classname folders ('notumor', 'tumor')
4. Run setup:
  python setup.py

5.Train the model:
python train.py


6. Evaluate model:
python evaluate.py
