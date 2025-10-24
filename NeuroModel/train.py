import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from config import *
from model import get_model
from preprocessing import get_transforms

# Define the device to use (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = sorted(os.listdir(DATASET_PATH))  # If every subfolder in DATASET_PATH is a class
 


def train():
    transform = get_transforms(224)
    train_dataset = ImageFolder("C:\Users\kshit\Downloads\Dataset\Training", transform=transform)
    val_dataset = ImageFolder("C:\Users\kshit\Downloads\Dataset\Testing", transform=transform)
    
    # Use paths from config.py
    train_dataset = ImageFolder(DATASET_PATH, transform=transform)
    # Renaming val_dataset/val_loader to test_dataset/test_loader for clarity, as it uses TEST_DATASET_PATH
    test_dataset = ImageFolder(TEST_DATASET_PATH, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = get_model(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model = get_model(num_classes=NUM_CLASSES).to(DEVICE) # Use NUM_CLASSES from config.py
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}')
        
        # Add validation logic or save checkpoints here

    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    train()
