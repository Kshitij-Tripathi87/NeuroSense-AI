import torch
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall
from preprocessing import load_data
from config import TESTSET_PATH, MODEL_PATH, CLASSES

# Load test data
X_test, y_test = load_data(TESTSET_PATH)
X_test_tensor = torch.from_numpy(X_test).float()
class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
y_test_indices = torch.tensor([class_to_idx[label] for label in y_test])

# Load PyTorch model
model = torch.load(MODEL_PATH)
model.eval()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)
X_test_tensor = X_test_tensor.to(DEVICE)
y_test_indices = y_test_indices.to(DEVICE)

# Run prediction
with torch.no_grad():
    outputs = model(X_test_tensor)
    if outputs.shape[1] == 1:
        predictions = (outputs > 0.5).long().squeeze()
    else:
        predictions = torch.argmax(outputs, dim=1)

# Metrics
accuracy = Accuracy(task="multiclass", num_classes=len(CLASSES))
confmat = ConfusionMatrix(task="multiclass", num_classes=len(CLASSES))
f1 = F1Score(task="multiclass", num_classes=len(CLASSES), average="macro")
precision = Precision(task="multiclass", num_classes=len(CLASSES), average="macro")
recall = Recall(task="multiclass", num_classes=len(CLASSES), average="macro")

print("Accuracy:", accuracy(predictions, y_test_indices).item())
print("F1 Score:", f1(predictions, y_test_indices).item())
print("Precision:", precision(predictions, y_test_indices).item())
print("Recall:", recall(predictions, y_test_indices).item())
print("Confusion Matrix:\n", confmat(predictions, y_test_indices).cpu().numpy())
