import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=4, pretrained=True):

    
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    
    num_features = model.fc.in_features    
    
    model.fc = nn.Linear(num_features, num_classes)
    
    return model
