# model.py
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, pretrained=True, device='cpu'):
    model = models.mobilenet_v2(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    # заменим head
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    model.to(device)
    return model
