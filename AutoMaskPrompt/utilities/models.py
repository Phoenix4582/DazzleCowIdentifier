import torch
import torch.nn as nn
import torchvision.models as models

class SelfReIDModel(torch.nn.Module):
    def __init__(self, backbone='resnet50', feature_dim=128):
        super().__init__()
        ldr = f"self.backbone_model = models.{backbone}()"
        print(ldr)
        exec(ldr)
        self.backbone_model.fc = nn.Sequential(self.backbone_model.fc, nn.ReLU(), nn.Linear(1000, feature_dim))

    def forward(self, x):
        return self.backbone_model(x)
