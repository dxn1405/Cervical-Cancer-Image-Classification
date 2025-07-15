import torch.nn as nn
import timm

class CrossViT(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            "crossvit_9_240",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=0.1,
            img_size=240
        )
    def forward(self, x):
        return self.model(x)