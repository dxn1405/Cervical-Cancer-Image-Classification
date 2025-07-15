import torch.nn as nn
import timm
import torch

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, img_size=224, drop_rate=0.2):
        super().__init__()
        self.swin = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            drop_path_rate=0.3
        )
        self.num_features = self.swin.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Dropout(drop_rate),
            nn.Linear(self.num_features, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(drop_rate),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(drop_rate/2),
            nn.Linear(512, num_classes)
        )
        self._init_weights(self.head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.swin.forward_features(x)
        return self.head(x)