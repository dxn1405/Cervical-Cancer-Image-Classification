import torch.nn as nn
import timm  # Using timm for pretrained DeiT

class DeiT(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, img_size=224, drop_rate=0.1):
        super().__init__()
        self.deit = timm.create_model(
            "deit_base_distilled_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            img_size=img_size
        )
        self.feature_norm = nn.LayerNorm(self.deit.embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.deit.embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.deit.embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(drop_rate/2),
            nn.Linear(1024, num_classes)
        )
        self._init_weights(self.head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.deit.forward_features(x)
        # first token is class, second is distill, combine
        cls, dist = x[:, 0], x[:, 1]
        feat = self.feature_norm(0.5 * cls + 0.5 * dist)
        return self.head(feat)