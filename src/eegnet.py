"""
EEGNet (Lawhern et al., 2018), standard formulation.

Block 1: temporal conv (learns frequency filters) -> depthwise spatial conv
          (learns one fixed spatial filter bank per temporal filter,
          constrained per-channel across all electrodes) -> BN -> ELU ->
          avg pool -> dropout
Block 2: separable conv (depthwise temporal + pointwise) -> BN -> ELU ->
          avg pool -> dropout
Classifier: flatten -> linear

The depthwise spatial conv (`self.spatial`) is the layer we come back to
in the Part 1 failure analysis and Part 2 improvement: it has kernel
shape (n_channels, 1), i.e. one weight per electrode per filter, learned
globally across the training set and then frozen at inference. There is
no mechanism for it to adapt to a new subject's electrode-to-source
mapping.
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_channels=64, n_times=256, n_classes=2,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5,
                 return_features=False):
        super().__init__()
        self.return_features = return_features

        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        # depthwise spatial filter: one (n_channels x 1) filter per temporal map,
        # groups=F1 so filters don't mix across temporal-frequency maps.
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            feat_dim = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(feat_dim, n_classes)

    def _forward_features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x.flatten(1)

    def forward(self, x):
        # x: (batch, n_channels, n_times) -> add channel dim for conv2d
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feats = self._forward_features(x)
        logits = self.classifier(feats)
        if self.return_features:
            return logits, feats
        return logits

    def spatial_filter_weights(self):
        """Expose the depthwise spatial conv weights for inspection (Part 1 analysis)."""
        return self.spatial[0].weight.detach().clone()  # (F1*D, 1, n_channels, 1)
