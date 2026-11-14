import torch
import torch.nn as nn


class WaferCNN(nn.Module):
    """Small CNN for 64x64 wafer maps -> 9 defect classes."""

    def __init__(self, num_classes: int = 9):
        super().__init__()

        # Feature extractor: 3 x (Conv -> BatchNorm -> ReLU -> Pool)
        # Shapes:  (1, 64, 64) -> (16, 32, 32) -> (32, 16, 16) -> (64, 8, 8)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Flatten -> 128-number "fingerprint" of the wafer
        self.embedding = nn.Sequential(
            nn.Flatten(),                  # (64, 8, 8) -> 4096
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Final vote: 128 -> 9 class scores
        self.classifier = nn.Linear(128, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 128-number fingerprint (used for 'similar wafers' on Day 12)."""
        return self.embedding(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class scores (logits). Softmax is applied when predicting."""
        return self.classifier(self.embed(x))