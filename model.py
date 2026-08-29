import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    """1D Denoising Autoencoder for ECG telemetry restoration."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                32, 16, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(
                16, 1, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class ECGClassifier(nn.Module):
    """1D CNN Arrhythmia Diagnostic Classifier."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))
