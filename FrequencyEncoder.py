import torch
import torch.nn as nn


class FrequencyEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=512):
        super(FrequencyEncoder, self).__init__()

        # FFT splits each channel into real + imaginary, so channels double
        freq_channels = in_channels * 2

        # Lightweight CNN to extract features from frequency map
        self.cnn = nn.Sequential(
            # Layer 1: 6 -> 64
            nn.Conv2d(freq_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Collapse spatial dims to (B, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Project to match CLIP embedding dim
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        # Apply 2D FFT
        x_fft = torch.fft.fft2(x)

        # Separate real and imaginary parts, concat along channel dim
        x_freq = torch.cat([x_fft.real, x_fft.imag], dim=1)  # (B, 2C, H, W)

        # Extract frequency features
        features = self.cnn(x_freq)       # (B, 256, H, W)
        features = self.gap(features)     # (B, 256, 1, 1)
        features = features.flatten(1)   # (B, 256)

        return self.fc(features)          # (B, out_dim)


if __name__ == "__main__":
    model = FrequencyEncoder(in_channels=3, out_dim=512)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")  # (4, 3, 224, 224)
    print(f"Output shape: {output.shape}")       # (4, 512)