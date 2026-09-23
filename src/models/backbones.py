import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone3D(nn.Module):
    """
    3D Volumetric Convolutional Backbone for Thoracic CT Nodule Patches.
    Processes volumetric inputs of shape (B, 1, D, H, W).
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 2 * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_block(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


class FastKANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Linear Layer with learnable B-spline / radial basis activations.
    """
    def __init__(self, in_features: int, out_features: int, num_grids: int = 8, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, num_grids) * 0.1)
        self.grid = nn.Parameter(torch.linspace(-2.0, 2.0, num_grids), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(F.silu(x), self.base_weight)
        x_exp = x.unsqueeze(-1)
        basis = torch.exp(-((x_exp - self.grid) ** 2) / 0.5)
        spline_out = torch.einsum("bij,oij->bo", basis, self.spline_weight)
        return base_out + spline_out
