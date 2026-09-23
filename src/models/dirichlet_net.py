import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import Backbone3D

def digamma_dirichlet_loss(alpha: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
    """
    Dirichlet distribution cross-entropy loss based on the Digamma function:
    L = \sum_k y_{i,k} [ \psi(S_i) - \psi(\alpha_{i,k}) ]
    """
    S = torch.sum(alpha, dim=-1, keepdim=True)
    expected_log_p = torch.digamma(alpha) - torch.digamma(S)
    loss = -torch.sum(target_distribution * expected_log_p, dim=-1)
    return loss.mean()

def masked_variance_alignment_loss(alpha: torch.Tensor, true_variance: torch.Tensor, reader_counts: torch.Tensor) -> torch.Tensor:
    """
    Masked Variance Alignment Loss:
    Aligns predicted variance \sigma^2(\alpha) with empirical multi-reader sample variance \sigma_i^2,
    explicitly masked with \mathbb{I}(R_i \ge 2) to prevent unobserved single-reader nodules from being
    penalized as zero-disagreement negatives.
    """
    S = torch.sum(alpha, dim=-1, keepdim=True)
    p_hat = alpha / S
    classes = torch.arange(1, alpha.size(-1) + 1, device=alpha.device, dtype=torch.float32)
    expected_rating = torch.sum(p_hat * classes, dim=-1)
    expected_sq_rating = torch.sum(p_hat * (classes ** 2), dim=-1)
    pred_variance = expected_sq_rating - (expected_rating ** 2)
    
    mask = (reader_counts >= 2).float()
    sq_err = (pred_variance - true_variance) ** 2
    if mask.sum() > 0:
        return (sq_err * mask).sum() / mask.sum()
    return torch.tensor(0.0, device=alpha.device)

def kl_dirichlet_uniform_loss(alpha: torch.Tensor) -> torch.Tensor:
    """
    Kullback-Leibler divergence between predicted Dirichlet and uniform prior Dir(1, 1, 1, 1, 1).
    Acts as an epistemic uncertainty regularizer.
    """
    K = alpha.size(-1)
    beta = torch.ones_like(alpha)
    S_alpha = torch.sum(alpha, dim=-1)
    S_beta = torch.sum(beta, dim=-1)
    
    ln_gamma_S_alpha = torch.lgamma(S_alpha)
    ln_gamma_S_beta = torch.lgamma(S_beta)
    sum_ln_gamma_alpha = torch.sum(torch.lgamma(alpha), dim=-1)
    sum_ln_gamma_beta = torch.sum(torch.lgamma(beta), dim=-1)
    
    psi_S_alpha = torch.digamma(S_alpha).unsqueeze(-1)
    psi_alpha = torch.digamma(alpha)
    sum_term = torch.sum((alpha - beta) * (psi_alpha - psi_S_alpha), dim=-1)
    
    kl = ln_gamma_S_alpha - sum_ln_gamma_alpha - ln_gamma_S_beta + sum_ln_gamma_beta + sum_term
    return kl.mean()


class MultiRaterDirichletModel(nn.Module):
    """
    MultiRater-Dirichlet: Second-order evidential Dirichlet model for tabular/morphological inputs.
    Outputs evidence e_k \ge 0 and Dirichlet concentrations \alpha_k = e_k + 1.
    """
    def __init__(self, in_features: int = 17, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> dict:
        logits = self.net(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / S
        uncertainty = alpha.size(-1) / S.squeeze(-1)
        return {"alpha": alpha, "prob": prob, "uncertainty": uncertainty, "evidence": evidence}


class MultiRaterDirichlet3DModel(nn.Module):
    """
    MultiRater-Dirichlet-3D: Track 1 autonomous CT-only evidential model operating directly on raw 3D voxels.
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.backbone = Backbone3D(in_channels=in_channels, feature_dim=feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone(x)
        logits = self.head(feat)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / S
        uncertainty = alpha.size(-1) / S.squeeze(-1)
        return {"alpha": alpha, "prob": prob, "uncertainty": uncertainty, "evidence": evidence, "features": feat}


class MultimodalDirichletModel(nn.Module):
    """
    Multimodal-Dirichlet: Track 2 assisted stream fusing 3D CT volumetric representations with
    17-dimensional radiological/morphological priors.
    """
    def __init__(self, in_channels: int = 1, morph_dim: int = 17, feature_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.backbone = Backbone3D(in_channels=in_channels, feature_dim=feature_dim)
        self.morph_encoder = nn.Sequential(
            nn.Linear(morph_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(feature_dim + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_vol: torch.Tensor, x_morph: torch.Tensor) -> dict:
        vol_feat = self.backbone(x_vol)
        morph_feat = self.morph_encoder(x_morph)
        fused = torch.cat([vol_feat, morph_feat], dim=-1)
        logits = self.fusion_head(fused)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / S
        uncertainty = alpha.size(-1) / S.squeeze(-1)
        return {"alpha": alpha, "prob": prob, "uncertainty": uncertainty, "evidence": evidence}
