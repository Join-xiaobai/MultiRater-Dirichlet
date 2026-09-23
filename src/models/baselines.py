import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import Backbone3D, FastKANLinear

class DetCEModel(nn.Module):
    """Deterministic Cross-Entropy on rounded consensus class."""
    def __init__(self, in_features: int = 17, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> dict:
        logits = self.net(x)
        prob = F.softmax(logits, dim=-1)
        conf, pred = torch.max(prob, dim=-1)
        return {"logits": logits, "prob": prob, "pred": pred + 1, "uncertainty": 1.0 - conf}


class DetRegModel(nn.Module):
    """Deterministic Scalar MSE Regression on mean rating."""
    def __init__(self, in_features: int = 17, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> dict:
        pred_rating = self.net(x).squeeze(-1)
        return {"pred_rating": pred_rating, "uncertainty": torch.zeros_like(pred_rating)}


class HeteroGaussModel(nn.Module):
    """Heteroscedastic Gaussian NLL predicting mean \mu and variance \sigma^2."""
    def __init__(self, in_features: int = 17, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.mu_head = nn.Linear(hidden_dim // 2, 1)
        self.var_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.shared(x)
        mu = self.mu_head(feat).squeeze(-1)
        log_var = self.var_head(feat).squeeze(-1)
        var = F.softplus(log_var) + 1e-4
        return {"mu": mu, "var": var, "uncertainty": var}


class MultiRaterEnsembleModel(nn.Module):
    """4-Head Reader Ensemble simulating 4 distinct radiologists."""
    def __init__(self, in_features: int = 17, hidden_dim: int = 64, num_raters: int = 4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_raters)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.shared(x)
        ratings = [h(feat).squeeze(-1) for h in self.heads]
        stacked = torch.stack(ratings, dim=-1)
        mean_rating = stacked.mean(dim=-1)
        var_rating = stacked.var(dim=-1, unbiased=False)
        return {"ratings": stacked, "mean_rating": mean_rating, "var_rating": var_rating, "uncertainty": var_rating}


class LDLNetModel(nn.Module):
    """Label Distribution Learning with Simplex KL Divergence (MedIA 2024)."""
    def __init__(self, in_features: int = 17, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> dict:
        logits = self.net(x)
        prob = F.softmax(logits, dim=-1)
        classes = torch.arange(1, prob.size(-1) + 1, device=x.device, dtype=torch.float32)
        mean_rating = torch.sum(prob * classes, dim=-1)
        sq_rating = torch.sum(prob * (classes ** 2), dim=-1)
        var_rating = sq_rating - (mean_rating ** 2)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)
        return {"prob": prob, "mean_rating": mean_rating, "var_rating": var_rating, "uncertainty": var_rating}


class DPCrowdNetModel(nn.Module):
    """Diversified & Personalized Multi-Rater Network (CVPR 2024)."""
    def __init__(self, in_features: int = 17, hidden_dim: int = 64, num_raters: int = 4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rater_embeddings = nn.Parameter(torch.randn(num_raters, 16) * 0.1)
        self.personal_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + 16, 32),
                nn.SiLU(),
                nn.Linear(32, 1)
            ) for _ in range(num_raters)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.shared(x)
        outs = []
        for i, head in enumerate(self.personal_heads):
            emb = self.rater_embeddings[i].unsqueeze(0).expand(feat.size(0), -1)
            fused = torch.cat([feat, emb], dim=-1)
            outs.append(head(fused).squeeze(-1))
        stacked = torch.stack(outs, dim=-1)
        return {"ratings": stacked, "mean_rating": stacked.mean(dim=-1), "uncertainty": stacked.var(dim=-1, unbiased=False)}


class A3NetModel(nn.Module):
    """Annotation Ambiguity Aware Network with Dynamic Loss Attenuation (CVPR 2025)."""
    def __init__(self, in_features: int = 17, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU()
        )
        self.cls_head = nn.Linear(hidden_dim // 2, num_classes)
        self.ambiguity_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone(x)
        logits = self.cls_head(feat)
        prob = F.softmax(logits, dim=-1)
        ambiguity = F.softplus(self.ambiguity_head(feat)).squeeze(-1)
        return {"logits": logits, "prob": prob, "ambiguity": ambiguity, "uncertainty": ambiguity}


class UCENetModel(nn.Module):
    """Uncertainty Co-Estimator Dual-Stream Network ."""
    def __init__(self, in_features: int = 17, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.stream_a = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        self.stream_b = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> dict:
        prob_a = F.softmax(self.stream_a(x), dim=-1)
        prob_b = F.softmax(self.stream_b(x), dim=-1)
        prob_mean = 0.5 * (prob_a + prob_b)
        jsd = 0.5 * (F.kl_div(prob_a.log(), prob_mean, reduction="none").sum(-1) +
                     F.kl_div(prob_b.log(), prob_mean, reduction="none").sum(-1))
        return {"prob": prob_mean, "prob_a": prob_a, "prob_b": prob_b, "uncertainty": jsd}


class KANNoduleModel(nn.Module):
    """Kolmogorov-Arnold Network (KAN) adapted baseline with learnable spline activations (2024-2026)."""
    def __init__(self, in_features: int = 17, hidden_dim: int = 32, num_classes: int = 5):
        super().__init__()
        self.kan1 = FastKANLinear(in_features, hidden_dim, num_grids=6)
        self.kan2 = FastKANLinear(hidden_dim, hidden_dim // 2, num_grids=6)
        self.out_head = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        h1 = self.kan1(x)
        h2 = self.kan2(h1)
        logits = self.out_head(h2)
        prob = F.softmax(logits, dim=-1)
        classes = torch.arange(1, prob.size(-1) + 1, device=x.device, dtype=torch.float32)
        mean_rating = torch.sum(prob * classes, dim=-1)
        var_rating = torch.sum(prob * (classes ** 2), dim=-1) - (mean_rating ** 2)
        return {"prob": prob, "mean_rating": mean_rating, "var_rating": var_rating, "uncertainty": var_rating}


class DetCE3DModel(nn.Module):
    """3D CNN with Cross-Entropy on rounded consensus class."""
    def __init__(self, in_channels: int = 1, feature_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.backbone = Backbone3D(in_channels=in_channels, feature_dim=feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone(x)
        logits = self.head(feat)
        prob = F.softmax(logits, dim=-1)
        conf, pred = torch.max(prob, dim=-1)
        return {"logits": logits, "prob": prob, "pred": pred + 1, "uncertainty": 1.0 - conf}


class HeteroGauss3DModel(nn.Module):
    """3D CNN predicting Gaussian mean and variance."""
    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        super().__init__()
        self.backbone = Backbone3D(in_channels=in_channels, feature_dim=feature_dim)
        self.mu_head = nn.Linear(feature_dim, 1)
        self.var_head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone(x)
        mu = self.mu_head(feat).squeeze(-1)
        log_var = self.var_head(feat).squeeze(-1)
        var = F.softplus(log_var) + 1e-4
        return {"mu": mu, "var": var, "uncertainty": var}
