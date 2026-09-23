#!/usr/bin/env python3
"""
Benchmark 2024-2026 SOTA Models & Classical Baselines on LIDC-IDRI Multi-Rater Disagreement


Models Evaluated:
Panel A: Classical Multi-Rater & Deterministic Baselines
1. Det-CE: Conventional Cross-Entropy on rounded consensus class
2. Det-Reg: Deterministic Scalar MSE Regression
3. Hetero-Gauss: Heteroscedastic Gaussian NLL
4. MultiRater-Ensemble: 4-Head Reader Ensemble simulating distinct radiologists

Panel B: Recent SOTA Multi-Rater Models (2024-2026)
5. LDL-Net (MedIA 2024): Label Distribution Learning with Simplex KL Divergence
6. DP-CrowdNet (CVPR 2024): Diversified & Personalized Multi-Rater Network
7. A3-Net (CVPR 2025): Annotation Ambiguity Aware Network with Adaptive Loss
8. UCE-Net : Uncertainty Co-Estimator Dual-Stream Network
9. KAN-Nodule (2025/2026): Kolmogorov-Arnold Network with Learnable Spline Activations

Panel C: Proposed Framework
10. MultiRater-Dirichlet (Proposed): Second-Order Evidential Dirichlet Belief Distribution
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm, wasserstein_distance
from sklearn.metrics import roc_auc_score, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

INPUT_FEATURES = [
    'subtlety_mean', 'calcification_mean', 'sphericity_mean', 
    'margin_mean', 'lobulation_mean', 'spiculation_mean', 
    'texture_mean', 'internalStructure_mean',
    'subtlety_std', 'calcification_std', 'sphericity_std', 
    'margin_std', 'lobulation_std', 'spiculation_std', 
    'texture_std', 'mean_diameter', 'mean_volume'
]

# ----------------- Datasets -----------------
class NoduleDataset(Dataset):
    def __init__(self, df, mean_std_norm=None):
        self.df = df.reset_index(drop=True)
        X = self.df[INPUT_FEATURES].values.astype(np.float32)

        if mean_std_norm is None:
            self.mu = np.mean(X, axis=0)
            self.sig = np.std(X, axis=0) + 1e-6
        else:
            self.mu, self.sig = mean_std_norm

        self.X = (X - self.mu) / self.sig
        
        self.y_mean = self.df['malignancy_mean'].values.astype(np.float32)
        self.y_std = self.df['malignancy_std'].values.astype(np.float32)
        self.y_class = np.clip(np.round(self.y_mean).astype(int) - 1, 0, 4)
        self.has_conflict = self.df['malignancy_direct_conflict'].values.astype(np.float32)
        self.high_disagree = self.df['malignancy_high_disagreement'].values.astype(np.float32)
        self.num_readers = self.df['num_readers'].values.astype(int)

        distributions = []
        padded_ratings_list = []
        rating_masks = []
        raw_ratings = []

        for _, row in self.df.iterrows():
            ratings = [int(v) for v in str(row['malignancy_ratings']).split(',')]
            dist = np.zeros(5, dtype=np.float32)
            for r in ratings:
                dist[r - 1] += 1.0
            dist /= len(ratings)
            distributions.append(dist)
            raw_ratings.append(ratings)

            p_rat = np.zeros(4, dtype=np.float32)
            p_mask = np.zeros(4, dtype=np.float32)
            for i, r in enumerate(ratings[:4]):
                p_rat[i] = float(r)
                p_mask[i] = 1.0
            padded_ratings_list.append(p_rat)
            rating_masks.append(p_mask)

        self.distributions = np.array(distributions, dtype=np.float32)
        self.padded_ratings = np.array(padded_ratings_list, dtype=np.float32)
        self.rating_masks = np.array(rating_masks, dtype=np.float32)
        self.raw_ratings = raw_ratings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.X[idx]),
            'y_mean': torch.tensor(self.y_mean[idx]),
            'y_std': torch.tensor(self.y_std[idx]),
            'y_class': torch.tensor(self.y_class[idx], dtype=torch.long),
            'has_conflict': torch.tensor(self.has_conflict[idx]),
            'high_disagree': torch.tensor(self.high_disagree[idx]),
            'dist': torch.tensor(self.distributions[idx]),
            'padded_ratings': torch.tensor(self.padded_ratings[idx]),
            'rating_mask': torch.tensor(self.rating_masks[idx]),
            'num_readers': self.num_readers[idx]
        }

# ----------------- Fast KAN Linear Layer -----------------
class FastKANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_grids=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / np.sqrt(in_features)))
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, num_grids) * 0.1)
        grid = torch.linspace(-2.0, 2.0, num_grids)
        self.register_buffer('grid', grid)
        self.inv_sigma = 1.0 / (grid[1] - grid[0])

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)
        x_expanded = x.unsqueeze(-1)
        basis = torch.exp(-((x_expanded - self.grid) * self.inv_sigma)**2)
        spline = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        return base + spline

# ----------------- Model Architectures -----------------

# Panel A: Classical Models
class DetCEModel(nn.Module):
    def __init__(self, in_dim=len(INPUT_FEATURES), hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)
        )
    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        classes = torch.arange(1, 6, device=x.device, dtype=torch.float32)
        pred_mean = (probs * classes).sum(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return {'logits': logits, 'probs': probs, 'pred_mean': pred_mean, 'uncertainty': entropy}

class DetRegModel(nn.Module):
    def __init__(self, in_dim=len(INPUT_FEATURES), hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        pred_mean = self.net(x).squeeze(-1) + 3.0
        return {'pred_mean': pred_mean, 'uncertainty': torch.zeros_like(pred_mean)}

class HeteroGaussModel(nn.Module):
    def __init__(self, in_dim=len(INPUT_FEATURES), hidden_dim=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mean_head(h).squeeze(-1) + 3.0
        logvar = torch.clamp(self.logvar_head(h).squeeze(-1), -4.0, 3.0)
        std = torch.exp(0.5 * logvar)
        return {'pred_mean': mu, 'pred_std': std, 'logvar': logvar, 'uncertainty': std}

class MultiRaterEnsembleModel(nn.Module):
    def __init__(self, in_dim=len(INPUT_FEATURES), hidden_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_heads)
        ])

    def forward(self, x):
        h = self.trunk(x)
        preds = torch.stack([head(h).squeeze(-1) + 3.0 for head in self.heads], dim=1)
        mean_pred = preds.mean(dim=1)
        std_pred = preds.std(dim=1, unbiased=False)
        return {'head_preds': preds, 'pred_mean': mean_pred, 'pred_std': std_pred, 'uncertainty': std_pred}

# Panel B: 2024-2026 SOTA Models
class LDLNetModel(nn.Module):
    """Label Distribution Learning on Medical Disagreement (MedIA 2024)"""
    def __init__(self, in_dim=len(INPUT_FEATURES), hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)
        )
    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        classes = torch.arange(1, 6, device=x.device, dtype=torch.float32)
        pred_mean = (probs * classes).sum(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return {'logits': logits, 'probs': probs, 'pred_mean': pred_mean, 'uncertainty': entropy}

class DPCrowdNetModel(nn.Module):
    """Diversified & Personalized Multi-Rater Network (CVPR 2024)"""
    def __init__(self, in_dim=len(INPUT_FEATURES), hidden_dim=64, num_raters=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        self.rater_emb = nn.Embedding(num_raters, 8)
        self.rater_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2 + 8, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            ) for _ in range(num_raters)
        ])

    def forward(self, x):
        h = self.trunk(x)
        head_preds = []
        for i in range(4):
            emb = self.rater_emb.weight[i].unsqueeze(0).expand(x.size(0), -1)
            h_i = torch.cat([h, emb], dim=-1)
            pred_i = self.rater_heads[i](h_i).squeeze(-1) + 3.0
            head_preds.append(pred_i)
        preds = torch.stack(head_preds, dim=1)
        pred_mean = preds.mean(dim=1)
        pred_var = preds.var(dim=1, unbiased=False)
        return {'head_preds': preds, 'pred_mean': pred_mean, 'uncertainty': pred_var}

class A3NetModel(nn.Module):
    """Annotation Ambiguity Aware Network (CVPR 2025)"""
    def __init__(self, in_dim=len(INPUT_FEATURES), hidden_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim // 2, 5)
        self.ambiguity_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        probs = F.softmax(logits, dim=-1)
        ambiguity = torch.sigmoid(self.ambiguity_head(feat)).squeeze(-1)
        classes = torch.arange(1, 6, device=x.device, dtype=torch.float32)
        pred_mean = (probs * classes).sum(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        uncertainty = ambiguity * 2.0 + entropy
        return {'logits': logits, 'probs': probs, 'ambiguity': ambiguity, 'pred_mean': pred_mean, 'uncertainty': uncertainty}

class UCENetModel(nn.Module):
    """Uncertainty Co-Estimator Network """
    def __init__(self, in_dim=len(INPUT_FEATURES), hidden_dim=64):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 5)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)
        )

    def forward(self, x):
        l1 = self.branch1(x)
        l2 = self.branch2(x)
        p1 = F.softmax(l1, dim=-1)
        p2 = F.softmax(l2, dim=-1)
        p_avg = 0.5 * (p1 + p2)
        classes = torch.arange(1, 6, device=x.device, dtype=torch.float32)
        pred_mean = (p_avg * classes).sum(dim=-1)
        co_disagreement = torch.norm(p1 - p2, p=2, dim=-1)
        entropy = -(p_avg * torch.log(p_avg + 1e-8)).sum(dim=-1)
        uncertainty = co_disagreement * 3.0 + entropy
        return {'logits1': l1, 'logits2': l2, 'probs': p_avg, 'pred_mean': pred_mean, 'uncertainty': uncertainty}

class KANNoduleModel(nn.Module):
    """Kolmogorov-Arnold Network for Nodule Disagreement (2025/2026)"""
    def __init__(self, in_dim=len(INPUT_FEATURES), hidden_dim=48, num_grids=8):
        super().__init__()
        self.kan1 = FastKANLinear(in_dim, hidden_dim, num_grids=num_grids)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(0.2)
        self.kan2 = FastKANLinear(hidden_dim, 24, num_grids=num_grids)
        self.bn2 = nn.BatchNorm1d(24)
        self.out_head = FastKANLinear(24, 5, num_grids=num_grids)

    def forward(self, x):
        h = self.drop(self.bn1(self.kan1(x)))
        h = self.bn2(self.kan2(h))
        logits = self.out_head(h)
        probs = F.softmax(logits, dim=-1)
        classes = torch.arange(1, 6, device=x.device, dtype=torch.float32)
        pred_mean = (probs * classes).sum(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return {'logits': logits, 'probs': probs, 'pred_mean': pred_mean, 'uncertainty': entropy}

# Panel C: Proposed Framework
class MultiRaterDirichletModel(nn.Module):
    """Second-Order Evidential Dirichlet Belief Distribution (Proposed)"""
    def __init__(self, in_dim=len(INPUT_FEATURES), hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)
        )
    def forward(self, x):
        alpha = F.softplus(self.net(x)) + 1.0
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / alpha_0
        classes = torch.arange(1, 6, device=x.device, dtype=torch.float32)
        pred_mean = (probs * classes).sum(dim=-1)
        dirichlet_variance = (alpha * (alpha_0 - alpha) / (alpha_0.pow(2) * (alpha_0 + 1))).sum(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        disagreement_score = dirichlet_variance * 5.0 + entropy
        return {'alpha': alpha, 'probs': probs, 'pred_mean': pred_mean, 'uncertainty': disagreement_score}

# ----------------- ECE Calculation -----------------
def compute_multiclass_ece(probs, true_dists, n_bins=10):
    confidences = np.max(probs, axis=-1)
    pred_classes = np.argmax(probs, axis=-1)
    true_confs = np.array([true_dists[i, pred_classes[i]] for i in range(len(probs))])
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_conf = np.mean(confidences[in_bin])
            avg_acc = np.mean(true_confs[in_bin])
            ece += np.abs(avg_conf - avg_acc) * prop_in_bin
    return float(ece)

# ----------------- Training Loop -----------------
def train_model(model_type, train_loader, cal_loader, epochs=60, lr=1e-3, weight_decay=1e-4):
    set_seed(42)
    if model_type == 'Det-CE':
        model = DetCEModel().to(DEVICE)
    elif model_type == 'Det-Reg':
        model = DetRegModel().to(DEVICE)
    elif model_type == 'Hetero-Gauss':
        model = HeteroGaussModel().to(DEVICE)
    elif model_type == 'MultiRater-Ensemble':
        model = MultiRaterEnsembleModel().to(DEVICE)
    elif model_type == 'LDL-Net':
        model = LDLNetModel().to(DEVICE)
    elif model_type == 'DP-CrowdNet':
        model = DPCrowdNetModel().to(DEVICE)
    elif model_type == 'A3-Net':
        model = A3NetModel().to(DEVICE)
    elif model_type == 'UCE-Net':
        model = UCENetModel().to(DEVICE)
    elif model_type == 'KAN-Nodule':
        model = KANNoduleModel().to(DEVICE)
    elif model_type == 'MultiRater-Dirichlet':
        model = MultiRaterDirichletModel().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    best_state = None

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch['x'].to(DEVICE)
            optimizer.zero_grad()

            if model_type == 'Det-CE':
                out = model(x)
                loss = F.cross_entropy(out['logits'], batch['y_class'].to(DEVICE))
            elif model_type == 'Det-Reg':
                out = model(x)
                loss = F.mse_loss(out['pred_mean'], batch['y_mean'].to(DEVICE))
            elif model_type == 'Hetero-Gauss':
                out = model(x)
                mu = out['pred_mean'].unsqueeze(1)
                logvar = out['logvar'].unsqueeze(1)
                var = torch.exp(logvar)
                ratings = batch['padded_ratings'].to(DEVICE)
                mask = batch['rating_mask'].to(DEVICE)
                sq_err = (ratings - mu).pow(2) * mask
                nll_terms = 0.5 * sq_err / var + 0.5 * logvar * mask
                loss = nll_terms.sum() / mask.sum()
            elif model_type == 'MultiRater-Ensemble':
                out = model(x)
                head_preds = out['head_preds']
                ratings = batch['padded_ratings'].to(DEVICE)
                mask = batch['rating_mask'].to(DEVICE)
                sorted_heads, _ = torch.sort(head_preds, dim=1)
                loss = ((sorted_heads - ratings).pow(2) * mask).sum() / mask.sum()
            elif model_type == 'LDL-Net':
                out = model(x)
                p_true = batch['dist'].to(DEVICE)
                loss = F.kl_div(torch.log(out['probs'] + 1e-8), p_true, reduction='batchmean')
            elif model_type == 'DP-CrowdNet':
                out = model(x)
                head_preds = out['head_preds']
                ratings = batch['padded_ratings'].to(DEVICE)
                mask = batch['rating_mask'].to(DEVICE)
                sorted_heads, _ = torch.sort(head_preds, dim=1)
                fit_loss = ((sorted_heads - ratings).pow(2) * mask).sum() / mask.sum()
                E = model.rater_emb.weight
                orth_loss = torch.norm(torch.matmul(E, E.t()) - torch.eye(4, device=DEVICE), p='fro')
                loss = fit_loss + 0.1 * orth_loss
            elif model_type == 'A3-Net':
                out = model(x)
                ce_per_sample = F.cross_entropy(out['logits'], batch['y_class'].to(DEVICE), reduction='none')
                amb = out['ambiguity']
                weighted_ce = (ce_per_sample * torch.exp(-amb)).mean()
                true_std = batch['y_std'].to(DEVICE)
                amb_loss = F.mse_loss(amb, true_std / 2.0)
                loss = weighted_ce + 0.5 * amb_loss
            elif model_type == 'UCE-Net':
                out = model(x)
                y_c = batch['y_class'].to(DEVICE)
                l1 = F.cross_entropy(out['logits1'], y_c)
                l2 = F.cross_entropy(out['logits2'], y_c)
                p_true = batch['dist'].to(DEVICE)
                co_loss = F.l1_loss(out['probs'], p_true)
                loss = 0.5 * (l1 + l2) + 0.5 * co_loss
            elif model_type == 'KAN-Nodule':
                out = model(x)
                p_true = batch['dist'].to(DEVICE)
                log_probs = torch.log(out['probs'] + 1e-8)
                loss = -(p_true * log_probs).sum(dim=-1).mean()
            elif model_type == 'MultiRater-Dirichlet':
                out = model(x)
                p_true = batch['dist'].to(DEVICE)
                log_probs = torch.log(out['probs'] + 1e-8)
                ce_loss = -(p_true * log_probs).sum(dim=-1).mean()
                pred_var = out['uncertainty']
                emp_var = batch['y_std'].to(DEVICE).pow(2)
                var_loss = F.mse_loss(pred_var, emp_var)
                loss = ce_loss + 0.5 * var_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        scheduler.step()

        # Validation
        model.eval()
        cal_loss = 0.0
        with torch.no_grad():
            for batch in cal_loader:
                x = batch['x'].to(DEVICE)
                if model_type == 'Det-CE':
                    out = model(x)
                    l = F.cross_entropy(out['logits'], batch['y_class'].to(DEVICE))
                elif model_type == 'Det-Reg':
                    out = model(x)
                    l = F.mse_loss(out['pred_mean'], batch['y_mean'].to(DEVICE))
                elif model_type == 'Hetero-Gauss':
                    out = model(x)
                    l = F.mse_loss(out['pred_mean'], batch['y_mean'].to(DEVICE)) + F.mse_loss(out['pred_std'], batch['y_std'].to(DEVICE))
                elif model_type == 'MultiRater-Ensemble':
                    out = model(x)
                    l = F.mse_loss(out['pred_mean'], batch['y_mean'].to(DEVICE)) + F.mse_loss(out['pred_std'], batch['y_std'].to(DEVICE))
                elif model_type == 'LDL-Net':
                    out = model(x)
                    p_true = batch['dist'].to(DEVICE)
                    l = F.kl_div(torch.log(out['probs'] + 1e-8), p_true, reduction='batchmean')
                elif model_type == 'DP-CrowdNet':
                    out = model(x)
                    l = F.mse_loss(out['pred_mean'], batch['y_mean'].to(DEVICE))
                elif model_type == 'A3-Net':
                    out = model(x)
                    l = F.cross_entropy(out['logits'], batch['y_class'].to(DEVICE))
                elif model_type == 'UCE-Net':
                    out = model(x)
                    y_c = batch['y_class'].to(DEVICE)
                    l = 0.5 * (F.cross_entropy(out['logits1'], y_c) + F.cross_entropy(out['logits2'], y_c))
                elif model_type == 'KAN-Nodule':
                    out = model(x)
                    p_true = batch['dist'].to(DEVICE)
                    l = -(p_true * torch.log(out['probs'] + 1e-8)).sum(dim=-1).mean()
                elif model_type == 'MultiRater-Dirichlet':
                    out = model(x)
                    p_true = batch['dist'].to(DEVICE)
                    l = -(p_true * torch.log(out['probs'] + 1e-8)).sum(dim=-1).mean()
                cal_loss += l.item() * len(x)

        cal_loss /= len(cal_loader.dataset)
        if cal_loss < best_loss:
            best_loss = cal_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model

# ----------------- Evaluation -----------------
def evaluate_model(model, model_type, test_loader):
    model.eval()
    all_preds_mean = []
    all_unc = []
    all_true_mean = []
    all_true_std = []
    all_conflict = []
    all_high_disagree = []
    all_true_dists = []
    all_pred_dists = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(DEVICE)
            out = model(x)

            all_preds_mean.extend(out['pred_mean'].cpu().numpy().tolist())
            all_unc.extend(out['uncertainty'].cpu().numpy().tolist())
            all_true_mean.extend(batch['y_mean'].numpy().tolist())
            all_true_std.extend(batch['y_std'].numpy().tolist())
            all_conflict.extend(batch['has_conflict'].numpy().tolist())
            all_high_disagree.extend(batch['high_disagree'].numpy().tolist())
            all_true_dists.extend(batch['dist'].numpy().tolist())

            if 'probs' in out:
                all_pred_dists.extend(out['probs'].cpu().numpy().tolist())
            elif model_type == 'Hetero-Gauss':
                mu = out['pred_mean'].cpu().numpy()
                sig = np.maximum(out['pred_std'].cpu().numpy(), 0.1)
                dists = []
                bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
                for m_val, s_val in zip(mu, sig):
                    cdfs = norm.cdf(bins, loc=m_val, scale=s_val)
                    p = np.diff(cdfs)
                    p = p / (p.sum() + 1e-8)
                    dists.append(p.tolist())
                all_pred_dists.extend(dists)
            elif model_type in ['MultiRater-Ensemble', 'DP-CrowdNet']:
                heads = out['head_preds'].cpu().numpy()
                dists = []
                for h in heads:
                    p = np.zeros(5)
                    for val in h:
                        c = int(np.clip(np.round(val) - 1, 0, 4))
                        p[c] += 0.25
                    dists.append(p.tolist())
                all_pred_dists.extend(dists)

    preds_mean = np.array(all_preds_mean)
    unc = np.array(all_unc)
    true_mean = np.array(all_true_mean)
    true_std = np.array(all_true_std)
    conflict = np.array(all_conflict)
    high_disagree = np.array(all_high_disagree)
    true_dists = np.array(all_true_dists)

    mse = float(mean_squared_error(true_mean, preds_mean))
    corr = float(np.corrcoef(true_mean, preds_mean)[0, 1])

    if len(all_pred_dists) > 0:
        pred_dists = np.array(all_pred_dists)
        w1_dists = [wasserstein_distance([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], u_weights=p_true, v_weights=p_pred)
                    for p_true, p_pred in zip(true_dists, pred_dists)]
        mean_w1 = round(float(np.mean(w1_dists)), 4)
        ece = round(compute_multiclass_ece(pred_dists, true_dists), 4)
    else:
        mean_w1 = None
        ece = None

    if np.all(unc == unc[0]):
        conflict_auc = 0.5000
        high_disagree_auc = 0.5000
    else:
        conflict_auc = round(float(roc_auc_score(conflict, unc)), 4)
        high_disagree_auc = round(float(roc_auc_score(high_disagree, unc)), 4)

    return {
        'Rating_MSE': round(mse, 4),
        'Pearson_r': round(corr, 4),
        'ECE': ece,
        'Wasserstein_EMD': mean_w1,
        'Conflict_AUROC': conflict_auc,
        'High_Disagree_AUROC': high_disagree_auc
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', default='medicine/lidc-reader-disagreement/07-engineering/data/nodules_metadata.csv')
    parser.add_argument('--splits_json', default='medicine/lidc-reader-disagreement/07-engineering/data/patient_splits.json')
    parser.add_argument('--output_dir', default='medicine/lidc-reader-disagreement/04-experiments/runs/gate1')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    df = pd.read_csv(args.metadata_csv)
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train'].copy()
        cal_df = df[df['split'] == 'cal'].copy()
        test_df = df[df['split'] == 'test'].copy()
    elif args.splits_json and os.path.exists(args.splits_json):
        with open(args.splits_json) as f:
            splits = json.load(f)
        patient_to_split = {p: info['split'] for p, info in splits['patients'].items()}
        df['split'] = df['patient_id'].map(patient_to_split)
        train_df = df[df['split'] == 'train'].copy()
        cal_df = df[df['split'] == 'cal'].copy()
        test_df = df[df['split'] == 'test'].copy()
    else:
        raise ValueError('No split column in metadata and no valid splits_json provided.')

    print(f"Nodules: Train={len(train_df)}, Cal={len(cal_df)}, Test={len(test_df)}")

    train_ds = NoduleDataset(train_df)
    cal_ds = NoduleDataset(cal_df, mean_std_norm=(train_ds.mu, train_ds.sig))
    test_ds = NoduleDataset(test_df, mean_std_norm=(train_ds.mu, train_ds.sig))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    models_to_run = [
        # Panel A: Classical Baselines
        'Det-CE',
        'Det-Reg',
        'Hetero-Gauss',
        'MultiRater-Ensemble',
        # Panel B: 2024-2026 SOTA Models
        'LDL-Net',
        'DP-CrowdNet',
        'A3-Net',
        'UCE-Net',
        'KAN-Nodule',
        # Panel C: Proposed Framework
        'MultiRater-Dirichlet'
    ]

    all_results = {}
    print("=========================================================================")
    print("STARTING BENCHMARK: Classical Baselines + 2024-2026 SOTA + Proposed Model")
    print("=========================================================================")
    for m in models_to_run:
        t0 = time.time()
        print(f">>> Training {m} (epochs={args.epochs}, lr=1e-3, seed=42)...")
        trained_model = train_model(m, train_loader, cal_loader, epochs=args.epochs)
        res = evaluate_model(trained_model, m, test_loader)
        res['Time_sec'] = round(time.time() - t0, 2)
        all_results[m] = res
        print(f"[{m}] Evaluation Results: {res}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, 'GATE1_SOTA_2024_2026.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
        print(f">>> Saved full benchmark results to: {out_json}")

if __name__ == '__main__':
    main()
