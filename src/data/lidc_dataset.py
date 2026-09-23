import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

MORPH_COLS = [
    "subtlety_mean", "internalStructure_mean", "calcification_mean",
    "sphericity_mean", "margin_mean", "lobulation_mean", "spiculation_mean", "texture_mean",
    "diameter_mm", "volume_mm3",
    "subtlety_var", "calcification_var", "sphericity_var", "margin_var",
    "lobulation_var", "spiculation_var", "texture_var"
]

def load_splits(splits_path: str) -> dict:
    with open(splits_path, "r") as f:
        return json.load(f)

class NoduleDataset(Dataset):
    """
    PyTorch Dataset for LIDC-IDRI Nodule Multimodal / Morphological Metadata.
    """
    def __init__(self, df: pd.DataFrame, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

        self.features = self.df[MORPH_COLS].values.astype(np.float32)
        self.features = np.nan_to_num(self.features, nan=0.0)

        self.mean_ratings = self.df["malignancy_mean"].values.astype(np.float32)
        self.var_ratings = self.df["malignancy_var"].values.astype(np.float32)
        self.var_ratings = np.nan_to_num(self.var_ratings, nan=0.0)

        self.consensus_labels = self.df["malignancy_consensus"].values.astype(np.int64) - 1
        self.conflict_flags = self.df["is_conflict"].values.astype(np.float32)
        self.reader_counts = self.df["num_readers"].values.astype(np.float32)

        distributions = []
        for _, row in self.df.iterrows():
            dist = np.array([row[f"rate_{k}"] for k in range(1, 6)], dtype=np.float32)
            total = dist.sum()
            if total > 0:
                dist = dist / total
            else:
                dist = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
            distributions.append(dist)
        self.distributions = np.array(distributions, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.tensor(self.features[idx]),
            "mean_rating": torch.tensor(self.mean_ratings[idx]),
            "var_rating": torch.tensor(self.var_ratings[idx]),
            "consensus_label": torch.tensor(self.consensus_labels[idx]),
            "conflict_flag": torch.tensor(self.conflict_flags[idx]),
            "reader_count": torch.tensor(self.reader_counts[idx]),
            "distribution": torch.tensor(self.distributions[idx]),
            "nodule_id": self.df.iloc[idx].get("nodule_id", str(idx))
        }


class Nodule3DDataset(Dataset):
    """
    PyTorch Dataset loading 3D Volumetric CT patches (.npz) and aligned metadata.
    """
    def __init__(self, df: pd.DataFrame, crops_dir: str, target_shape=(32, 32, 32)):
        self.df = df.reset_index(drop=True)
        self.crops_dir = crops_dir
        self.target_shape = target_shape

        self.features = self.df[MORPH_COLS].values.astype(np.float32)
        self.features = np.nan_to_num(self.features, nan=0.0)

        self.mean_ratings = self.df["malignancy_mean"].values.astype(np.float32)
        self.var_ratings = self.df["malignancy_var"].values.astype(np.float32)
        self.var_ratings = np.nan_to_num(self.var_ratings, nan=0.0)
        self.conflict_flags = self.df["is_conflict"].values.astype(np.float32)
        self.reader_counts = self.df["num_readers"].values.astype(np.float32)

        distributions = []
        for _, row in self.df.iterrows():
            dist = np.array([row[f"rate_{k}"] for k in range(1, 6)], dtype=np.float32)
            total = dist.sum()
            dist = dist / total if total > 0 else np.ones(5) / 5.0
            distributions.append(dist)
        self.distributions = np.array(distributions, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _resize_volume(self, vol: np.ndarray) -> np.ndarray:
        # Simple trilinear/nearest resizing to target shape
        from scipy.ndimage import zoom
        factors = [t / s for t, s in zip(self.target_shape, vol.shape)]
        return zoom(vol, factors, order=1)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        nodule_id = str(row.get("nodule_id", ""))
        crop_path = os.path.join(self.crops_dir, f"{nodule_id}.npz")
        
        if os.path.exists(crop_path):
            data = np.load(crop_path)
            vol = data["volume"].astype(np.float32)
            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
            if vol.shape != self.target_shape:
                vol = self._resize_volume(vol)
        else:
            vol = np.zeros(self.target_shape, dtype=np.float32)

        vol = np.expand_dims(vol, axis=0) # (1, D, H, W)

        return {
            "volume": torch.tensor(vol, dtype=torch.float32),
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "mean_rating": torch.tensor(self.mean_ratings[idx], dtype=torch.float32),
            "var_rating": torch.tensor(self.var_ratings[idx], dtype=torch.float32),
            "conflict_flag": torch.tensor(self.conflict_flags[idx], dtype=torch.float32),
            "reader_count": torch.tensor(self.reader_counts[idx], dtype=torch.float32),
            "distribution": torch.tensor(self.distributions[idx], dtype=torch.float32),
            "nodule_id": nodule_id
        }


class FlexNoduleDataset(Dataset):
    """
    Dataset supporting custom feature sub-selection for ablation experiments.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list):
        self.df = df.reset_index(drop=True)
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.features = np.nan_to_num(self.features, nan=0.0)

        self.mean_ratings = self.df["malignancy_mean"].values.astype(np.float32)
        self.var_ratings = np.nan_to_num(self.df["malignancy_var"].values.astype(np.float32), nan=0.0)
        self.conflict_flags = self.df["is_conflict"].values.astype(np.float32)
        self.reader_counts = self.df["num_readers"].values.astype(np.float32)

        distributions = []
        for _, row in self.df.iterrows():
            dist = np.array([row[f"rate_{k}"] for k in range(1, 6)], dtype=np.float32)
            total = dist.sum()
            dist = dist / total if total > 0 else np.ones(5) / 5.0
            distributions.append(dist)
        self.distributions = np.array(distributions, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "mean_rating": torch.tensor(self.mean_ratings[idx], dtype=torch.float32),
            "var_rating": torch.tensor(self.var_ratings[idx], dtype=torch.float32),
            "conflict_flag": torch.tensor(self.conflict_flags[idx], dtype=torch.float32),
            "reader_count": torch.tensor(self.reader_counts[idx], dtype=torch.float32),
            "distribution": torch.tensor(self.distributions[idx], dtype=torch.float32)
        }
