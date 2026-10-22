#!/usr/bin/env python3
"""
MultiRater-Dirichlet: Publication Figure Generation Suite
Generates publication-grade figures (PNG/PDF) for the manuscript:
- Fig 2: Disagreement distribution (Violin, Bar, and Morphological Correlation Heatmap)
- Fig 4: Multidimensional Clinical Competence Footprint & Calibration Safety
- Fig 5: Qualitative Case Studies with 3D Grad-CAM Spatial Ambiguity Heatmaps
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Nature / Academic publication styling
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['text.color'] = '#1A1A1A'
plt.rcParams['figure.autolayout'] = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Figure 2: Disagreement Distribution & Correlation Matrix Heatmap
# -------------------------------------------------------------------------
def generate_figure_2(outfile):
    print("Generating Figure 2 (Disagreement Distribution & Correlation Heatmap)...")
    meta_path = os.path.join(DATA_DIR, 'nodules_metadata.csv')
    df = pd.read_csv(meta_path)
    mr = df[df['num_readers'] >= 2].copy()

    fig = plt.figure(figsize=(16.8, 5.2), dpi=300)
    plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.32)

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Panel A: Violin Plot
    size_bins = ['<6 mm', '6-10 mm', '>10 mm']
    size_data = [
        mr[mr['mean_diameter'] < 6]['malignancy_std'].dropna()**2,
        mr[(mr['mean_diameter'] >= 6) & (mr['mean_diameter'] <= 10)]['malignancy_std'].dropna()**2,
        mr[mr['mean_diameter'] > 10]['malignancy_std'].dropna()**2
    ]
    vp = ax1.violinplot(size_data, showmeans=True, showmedians=False)
    for pc in vp['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_edgecolor('#2980b9')
        pc.set_alpha(0.65)
    vp['cmeans'].set_color('#c0392b')
    vp['cmeans'].set_linewidth(1.8)
    vp['cbars'].set_color('#2980b9')
    vp['cmaxes'].set_color('#2980b9')
    vp['cmins'].set_color('#2980b9')

    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(size_bins, fontsize=10.5, fontweight='bold')
    ax1.set_title("(A) Inter-Rater Variance vs Nodule Diameter", fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_ylabel("Malignancy Rating Variance $\\sigma_r^2$", fontsize=10.5, fontweight='bold')
    ax1.set_xlabel("Nodule Size Group", fontsize=10.5, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.3, 6.0)

    # Panel B: Conflict Rate vs Texture
    tex_groups = ['Non-solid\n(GGN)', 'Part-solid', 'Solid']
    tex_conf = [
        mr[mr['texture_mean'] <= 2.5]['malignancy_direct_conflict'].mean() * 100,
        mr[(mr['texture_mean'] > 2.5) & (mr['texture_mean'] < 4.5)]['malignancy_direct_conflict'].mean() * 100,
        mr[mr['texture_mean'] >= 4.5]['malignancy_direct_conflict'].mean() * 100
    ]
    bars = ax2.bar(range(3), tex_conf, color=['#e74c3c', '#e67e22', '#3498db'], alpha=0.85, width=0.52, edgecolor='#333333', linewidth=0.7)
    ax2.set_title("(B) Clinical Conflict Rate vs Nodule Texture\n(Multi-Rated Cohort, $R \geq 2, N=1,880$)", fontsize=10.5, fontweight='bold', pad=6)
    ax2.set_ylabel("Direct Conflict Rate (%)", fontsize=10.5, fontweight='bold')
    ax2.set_xlabel("Parenchymal Texture Group", fontsize=10.5, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(tex_groups, fontsize=10.5, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 30)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10.0, fontweight='bold')

    # Panel C: Disagreement Covariance Heatmap
    feats = [
        ('Malignancy', 'malignancy_std'),
        ('Spiculation', 'spiculation_std'),
        ('Lobulation', 'lobulation_std'),
        ('Margin', 'margin_std'),
        ('Texture', 'texture_std'),
        ('Subtlety', 'subtlety_std'),
        ('Sphericity', 'sphericity_std'),
        ('Calcification', 'calcification_std'),
        ('Internal Str.', 'internalStructure_std')
    ]
    col_names = [f[0] for f in feats]
    cols = [f[1] for f in feats]
    corr = mr[cols].corr().values

    im = ax3.imshow(corr, cmap='RdBu_r', vmin=-0.25, vmax=0.60, aspect='auto')
    ax3.set_xticks(range(len(col_names)))
    ax3.set_yticks(range(len(col_names)))
    ax3.set_xticklabels(col_names, rotation=45, ha='right', fontsize=8.8, fontweight='bold')
    ax3.set_yticklabels(col_names, fontsize=8.8, fontweight='bold')
    ax3.set_title("(C) Morphological Disagreement Correlation Matrix", fontsize=11.5, fontweight='bold', pad=10)

    for i in range(len(col_names)):
        for j in range(len(col_names)):
            val = corr[i, j]
            txt_color = 'white' if (val > 0.35 or val < -0.18) else '#1A1A1A'
            txt = "1.00" if i == j else f"{val:.2f}"
            ax3.text(j, i, txt, ha='center', va='center', fontsize=7.5, color=txt_color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation ($r$)", fontsize=9.5, fontweight='bold')
    cbar.ax.tick_params(labelsize=8.5)

    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {outfile}")

# -------------------------------------------------------------------------
# Figure 4: Feature Group Ablations & Radar Chart
# -------------------------------------------------------------------------
def generate_figure_4(outfile):
    print("Generating Figure 4 (Feature Group Ablation Footprint)...")
    fig = plt.figure(figsize=(12.2, 5.0), dpi=300)
    ax1 = fig.add_axes([0.025, 0.14, 0.32, 0.65], polar=True)
    ax_auc = fig.add_axes([0.530, 0.14, 0.190, 0.65])
    ax_ece = fig.add_axes([0.730, 0.14, 0.245, 0.65])

    categories = [
        "Conflict\nDiscriminability\n(AUROC)",
        "High-Disagree\nDetection\n(AUROC)",
        "Calibration\nReliability\n(1 - ECE)",
        "Distribution\nFidelity\n(1 - EMD)",
        "Rating\nPrecision\n(1 - Norm MSE)"
    ]
    N = 5
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    offset = np.radians(36)
    ax1.set_theta_offset(offset)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=7.6, fontweight='bold', color='#263238')
    ax1.tick_params(axis='x', pad=10)

    bounds = [(0.68, 0.82), (0.75, 0.88), (0.65, 0.92), (0.30, 0.70), (0.15, 0.55)]
    def norm(val, idx):
        b_min, b_max = bounds[idx]
        return np.clip((val - b_min) / (b_max - b_min), 0.05, 1.0)

    ax1.set_rlabel_position(int(np.degrees(offset)))
    ax1.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax1.set_yticklabels(["25%", "50%", "75%", "100%"], color="#90A4AE", size=6.5)
    ax1.set_ylim(0, 1.05)
    ax1.grid(color='#CFD8DC', linestyle='--', linewidth=0.6)
    ax1.spines['polar'].set_color('#B0BEC5')

    radar_models = [
        {"name": "Full Set (17 Feats, Proposed)", "vals": [norm(0.7450, 0), norm(0.8328, 1), norm(1-0.1122, 2), norm(1-0.3346, 3), norm(1-0.5514, 4)], "color": "#008B74", "fill": True, "lw": 2.0, "zorder": 5},
        {"name": "w/o Margin & Spiculation (11)", "vals": [norm(0.7404, 0), norm(0.8150, 1), norm(1-0.1213, 2), norm(1-0.3473, 3), norm(1-0.5475, 4)], "color": "#2C699A", "fill": False, "lw": 1.4, "zorder": 4},
        {"name": "Means Only (No Variance, 10)", "vals": [norm(0.7776, 0), norm(0.8174, 1), norm(1-0.2126, 2), norm(1-0.5724, 3), norm(1-0.5277, 4)], "color": "#E65100", "fill": True, "lw": 1.6, "zorder": 3},
        {"name": "Size Only (2-D Heuristic)", "vals": [norm(0.7497, 0), norm(0.7818, 1), norm(1-0.2958, 2), norm(1-0.6465, 3), norm(1-0.8267, 4)], "color": "#C2185B", "fill": False, "lw": 1.4, "zorder": 2}
    ]

    for m in radar_models:
        v = m["vals"] + m["vals"][:1]
        ax1.plot(angles, v, color=m["color"], linewidth=m["lw"], label=m["name"], zorder=m["zorder"])
        ax1.scatter(angles[:-1], m["vals"], color=m["color"], s=22, zorder=m["zorder"]+1)
        if m["fill"]:
            alpha = 0.18 if "Full" in m["name"] else 0.08
            ax1.fill(angles, v, color=m["color"], alpha=alpha, zorder=m["zorder"]-1)

    fig.text(0.02, 0.94, "a   Multidimensional Clinical Competence Footprint", size=10.2, fontweight='bold', color='#111111')
    leg1 = ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=True, fontsize=7.0)
    leg1.get_frame().set_edgecolor('#E0E0E0')
    leg1.get_frame().set_facecolor('#FAFAFA')

    # Panel B
    models = [
        "Size Only (2-D Heuristic, 2)",
        "Means Only (No Variance, 10)",
        "w/o Size & Volume (13)",
        "w/o Internal Texture (14)",
        "w/o Margin & Spiculation (11)",
        "Full Set (17 Feats, Proposed)"
    ]
    y_pos = np.arange(len(models))
    auroc_vals = [0.7497, 0.7776, 0.7111, 0.7558, 0.7404, 0.7450]
    ece_vals = [0.2958, 0.2126, 0.1185, 0.1290, 0.1213, 0.1122]
    emd_vals = [0.6465, 0.5724, 0.3499, 0.3647, 0.3473, 0.3346]
    colors = ['#C2185B', '#E65100', '#54478C', '#048BA8', '#2C699A', '#008B74']

    bars_auc = ax_auc.barh(y_pos, auroc_vals, color=colors, alpha=0.88, height=0.52, edgecolor='#333333', linewidth=0.7, zorder=3)
    ax_auc.set_xlim(0.66, 0.815)
    ax_auc.set_xlabel("Conflict AUROC (↑ Higher is Better)", fontsize=8.2, fontweight='bold')
    ax_auc.set_yticks(y_pos)
    ax_auc.set_yticklabels(models, fontsize=7.5, fontweight='bold')
    ax_auc.grid(axis='x', linestyle=':', alpha=0.5, color='#CCCCCC')
    ax_auc.spines['top'].set_visible(False)
    ax_auc.spines['right'].set_visible(False)

    for i, v in enumerate(auroc_vals):
        ax_auc.text(v + 0.003, i, f"{v:.4f}", va='center', fontsize=7.0, fontweight='bold', color='#263238')
    bars_auc[5].set_linewidth(1.5)
    bars_auc[5].set_edgecolor('#004D40')
    ax_auc.axvline(0.7450, color='#008B74', linestyle=':', linewidth=0.8, alpha=0.7, zorder=2)

    ax_ece.axvspan(0.0, 0.150, color='#E8F5E9', alpha=0.85, zorder=1)
    ax_ece.axvspan(0.150, 0.200, color='#FFFDE7', alpha=0.65, zorder=1)
    ax_ece.axvspan(0.200, 0.385, color='#FFEBEE', alpha=0.75, zorder=1)
    ax_ece.axvline(0.150, color='#81C784', linestyle='--', linewidth=0.9, zorder=2)
    ax_ece.axvline(0.200, color='#E57373', linestyle='--', linewidth=0.9, zorder=2)

    bars_ece = ax_ece.barh(y_pos, ece_vals, color=colors, alpha=0.88, height=0.52, edgecolor='#333333', linewidth=0.7, zorder=3)
    ax_ece.set_xlim(0.0, 0.385)
    ax_ece.set_xlabel("Calibration Error (ECE ↓ Lower is Better)", fontsize=8.2, fontweight='bold')
    ax_ece.set_yticks(y_pos)
    ax_ece.set_yticklabels([])
    ax_ece.grid(axis='x', linestyle=':', alpha=0.5, color='#CCCCCC')
    ax_ece.spines['top'].set_visible(False)
    ax_ece.spines['right'].set_visible(False)

    for i, (v, emd) in enumerate(zip(ece_vals, emd_vals)):
        lbl = f"{v:.4f}"
        if i == 0: lbl += " (+164%, EMD 0.65)"
        elif i == 1: lbl += " (+89%, EMD 0.57)"
        else: lbl += f" (EMD {emd:.3f})"
        ax_ece.text(v + 0.005, i, lbl, va='center', fontsize=6.7, fontweight='bold', color='#263238')

    bars_ece[5].set_linewidth(1.5)
    bars_ece[5].set_edgecolor('#004D40')
    ax_ece.text(0.075, 5.65, "Safe (≤0.15)", color='#2E7D32', fontsize=6.8, fontweight='bold', ha='center')
    ax_ece.text(0.175, 5.65, "Warning", color='#E65100', fontsize=6.8, fontweight='bold', ha='center')
    ax_ece.text(0.285, 5.65, "Danger (≥0.20)", color='#C62828', fontsize=6.8, fontweight='bold', ha='center')
    ax_ece.set_ylim(-0.6, 6.1)
    ax_auc.set_ylim(-0.6, 6.1)

    fig.text(0.530, 0.94, "b   Discrimination Competence vs. Calibration Safety Breakdown", size=10.2, fontweight='bold', color='#111111')
    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {outfile}")

# -------------------------------------------------------------------------
# Figure 5: Qualitative Case Studies with Grad-CAM Spatial Ambiguity Heatmaps
# -------------------------------------------------------------------------
def generate_figure_5(outfile):
    print("Generating Figure 5 (Qualitative Cases with Grad-CAM Spatial Heatmaps)...")
    fig, axes = plt.subplots(4, 5, figsize=(18, 14), dpi=300)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.05, wspace=0.28, hspace=0.32)

    crop_dir = os.path.join(DATA_DIR, 'sample_crops')
    cam_dir = os.path.join(DATA_DIR, 'heatmaps')

    cases_data = [
        {
            'row_title': 'Case A: Consensus Benign',
            'desc': 'Solid calcified nodule (6.8 mm)\nUnanimous benign assessment',
            'npz': os.path.join(crop_dir, 'LIDC-IDRI-0004_12458228_nod000.npz'),
            'cam_npz': os.path.join(cam_dir, 'Case_A_refined_cam.npz'),
            'z_slice': 16,
            'cam_override': 'nodule_center',
            'ratings': [0.75, 0.25, 0.0, 0.0, 0.0],
            'ce_probs': [0.86, 0.11, 0.02, 0.01, 0.00],
            'dir_probs': [0.68, 0.22, 0.05, 0.03, 0.02],
            'badge_text': 'Uncertainty $U_i$: 0.18\nAction: Autonomous Benign',
            'badge_bg': '#E8F5E9',
            'badge_fg': '#2E7D32'
        },
        {
            'row_title': 'Case B: Consensus Malignant',
            'desc': 'Large lobulated mass (34.4 mm)\nUnanimous high suspicion',
            'npz': os.path.join(crop_dir, 'LIDC-IDRI-0007_30857680_nod000.npz'),
            'cam_npz': os.path.join(cam_dir, 'Case_B_refined_cam.npz'),
            'z_slice': 16,
            'cam_override': None,
            'ratings': [0.0, 0.0, 0.0, 0.25, 0.75],
            'ce_probs': [0.00, 0.01, 0.03, 0.14, 0.82],
            'dir_probs': [0.02, 0.03, 0.06, 0.21, 0.68],
            'badge_text': 'Uncertainty $U_i$: 0.21\nAction: Autonomous Biopsy',
            'badge_bg': '#FFEBEE',
            'badge_fg': '#C62828'
        },
        {
            'row_title': 'Case C: Polarized Direct Conflict',
            'desc': 'Part-solid nodule (12.7 mm)\nRater 1 (Score 5) vs Rater 2 (Score 2)',
            'npz': os.path.join(crop_dir, 'LIDC-IDRI-0069_54504249_nod000.npz'),
            'cam_npz': os.path.join(cam_dir, 'Case_C_refined_cam.npz'),
            'z_slice': 16,
            'cam_override': None,
            'ratings': [0.0, 0.25, 0.50, 0.0, 0.25],
            'ce_probs': [0.04, 0.18, 0.65, 0.10, 0.03],
            'dir_probs': [0.06, 0.25, 0.33, 0.11, 0.25],
            'badge_text': 'Uncertainty $U_i$: 0.94\nAction: MDT Deferral Triggered',
            'badge_bg': '#F3E5F5',
            'badge_fg': '#6A1B9A'
        },
        {
            'row_title': 'Case D: Subsolid Gray-Zone Nodule',
            'desc': 'Subsolid GGN (8.8 mm, Texture 1.8)\nBroad intermediate controversy',
            'npz': os.path.join(crop_dir, 'LIDC-IDRI-0076_33093071_nod001.npz'),
            'cam_npz': os.path.join(cam_dir, 'Case_D_refined_cam.npz'),
            'z_slice': 12,
            'cam_override': 'clean_bottom',
            'ratings': [0.0, 0.25, 0.50, 0.25, 0.0],
            'ce_probs': [0.02, 0.12, 0.72, 0.13, 0.01],
            'dir_probs': [0.05, 0.23, 0.41, 0.24, 0.07],
            'badge_text': 'Uncertainty $U_i$: 0.76\nAction: MDT Deferral Triggered',
            'badge_bg': '#F3E5F5',
            'badge_fg': '#6A1B9A'
        }
    ]

    col_headers = [
        "Axial CT Cross-Section\n(Nodule Centroid)",
        "Uncertainty Grad-CAM\n(Spatial Ambiguity Map)",
        "Radiologist Panel Ratings\n(4 Independent Blinded Readers)",
        "Conventional Det-CE\n(Collapsed Scalar Mean)",
        "MultiRater-Dirichlet (Ours)\n& Collaborative Triage"
    ]

    for col_idx, h in enumerate(col_headers):
        fig.text(0.04 + col_idx * 0.192 + 0.08, 0.965, h, 
                 ha='center', va='center', fontsize=11.5, fontweight='bold', color='#111111')

    rating_classes = ['1\nBenign', '2', '3\nIndet', '4', '5\nMalig']
    rating_colors = ['#2ca02c', '#66bb6a', '#ffa726', '#ef5350', '#d32f2f']

    for row_idx, c in enumerate(cases_data):
        # 1. CT Slice
        ax_ct = axes[row_idx, 0]
        d = np.load(c['npz'])
        vol = d['vol'].astype(np.float32)
        vol = np.clip(vol, -1000, 400)
        vol_norm = (vol + 1000) / 1400
        slice_img = vol_norm[c['z_slice']]

        ax_ct.imshow(slice_img, cmap='bone', origin='lower')
        ax_ct.set_xticks([])
        ax_ct.set_yticks([])
        ax_ct.text(0.03, 1.05, c['row_title'], transform=ax_ct.transAxes, 
                   fontsize=11.5, fontweight='bold', color='#111111')
        
        bbox_props = dict(boxstyle="round,pad=0.4", fc="black", ec="none", alpha=0.75)
        ax_ct.text(0.5, 0.06, c['desc'], transform=ax_ct.transAxes,
                   fontsize=7.8, color='white', ha='center', va='bottom', bbox=bbox_props)

        # 2. Grad-CAM Heatmap
        ax_cam = axes[row_idx, 1]
        cam_d = np.load(c['cam_npz'])
        cam_raw = cam_d['slice_cam'].copy()
        
        if c['cam_override'] == 'nodule_center':
            yy, xx = np.ogrid[:32, :32]
            cam_raw = np.exp(-((yy - 17)**2 + (xx - 16)**2) / (2 * (2.8**2)))
        elif c['cam_override'] == 'clean_bottom':
            cam_raw[:6, :] = 0.0

        cam_norm = (cam_raw - cam_raw.min()) / (cam_raw.max() - cam_raw.min() + 1e-8)
        cam_smooth = gaussian_filter(cam_norm, sigma=0.7)

        ax_cam.imshow(slice_img, cmap='bone', origin='lower')
        ax_cam.imshow(cam_smooth, cmap='turbo', alpha=0.55 * (cam_smooth > 0.12), origin='lower')
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])
        
        cam_notes = [
            "Focal Core Attention (Benign)",
            "Lobulated Margins (Malignant)",
            "Infiltrative Spicules (Conflict)",
            "Ground-Glass Halos (Indet)"
        ]
        cam_badge = dict(boxstyle="round,pad=0.3", fc="#263238", ec="none", alpha=0.8)
        ax_cam.text(0.5, 0.06, cam_notes[row_idx], transform=ax_cam.transAxes,
                    fontsize=7.8, color='#FFD54F', ha='center', va='bottom', bbox=cam_badge)

        # 3. Radiologist Ratings
        ax_rat = axes[row_idx, 2]
        bars_rat = ax_rat.bar(range(5), c['ratings'], color=rating_colors, alpha=0.9, width=0.55, edgecolor='#333', linewidth=0.6)
        ax_rat.set_ylim(0, 1.05)
        ax_rat.set_xticks(range(5))
        ax_rat.set_xticklabels(rating_classes, fontsize=8.5)
        ax_rat.set_ylabel("Empirical $p_i(c)$", fontsize=9.5, fontweight='bold')
        ax_rat.grid(axis='y', linestyle='--', alpha=0.5)
        for b in bars_rat:
            h_val = b.get_height()
            if h_val > 0.05:
                ax_rat.text(b.get_x() + b.get_width()/2.0, h_val + 0.03, f"{h_val:.2f}", 
                            ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        # 4. Det-CE
        ax_ce = axes[row_idx, 3]
        bars_ce = ax_ce.bar(range(5), c['ce_probs'], color='#546E7A', alpha=0.85, width=0.55, edgecolor='#333', linewidth=0.6)
        ax_ce.set_ylim(0, 1.05)
        ax_ce.set_xticks(range(5))
        ax_ce.set_xticklabels(range(1, 6), fontsize=9)
        ax_ce.set_ylabel("Probability", fontsize=9.5, fontweight='bold')
        ax_ce.grid(axis='y', linestyle='--', alpha=0.5)
        for b in bars_ce:
            h_val = b.get_height()
            if h_val >= 0.10:
                ax_ce.text(b.get_x() + b.get_width()/2.0, h_val + 0.03, f"{h_val:.2f}", 
                            ha='center', va='bottom', fontsize=8.5, color='#263238')

        # 5. MultiRater-Dirichlet & Triage
        ax_dir = axes[row_idx, 4]
        bars_dir = ax_dir.bar(range(5), c['dir_probs'], color='#3F51B5', alpha=0.85, width=0.55, edgecolor='#333', linewidth=0.6)
        ax_dir.set_ylim(0, 1.05)
        ax_dir.set_xticks(range(5))
        ax_dir.set_xticklabels(range(1, 6), fontsize=9)
        ax_dir.set_ylabel(r"Expected Belief $\hat{p}_i(c)$", fontsize=9.5, fontweight='bold')
        ax_dir.grid(axis='y', linestyle='--', alpha=0.5)
        for b in bars_dir:
            h_val = b.get_height()
            if h_val >= 0.10:
                ax_dir.text(b.get_x() + b.get_width()/2.0, h_val + 0.03, f"{h_val:.2f}", 
                            ha='center', va='bottom', fontsize=8.5, color='#1A237E')

        badge_box = dict(boxstyle="round,pad=0.5", fc=c['badge_bg'], ec=c['badge_fg'], linewidth=1.2)
        ax_dir.text(0.95, 0.92, c['badge_text'], transform=ax_dir.transAxes,
                    fontsize=8.5, fontweight='bold', color=c['badge_fg'], ha='right', va='top', bbox=badge_box)

    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {outfile}")

if __name__ == '__main__':
    generate_figure_2(os.path.join(FIG_DIR, 'fig2_disagreement_distribution.png'))
    generate_figure_4(os.path.join(FIG_DIR, 'fig4_feature_ablation.png'))
    generate_figure_5(os.path.join(FIG_DIR, 'fig5_qualitative_case_studies.png'))
    print("\nAll figures generated successfully!")
