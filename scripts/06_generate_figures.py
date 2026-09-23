import os
import numpy as np
import matplotlib.pyplot as plt

# Nature publication styling
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['text.color'] = '#1A1A1A'
plt.rcParams['figure.autolayout'] = False

def create_figure(outfile):
    fig = plt.figure(figsize=(12.2, 5.0), dpi=300)

    # 1. Panel A: Radar Chart positioned safely to the left
    ax1 = fig.add_axes([0.025, 0.14, 0.32, 0.65], polar=True)

    # 2. Panel B: Two paired horizontal bar charts
    ax_auc = fig.add_axes([0.530, 0.14, 0.190, 0.65])
    ax_ece = fig.add_axes([0.730, 0.14, 0.245, 0.65])

    # =====================================================================
    # PANEL A: Radar Chart
    # =====================================================================
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

    # =====================================================================
    # PANEL B: Paired Horizontal Bar (AUROC vs. Calibration Safety)
    # =====================================================================
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

    # --- SUBPANEL B1: AUROC ---
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

    # Reference baseline dashed line for Full Set
    ax_auc.axvline(0.7450, color='#008B74', linestyle=':', linewidth=0.8, alpha=0.7, zorder=2)

    # --- SUBPANEL B2: ECE with Clinical Safety Risk Zones ---
    ax_ece.axvspan(0.0, 0.150, color='#E8F5E9', alpha=0.85, zorder=1)
    ax_ece.axvspan(0.150, 0.200, color='#FFFDE7', alpha=0.65, zorder=1)
    ax_ece.axvspan(0.200, 0.385, color='#FFEBEE', alpha=0.75, zorder=1)

    ax_ece.axvline(0.150, color='#81C784', linestyle='--', linewidth=0.9, zorder=2)
    ax_ece.axvline(0.200, color='#E57373', linestyle='--', linewidth=0.9, zorder=2)

    bars_ece = ax_ece.barh(y_pos, ece_vals, color=colors, alpha=0.88, height=0.52, edgecolor='#333333', linewidth=0.7, zorder=3)
    ax_ece.set_xlim(0.0, 0.385)
    ax_ece.set_xlabel("Calibration Error (ECE ↓ Lower is Better)", fontsize=8.2, fontweight='bold')
    ax_ece.set_yticks(y_pos)
    ax_ece.set_yticklabels([]) # Shared y-axis
    ax_ece.grid(axis='x', linestyle=':', alpha=0.5, color='#CCCCCC')
    ax_ece.spines['top'].set_visible(False)
    ax_ece.spines['right'].set_visible(False)

    for i, (v, emd) in enumerate(zip(ece_vals, emd_vals)):
        lbl = f"{v:.4f}"
        if i == 0:
            lbl += " (+164%, EMD 0.65)"
        elif i == 1:
            lbl += " (+89%, EMD 0.57)"
        else:
            lbl += f" (EMD {emd:.3f})"
        ax_ece.text(v + 0.005, i, lbl, va='center', fontsize=6.7, fontweight='bold', color='#263238')

    bars_ece[5].set_linewidth(1.5)
    bars_ece[5].set_edgecolor('#004D40')

    # Zone indicators at the top of B2
    ax_ece.text(0.075, 5.65, "Safe (≤0.15)", color='#2E7D32', fontsize=6.8, fontweight='bold', ha='center')
    ax_ece.text(0.175, 5.65, "Warning", color='#E65100', fontsize=6.8, fontweight='bold', ha='center')
    ax_ece.text(0.285, 5.65, "Danger (≥0.20)", color='#C62828', fontsize=6.8, fontweight='bold', ha='center')
    ax_ece.set_ylim(-0.6, 6.1)
    ax_auc.set_ylim(-0.6, 6.1)

    fig.text(0.530, 0.94, "b   Discrimination Competence vs. Calibration Safety Breakdown", size=10.2, fontweight='bold', color='#111111')

    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    out1 = "figures/fig4_feature_ablation.png"
    create_figure(out1)
