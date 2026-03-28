#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════════
  WILDFIRE SPREAD PREDICTION — VISUALISATION PIPELINE                         
  Upper West Ghana · Plots, Maps & Figures                                    
                                                                              
  Requires pipeline_core.py to have been run first in the same session,       
  OR import this module from pipeline_core.py directly.                      
                                                                              
  Usage:  run pipeline_core.py, then call generate_all_figures()             
          OR:  exec(open("pipeline_core.py").read())                         
               exec(open("pipeline_viz.py").read())                           
══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.colors import (LinearSegmentedColormap,
                                BoundaryNorm, ListedColormap, Normalize)
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import (roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score,
                              confusion_matrix)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import os


# ──────────────────────────────────────────────────────────────────────────────
#  COLOUR MAPS
# ──────────────────────────────────────────────────────────────────────────────
FIRE_CMAP = LinearSegmentedColormap.from_list("fire",
    ["#0d0d0d","#6b0000","#cc2200","#ff6600","#ffaa00","#ffee44"])
PROB_CMAP = LinearSegmentedColormap.from_list("prob",
    ["#ffffff","#ffffa0","#ffcc00","#ff6600","#cc0000","#660000"])
NDVI_CMAP = LinearSegmentedColormap.from_list("ndvi",
    ["#8B4513","#D2B48C","#FFFF99","#90EE90","#228B22","#006400"])
ELEV_CMAP = LinearSegmentedColormap.from_list("elev",
    ["#2E8B57","#8FBC8F","#F4A460","#CD853F","#A0522D","#696969"])
RISK_CMAP = LinearSegmentedColormap.from_list("risk",
    ["#1a9850","#91cf60","#d9ef8b","#fee08b","#fc8d59","#d73027"])


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def savefig(subdir_key, filename, dpi=180):
    path = os.path.join(SUBDIRS[subdir_key], filename)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"    ✓ {path}")


def savefig_root(filename, dpi=180):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"    ✓ {path}")


def valid_pairs(lst):
    return [(i + 1, v) for i, v in enumerate(lst) if not np.isnan(v)]


def _sm(arr, s=3.0):
    a = arr.copy()
    a[np.isnan(a)] = np.nanmean(a) if not np.all(np.isnan(a)) else 0.0
    return gaussian_filter(a, sigma=s)


def norm01(a):
    mn, mx = np.nanmin(a), np.nanmax(a)
    return (a - mn) / (mx - mn + 1e-9)


def add_aoi_border(ax, lw=1.5, color="#333333"):
    from matplotlib.patches import Rectangle
    rect = Rectangle((EXT[0], EXT[2]), EXT[1]-EXT[0], EXT[3]-EXT[2],
                     linewidth=lw, edgecolor=color, facecolor="none",
                     transform=ax.transData, zorder=5)
    ax.add_patch(rect)


def grid_ticks(ax):
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.tick_params(labelsize=8)
    lon_range = EXT[1] - EXT[0]
    lat_range = EXT[3] - EXT[2]
    step = max(0.5, round(max(lon_range, lat_range) / 4, 1))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN FIGURE GENERATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  STAGE 8 · GENERATING ALL FIGURES")
print("═" * 70)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.linewidth":   0.8,
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
})

pred_s     = clstm_preds[-1]
true_s     = Y_test[-1]
cmap_ph    = plt.cm.plasma(np.linspace(0.1, 0.9, PRED_HORIZON))
horizons_x = list(range(1, PRED_HORIZON + 1))

# Build time-averaged spatial maps
feat_mean = {f: np.full((H, W), np.nan) for f in FEATURES}
fire_map   = np.zeros((H, W))
spread_map = np.zeros((H, W))

for feat in FEATURES:
    grp = df_raw.groupby(["lat", "lon"])[feat].mean()
    for (la, lo), val in grp.items():
        if la in lat_to_idx and lo in lon_to_idx:
            feat_mean[feat][lat_to_idx[la], lon_to_idx[lo]] = val

for (la, lo), val in df_raw.groupby(["lat","lon"])["fire_t"].mean().items():
    if la in lat_to_idx and lo in lon_to_idx:
        fire_map[lat_to_idx[la], lon_to_idx[lo]] = val

for (la, lo), val in df_raw.groupby(["lat","lon"])["spread"].mean().items():
    if la in lat_to_idx and lo in lon_to_idx:
        spread_map[lat_to_idx[la], lon_to_idx[lo]] = val


# ══════════════════════════════════════════════════════════════════════════════
#  01_study_area/
# ══════════════════════════════════════════════════════════════════════════════
print("\n  ── 01_study_area/ ──")

fig = plt.figure(figsize=(16, 13))
fig.suptitle("Area of Interest — Upper West Ghana\nSpatial Overview",
             fontsize=15, fontweight="bold", y=0.98)
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
panels = [
    (gs[0, 0], fire_map,               FIRE_CMAP, "Mean Fire Frequency",    "P(fire)"),
    (gs[0, 1], feat_mean["ndvi"],      NDVI_CMAP, "Mean NDVI",              "NDVI"),
    (gs[1, 0], feat_mean["elevation"], ELEV_CMAP, "Elevation",              "m a.s.l."),
    (gs[1, 1], spread_map,             PROB_CMAP, "Mean Spread Probability","P(spread)"),
]
for spec, data, cmap, title, cblabel in panels:
    ax = fig.add_subplot(spec)
    d  = _sm(data, s=0.5)
    im = ax.imshow(d, cmap=cmap, origin="lower", extent=EXT, aspect="auto")
    add_aoi_border(ax); grid_ticks(ax)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cblabel, fontsize=8); cb.ax.tick_params(labelsize=7)
    peak = np.unravel_index(np.nanargmax(d), d.shape)
    ax.plot(lons[peak[1]], lats[peak[0]], "w*", ms=10, zorder=6,
            markeredgecolor="black", markeredgewidth=0.5)
fig.text(0.5, 0.01,
         "★ = peak value cell    |    Gaussian-smoothed for display  (σ = 0.5 cells)",
         ha="center", fontsize=8, color="#555555")
savefig("study_area", "aoi_overview.png")

fire_lons = df_raw.loc[df_raw["fire_t"] == 1, "lon"].values
fire_lats = df_raw.loc[df_raw["fire_t"] == 1, "lat"].values
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Fire Occurrence Density — Kernel Density Estimation", fontsize=13, fontweight="bold")
if len(fire_lons) >= 5:
    lon_grid = np.linspace(EXT[0], EXT[1], 80)
    lat_grid = np.linspace(EXT[2], EXT[3], 80)
    LG, LtG  = np.meshgrid(lon_grid, lat_grid)
    kernel   = gaussian_kde(np.vstack([fire_lons, fire_lats]))
    kde_vals = kernel(np.vstack([LG.ravel(), LtG.ravel()])).reshape(80, 80)
    kde_vals = (kde_vals - kde_vals.min()) / (kde_vals.max() - kde_vals.min() + 1e-9)
    im0 = axes[0].imshow(kde_vals, cmap=FIRE_CMAP, origin="lower",
                          extent=EXT, aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("Fire KDE Density Surface", fontweight="bold")
    grid_ticks(axes[0]); add_aoi_border(axes[0])
    plt.colorbar(im0, ax=axes[0], label="Normalised Density")
    axes[1].imshow(_sm(fire_map), cmap="Greys", origin="lower",
                   extent=EXT, aspect="auto", alpha=0.4)
    cs = axes[1].contourf(LG, LtG, kde_vals, levels=10, cmap=FIRE_CMAP, alpha=0.75)
    axes[1].contour(LG, LtG, kde_vals, levels=[0.25, 0.5, 0.75],
                    colors="white", linewidths=0.8, alpha=0.7)
    axes[1].set_title("KDE Contours + Fire Grid", fontweight="bold")
    grid_ticks(axes[1]); add_aoi_border(axes[1])
    plt.colorbar(cs, ax=axes[1], label="Normalised Density")
else:
    for ax in axes:
        ax.text(0.5, 0.5, "Insufficient fire records for KDE",
                ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
savefig("study_area", "fire_density.png")

years    = sorted(df_raw["year"].unique())
calendar = {}
for y in years:
    calendar[y] = [df_raw[(df_raw["year"]==y) & (df_raw["month"]==m)]["fire_t"].mean()
                   for m in range(1, 13)]
import pandas as pd
calendar = pd.DataFrame(calendar, index=MONTH_NAMES)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Fire Activity Calendar — Monthly × Yearly", fontsize=13, fontweight="bold")
sns.heatmap(calendar, cmap=FIRE_CMAP, ax=axes[0], annot=True, fmt=".2f",
            annot_kws={"size": 8}, linewidths=0.4,
            cbar_kws={"label": "Mean Fire Rate"}, vmin=0,
            vmax=max(calendar.values.max(), 1e-6))
axes[0].set_title("Monthly Fire Rate Heatmap", fontweight="bold")
axes[0].set_xlabel("Year"); axes[0].set_ylabel("Month"); axes[0].tick_params(labelsize=8)
monthly_avg = [df_raw[df_raw["month"]==m]["fire_t"].mean() for m in range(1,13)]
monthly_std = [df_raw[df_raw["month"]==m]["fire_t"].std()  for m in range(1,13)]
max_avg     = max(monthly_avg + [1e-9])
axes[1].bar(range(1,13), monthly_avg,
            color=[FIRE_CMAP(v/max_avg) for v in monthly_avg], alpha=0.85)
axes[1].errorbar(range(1,13), monthly_avg, yerr=monthly_std, fmt="none",
                  ecolor="black", capsize=4, lw=1.2)
axes[1].set_xticks(range(1,13)); axes[1].set_xticklabels(MONTH_NAMES, rotation=45, fontsize=8)
axes[1].set_title("Mean Monthly Fire Rate ± σ", fontweight="bold")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Mean Fire Rate"); axes[1].grid(alpha=0.25, axis="y")
plt.tight_layout()
savefig("study_area", "monthly_calendar.png")


# ══════════════════════════════════════════════════════════════════════════════
#  02_model_training/
# ══════════════════════════════════════════════════════════════════════════════
print("\n  ── 02_model_training/ ──")

fig, ax = plt.subplots(figsize=(9, 4))
ep_x = list(range(1, len(train_losses) + 1))
ax.plot(ep_x, train_losses, lw=2.5, color="#e63946", label=f"Train loss (fold {final['fold']})")
ax.plot(ep_x, val_losses,   lw=2.5, color="#457b9d", ls="--", label=f"Val loss (fold {final['fold']})")
ax.fill_between(ep_x, train_losses, alpha=0.15, color="#e63946")
ax.fill_between(ep_x, val_losses,   alpha=0.15, color="#457b9d")
ax.set_xlabel("Epoch"); ax.set_ylabel("Focal Loss")
ax.set_title(f"ConvLSTM — Training & Validation Loss  "
             f"(focal α={FOCAL_ALPHA}, final CV fold)", fontsize=14, fontweight="bold")
ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout()
savefig("model_training", "loss_curves.png")

Y_tr_t_fin = final["_Y_tr_t"]
Y_sm_fin   = final["_Y_tr_smote"]
if SMOTE_AVAILABLE and len(np.unique(Y_tr_t_fin)) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Class Distribution Before & After SMOTE (final fold)",
                 fontsize=13, fontweight="bold")
    for ax_i, (labels, title) in enumerate([
        (Y_tr_t_fin, "Before SMOTE"),
        (Y_sm_fin,   "After SMOTE"),
    ]):
        counts = np.bincount(labels)
        bars   = axes[ax_i].bar(["No Spread (0)", "Spread (1)"],
                                 counts, color=["#457b9d", "#e63946"],
                                 alpha=0.85, edgecolor="white")
        for bar, cnt in zip(bars, counts):
            axes[ax_i].text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + max(counts)*0.01,
                            str(cnt), ha="center", va="bottom", fontweight="bold")
        axes[ax_i].set_title(title, fontweight="bold")
        axes[ax_i].set_ylabel("Count"); axes[ax_i].grid(alpha=0.25, axis="y")
    plt.tight_layout()
    savefig("model_training", "smote_balance.png")


# ══════════════════════════════════════════════════════════════════════════════
#  03_model_evaluation/
# ══════════════════════════════════════════════════════════════════════════════
print("\n  ── 03_model_evaluation/ ──")

fig, ax = plt.subplots(figsize=(7, 7))
for h in range(PRED_HORIZON):
    y_t = (Y_test[:,h].reshape(len(X_test),-1).mean(axis=-1) > TAB_THRESH).astype(int)
    y_p = np.clip(minmax_scale(clstm_preds[:,h].reshape(len(X_test),-1).mean(axis=-1)), 0.0, 1.0)
    if len(np.unique(y_t)) < 2: continue
    fpr, tpr, _ = roc_curve(y_t, y_p)
    ax.plot(fpr, tpr, color=cmap_ph[h], lw=2,
            label=f"Day+{h+1} (AUC={roc_auc_score(y_t, y_p):.3f})")
ax.plot([0,1],[0,1],"--",color="gray")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (final CV fold)", fontsize=14, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
savefig("model_evaluation", "roc.png")

fig, ax = plt.subplots(figsize=(7, 7))
for h in range(PRED_HORIZON):
    y_t = (Y_test[:,h].reshape(len(X_test),-1).mean(axis=-1) > TAB_THRESH).astype(int)
    y_p = np.clip(minmax_scale(clstm_preds[:,h].reshape(len(X_test),-1).mean(axis=-1)), 0.0, 1.0)
    if len(np.unique(y_t)) < 2: continue
    prec, rec, _ = precision_recall_curve(y_t, y_p)
    ap = average_precision_score(y_t, y_p)
    ax.plot(rec, prec, color=cmap_ph[h], lw=2, label=f"Day+{h+1} (AP={ap:.3f})")
baseline = Y_te_t.mean() if len(Y_te_t) > 0 else 0.0
ax.axhline(baseline, ls="--", color="gray", label=f"Baseline (prevalence={baseline:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves  [PRIMARY metric]", fontsize=14, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
savefig("model_evaluation", "pr.png")

y_t_d1 = (Y_test[:,0].reshape(len(X_test),-1).mean(axis=-1) > TAB_THRESH).astype(int)
y_p_d1 = np.clip(minmax_scale(clstm_preds[:,0].reshape(len(X_test),-1).mean(axis=-1)), 0.0, 1.0)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Probability Calibration — Day+1 (final CV fold)", fontsize=13, fontweight="bold")
if len(np.unique(y_t_d1)) >= 2:
    n_bins = min(10, len(y_t_d1)//2)
    pt_, pp_ = calibration_curve(y_t_d1, y_p_d1, n_bins=n_bins)
    axes[0].plot(pp_, pt_, marker="o", lw=2.5, color="#e63946", ms=8)
    axes[0].plot([0,1],[0,1],"--",color="gray")
    axes[0].set_xlabel("Mean Predicted Probability"); axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Reliability Diagram"); axes[0].grid(alpha=0.3)
axes[1].hist(y_p_d1[y_t_d1==0], bins=30, alpha=0.65, color="#457b9d",
             label="No Spread", density=True)
axes[1].hist(y_p_d1[y_t_d1==1], bins=30, alpha=0.65, color="#e63946",
             label="Spread", density=True)
axes[1].axvline(OPT_THRESH_ENS, ls="--", color="black",
                label=f"Opt. threshold={OPT_THRESH_ENS:.2f}")
axes[1].set_xlabel("Predicted Probability"); axes[1].set_ylabel("Density")
axes[1].set_title("Score Distribution"); axes[1].legend(); axes[1].grid(alpha=0.25)
plt.tight_layout()
savefig("model_evaluation", "calibration.png")

y_pred_bin = (ensemble_p >= OPT_THRESH_ENS).astype(int)
cm = confusion_matrix(Y_te_t, y_pred_bin)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(f"Ensemble Confusion Matrix  (thr={OPT_THRESH_ENS:.2f}, final CV fold)",
             fontsize=13, fontweight="bold")
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Spread","Spread"], yticklabels=["No Spread","Spread"],
            ax=axes[0], linewidths=0.5)
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title(f"Confusion Matrix (thr={OPT_THRESH_ENS:.2f})")
cm_norm = cm.astype(float) / (cm.sum(axis=1,keepdims=True) + 1e-9)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["No Spread","Spread"], yticklabels=["No Spread","Spread"],
            ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
axes[1].set_title("Normalised Confusion Matrix")
plt.tight_layout()
savefig("model_evaluation", "confusion_matrix.png")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Forecast Performance Across Horizons  "
             "[F1 & AP = primary; AUC for reference]", fontsize=13, fontweight="bold")
for ax, (label, data, color, marker, pos) in zip(
    axes.flat,
    [("F1 (opt thr)",          f1_h,    "#4caf50", "s", (0,0)),
     ("Average Precision (AP)",ap_h,    "#ff9800", "^", (0,1)),
     ("ROC-AUC (reference)",   auc_h,   "#2196f3", "o", (1,0)),
     ("Brier Score",           brier_h, "#e63946", "D", (1,1))]):
    vp = valid_pairs(data)
    if vp:
        hx, hy = zip(*vp)
        if label == "Brier Score":
            ax.bar(hx, hy, color=color, alpha=0.8)
        else:
            ax.plot(hx, hy, marker=marker, lw=2.5, color=color, ms=8)
    if label != "Brier Score":
        ax.set_ylim(0 if label != "ROC-AUC (reference)" else 0.3, 1.1)
        if label == "ROC-AUC (reference)":
            ax.axhline(0.5, ls="--", color="gray")
    ax.set_title(label, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (days)"); ax.grid(alpha=0.3)
plt.tight_layout()
savefig("model_evaluation", "horizon_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
#  07_cv_results/
# ══════════════════════════════════════════════════════════════════════════════
print("\n  ── 07_cv_results/ ──")

fold_ids = [r["fold"] for r in cv_records]

# ── cv_fold_metrics.png ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"K-Fold Temporal CV — Per-Fold Metrics  (Day+1, {N_FOLDS} folds)",
             fontsize=13, fontweight="bold")
for ax, (metric_label, vals, color) in zip(axes, [
    ("AUC", [r["horizon_auc"][0] for r in cv_records], "#2196f3"),
    ("F1",  [r["horizon_f1"][0]  for r in cv_records], "#4caf50"),
    ("AP",  [r["horizon_ap"][0]  for r in cv_records], "#ff9800"),
]):
    clean = [v if not np.isnan(v) else 0.0 for v in vals]
    bars  = ax.bar(fold_ids, clean, color=color, alpha=0.8, edgecolor="white")
    mean_v = np.nanmean(vals)
    ax.axhline(mean_v, ls="--", color="black", lw=1.5, label=f"Mean = {mean_v:.3f}")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}" if not np.isnan(v) else "n/a",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_xlabel("Fold"); ax.set_ylabel(metric_label)
    ax.set_title(f"Day+1  {metric_label}  per Fold", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y"); ax.set_xticks(fold_ids)
plt.tight_layout()
savefig("cv_results", "cv_fold_metrics.png")

# ── cv_horizon_bands.png ──────────────────────────────────────────────────────
cv_f1_mat  = np.array([r["horizon_f1"]  for r in cv_records])
cv_auc_mat = np.array([r["horizon_auc"] for r in cv_records])
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle("CV Mean ± Std Across Forecast Horizons", fontsize=13, fontweight="bold")
hx = np.arange(1, PRED_HORIZON + 1)
for ax, (mat, label, color) in zip(axes, [
    (cv_f1_mat,  "F1  (primary)",        "#4caf50"),
    (cv_auc_mat, "ROC-AUC  (reference)", "#2196f3"),
]):
    means = np.nanmean(mat, axis=0)
    stds  = np.nanstd(mat,  axis=0)
    ax.plot(hx, means, marker="o", lw=2.5, color=color, ms=8)
    ax.fill_between(hx, means - stds, means + stds, alpha=0.25, color=color)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Forecast Horizon (days)"); ax.set_ylabel(label)
    ax.set_title(f"{label} — Mean ± 1σ  across {len(cv_records)} folds",
                 fontweight="bold")
    ax.grid(alpha=0.3); ax.set_xticks(hx)
plt.tight_layout()
savefig("cv_results", "cv_horizon_bands.png")

# ── cv_train_sizes.png ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
tr_sizes = [r["train_size"] for r in cv_records]
te_sizes = [r["test_size"]  for r in cv_records]
ax2 = ax.twinx()
ax.bar(fold_ids, tr_sizes, alpha=0.6, color="#457b9d", label="Train seqs")
ax.bar(fold_ids, te_sizes, alpha=0.6, color="#e63946",
       label="Test seqs", bottom=tr_sizes)
ax2.plot(fold_ids, [r["ensemble_auc"] for r in cv_records],
         marker="D", color="#ff9800", lw=2, ms=8, label="Ensemble AUC")
ax.set_xlabel("Fold"); ax.set_ylabel("Sequence count")
ax2.set_ylabel("Ensemble AUC (Day+1)"); ax2.set_ylim(0, 1.1)
ax.set_title("Train / Test Sizes & Ensemble AUC per Fold", fontweight="bold")
lines1, l1 = ax.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, l1 + l2, fontsize=9)
ax.grid(alpha=0.2, axis="y"); ax.set_xticks(fold_ids)
plt.tight_layout()
savefig("cv_results", "cv_train_sizes.png")


# ══════════════════════════════════════════════════════════════════════════════
#  04_spatial_forecast/
# ══════════════════════════════════════════════════════════════════════════════
print("\n  ── 04_spatial_forecast/ ──")

fig = plt.figure(figsize=(PRED_HORIZON * 3.5, 10))
fig.suptitle("Multi-Horizon Fire Spread Forecast — Upper West Ghana",
             fontsize=12, fontweight="bold", y=1.01)
for h in range(PRED_HORIZON):
    ax1 = fig.add_subplot(3, PRED_HORIZON, h + 1)
    im1 = ax1.imshow(pred_s[h], cmap=PROB_CMAP, vmin=0, vmax=1,
                     origin="lower", extent=EXT, aspect="auto")
    ax1.set_title(f"Day+{h+1}", fontsize=10, fontweight="bold"); ax1.set_xlabel("Lon")
    (ax1.set_ylabel("Lat") if h == 0 else ax1.set_yticks([]))
    plt.colorbar(im1, ax=ax1, fraction=0.05, pad=0.03)

    ax2 = fig.add_subplot(3, PRED_HORIZON, PRED_HORIZON + h + 1)
    im2 = ax2.imshow(true_s[h], cmap="Reds", vmin=0, vmax=1,
                     origin="lower", extent=EXT, aspect="auto")
    ax2.set_xlabel("Lon")
    (ax2.set_ylabel("Lat") if h == 0 else ax2.set_yticks([]))
    plt.colorbar(im2, ax=ax2, fraction=0.05, pad=0.03)

    ax3 = fig.add_subplot(3, PRED_HORIZON, 2*PRED_HORIZON + h + 1)
    im3 = ax3.imshow(np.abs(pred_s[h] - true_s[h]), cmap="YlOrRd",
                     vmin=0, vmax=1, origin="lower", extent=EXT, aspect="auto")
    ax3.set_xlabel("Lon")
    (ax3.set_ylabel("Lat") if h == 0 else ax3.set_yticks([]))
    plt.colorbar(im3, ax=ax3, fraction=0.05, pad=0.03)
plt.tight_layout()
savefig("spatial_forecast", "multihorizon_maps.png")

mae_map  = np.abs(pred_s - true_s).mean(axis=0)
bias_map = (pred_s - true_s).mean(axis=0)
hotspot  = np.where(mae_map > np.percentile(mae_map, 75), mae_map, np.nan)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Spatial Prediction Error Analysis", fontsize=13, fontweight="bold")
im0 = axes[0].imshow(mae_map,  cmap="inferno", origin="lower", extent=EXT, aspect="auto")
axes[0].set_title("MAE"); plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(bias_map, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                     origin="lower", extent=EXT, aspect="auto")
axes[1].set_title("Bias"); plt.colorbar(im1, ax=axes[1])
im2 = axes[2].imshow(hotspot,  cmap="hot", origin="lower", extent=EXT, aspect="auto")
axes[2].set_title("High-Error Hotspots"); plt.colorbar(im2, ax=axes[2])
plt.tight_layout()
savefig("spatial_forecast", "error_maps.png")


# ══════════════════════════════════════════════════════════════════════════════
#  05_interpretation/
# ══════════════════════════════════════════════════════════════════════════════
print("\n  ── 05_interpretation/ ──")
print("    Computing feature importance …")


def fl_mean(X):
    return X.mean(axis=(1, 3, 4))


X_tr_fi = X_all[: cv_records[-1]["train_size"]]
Y_tr_fi = Y_all[: cv_records[-1]["train_size"]]
y_fi_tr = (Y_tr_fi[:,0].reshape(len(X_tr_fi),-1).mean(axis=-1) > TAB_THRESH).astype(int)
y_fi    = (Y_test[:,0].reshape(len(X_test),-1).mean(axis=-1)    > TAB_THRESH).astype(int)

lr_fi = LogisticRegression(max_iter=500, C=0.5, class_weight="balanced", random_state=42)
lr_fi.fit(fl_mean(X_tr_fi), y_fi_tr)
base_auc_fi = (roc_auc_score(y_fi, lr_fi.predict_proba(fl_mean(X_test))[:,1])
               if len(np.unique(y_fi)) >= 2 else 0.5)

perm_imps = []
for fi in range(C):
    Xp = X_test.copy()
    Xp[:,:,fi] = X_test[np.random.permutation(len(X_test)),:,fi]
    drop_auc = (roc_auc_score(y_fi, lr_fi.predict_proba(fl_mean(Xp))[:,1])
                if len(np.unique(y_fi)) >= 2 else 0.5)
    perm_imps.append(base_auc_fi - drop_auc)

ord_p = np.argsort(perm_imps)
ord_g = np.argsort(gb.feature_importances_)
ord_r = np.argsort(rf.feature_importances_)
fl_nm = lambda idxs: [FEATURE_LABELS[FEATURES[i]] for i in idxs]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Feature Importance — Three Models", fontsize=13, fontweight="bold")
axes[0].barh(fl_nm(ord_p), [perm_imps[i] for i in ord_p],
             color=["#e63946" if perm_imps[i]>0 else "#457b9d" for i in ord_p])
axes[0].set_title("Permutation (ConvLSTM+LR)"); axes[0].set_xlabel("ΔAUC")
axes[0].grid(alpha=0.25, axis="x")
axes[1].barh(fl_nm(ord_g), gb.feature_importances_[ord_g], color="#ff9800")
axes[1].set_title("Gradient Boosting  (SMOTE + balanced weights)")
axes[1].grid(alpha=0.25, axis="x")
axes[2].barh(fl_nm(ord_r), rf.feature_importances_[ord_r], color="#4caf50")
axes[2].set_title("Random Forest  (balanced weights)")
axes[2].grid(alpha=0.25, axis="x")
plt.tight_layout()
savefig("interpretation", "feature_importance.png")

wi  = FEATURES.index("wind_speed")
wdi = FEATURES.index("wind_dir")
ws_raw = X_test[-1,-1,wi]  * scaler.scale_[wi]  + scaler.mean_[wi]
wd_raw = X_test[-1,-1,wdi] * scaler.scale_[wdi] + scaler.mean_[wdi]
U = ws_raw * np.cos(np.radians(wd_raw))
V = ws_raw * np.sin(np.radians(wd_raw))
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(pred_s[0], cmap=FIRE_CMAP, vmin=0, vmax=1,
               origin="lower", extent=EXT, aspect="auto")
step = max(1, H // 8)
ax.quiver(LON2D[::step,::step], LAT2D[::step,::step],
          U[::step,::step], V[::step,::step],
          color="cyan", alpha=0.85, scale=60, width=0.003)
ax.set_title("Day+1 Spread + Wind Field", fontsize=12, fontweight="bold")
ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
plt.colorbar(im, ax=ax, label="P(spread)")
plt.tight_layout()
savefig("interpretation", "wind_alignment.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Weather Drivers of Fire Spread", fontsize=13, fontweight="bold")
for col, ax, color, label in [
    ("temp",              axes[0], "#e63946", "Temperature (°C)"),
    ("relative_humidity", axes[1], "#457b9d", "Rel. Humidity (%)"),
]:
    x_vals = df_raw[col].values; y_vals = df_raw["spread"].values
    ax.scatter(x_vals, y_vals, alpha=0.3, s=6, color=color, rasterized=True)
    ok = ~np.isnan(x_vals) & ~np.isnan(y_vals)
    if ok.sum() > 2:
        z = np.polyfit(x_vals[ok], y_vals[ok], 1)
        xline = np.linspace(x_vals[ok].min(), x_vals[ok].max(), 100)
        ax.plot(xline, np.polyval(z, xline), lw=2, color="black", ls="--")
    ax.set_xlabel(label); ax.set_ylabel("Spread (0/1)")
    ax.set_title(f"{label} vs Spread"); ax.grid(alpha=0.25)
plt.tight_layout()
savefig("interpretation", "weather_vs_spread.png")


# ══════════════════════════════════════════════════════════════════════════════
#  06_application/
# ══════════════════════════════════════════════════════════════════════════════
print("\n  ── 06_application/ ──")

cumulative = gaussian_filter(pred_s.sum(axis=0), 0.8)
fig, axes  = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Cumulative Fire Spread Risk — 7-Day Horizon", fontsize=13, fontweight="bold")
im0 = axes[0].imshow(cumulative, cmap=FIRE_CMAP, origin="lower",
                     extent=EXT, aspect="auto")
axes[0].set_title("Cumulative Burn Probability")
axes[0].set_xlabel("Lon"); axes[0].set_ylabel("Lat")
plt.colorbar(im0, ax=axes[0], label="Σ P(spread)")
bounds = [0, 1.5, 3.0, 4.5, 7.0]
c4     = ["#4caf50", "#ffeb3b", "#ff9800", "#f44336"]
im1    = axes[1].imshow(pred_s.sum(axis=0), cmap=ListedColormap(c4),
                        norm=BoundaryNorm(bounds, len(c4)),
                        origin="lower", extent=EXT, aspect="auto")
axes[1].set_title("Risk Classification"); axes[1].set_xlabel("Lon")
axes[1].legend(handles=[mpatches.Patch(color=c, label=l)
               for c, l in zip(c4, ["Low","Moderate","High","Extreme"])],
               loc="lower right")
plt.colorbar(im1, ax=axes[1], ticks=bounds)
plt.tight_layout()
savefig("application", "cumulative_burn.png")

temp_idx  = norm01(feat_mean["temp"])
sm_idx    = 1.0 - norm01(feat_mean["soil_moisture"])
ndvi_idx  = 1.0 - norm01(feat_mean["ndvi"])
ws_idx    = norm01(feat_mean["wind_speed"])
slope_idx = norm01(feat_mean["slope"])
weights = {"temp":0.30,"dryness":0.25,"ndvi":0.20,"wind":0.15,"slope":0.10}
risk_index = (weights["temp"]*temp_idx + weights["dryness"]*sm_idx +
              weights["ndvi"]*ndvi_idx + weights["wind"]*ws_idx +
              weights["slope"]*slope_idx)
risk_classes = np.digitize(risk_index, bins=[0.2,0.4,0.6,0.8]).clip(0,4)
risk_labels  = ["Very Low","Low","Moderate","High","Very High"]
risk_colors  = ["#1a9850","#91cf60","#fee08b","#fc8d59","#d73027"]

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle("Composite Fire Risk Index — Upper West Ghana AOI", fontsize=13, fontweight="bold")
im0 = axes[0].imshow(_sm(risk_index,s=1.0), cmap=RISK_CMAP, origin="lower",
                      extent=EXT, aspect="auto", vmin=0, vmax=1)
axes[0].set_title("Continuous Risk Index\n" +
                   " + ".join(f"{int(v*100)}%·{k}" for k,v in weights.items()),
                   fontweight="bold", fontsize=9)
grid_ticks(axes[0]); add_aoi_border(axes[0])
plt.colorbar(im0, ax=axes[0], label="Risk Index [0–1]")
im1 = axes[1].imshow(risk_classes, cmap=ListedColormap(risk_colors),
                      origin="lower", extent=EXT, aspect="auto", interpolation="nearest")
axes[1].set_title("Classified Risk Zones", fontweight="bold")
grid_ticks(axes[1]); add_aoi_border(axes[1])
axes[1].legend(handles=[mpatches.Patch(color=risk_colors[i], label=risk_labels[i])
               for i in range(5)], loc="lower right", fontsize=8, framealpha=0.9)
plt.colorbar(im1, ax=axes[1], ticks=[0.5,1.5,2.5,3.5,4.5], boundaries=np.arange(0,5.1))
risk_area_pct = [(risk_classes==k).sum()/risk_classes.size*100 for k in range(5)]
for k,(pct,label,color) in enumerate(zip(risk_area_pct,risk_labels,risk_colors)):
    axes[1].text(EXT[1]+0.05, EXT[3]-0.05-k*0.18, f"{label}: {pct:.0f}%",
                 fontsize=7, color=color, transform=axes[1].transData, clip_on=False,
                 fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
plt.tight_layout()
savefig("application", "risk_composite.png")

temp_norm = norm01(feat_mean["temp"])
composite = 0.5*fire_map + 0.3*spread_map + 0.2*temp_norm
flat_idx  = np.argsort(composite.ravel())[::-1]
top_n     = min(20, len(flat_idx))
top_rows  = [np.unravel_index(i,(H,W)) for i in flat_idx[:top_n]]
top_lats  = [lats[r] for r,c in top_rows]; top_lons = [lons[c] for r,c in top_rows]
top_scores= [composite[r,c] for r,c in top_rows]
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Top Fire Hotspot Ranking — Composite Risk Score", fontsize=13, fontweight="bold")
im = axes[0].imshow(_sm(composite), cmap=RISK_CMAP, origin="lower",
                    extent=EXT, aspect="auto", vmin=0, vmax=1)
axes[0].scatter(top_lons, top_lats, c=top_scores, cmap=FIRE_CMAP,
                s=90, zorder=6, edgecolors="white", linewidths=0.7)
for rank,(la,lo) in enumerate(zip(top_lats[:5],top_lons[:5])):
    axes[0].annotate(f"#{rank+1}",(lo,la),xytext=(4,4),textcoords="offset points",
                     fontsize=7,color="white",fontweight="bold",
                     path_effects=[pe.withStroke(linewidth=1.5,foreground="black")])
axes[0].set_title("Composite Risk Surface + Top Cells", fontweight="bold")
grid_ticks(axes[0]); add_aoi_border(axes[0])
plt.colorbar(im, ax=axes[0], label="Composite Risk [0–1]")
bar_labels = [f"#{i+1} ({top_lats[i]:.2f}°N, {top_lons[i]:.2f}°E)" for i in range(top_n)]
axes[1].barh(bar_labels[::-1], top_scores[::-1],
             color=[FIRE_CMAP(s) for s in top_scores[::-1]], alpha=0.9)
axes[1].set_xlabel("Composite Risk Score")
axes[1].set_title("Ranked Hotspots", fontweight="bold")
axes[1].grid(alpha=0.25, axis="x"); axes[1].tick_params(labelsize=7)
plt.tight_layout()
savefig("application", "hotspot_ranking.png")

exp_flat  = fire_map.ravel(); sen_flat = sm_idx.ravel(); risk_flat = risk_index.ravel()
mask = ~np.isnan(exp_flat) & ~np.isnan(sen_flat) & ~np.isnan(risk_flat)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Vulnerability Quadrant Analysis — Exposure vs Sensitivity",
             fontsize=13, fontweight="bold")
sc = axes[0].scatter(sen_flat[mask], exp_flat[mask], c=risk_flat[mask],
                      cmap=RISK_CMAP, s=40, alpha=0.65, edgecolors="none")
med_sen = np.nanmedian(sen_flat[mask]); med_exp = np.nanmedian(exp_flat[mask])
axes[0].axhline(med_exp,ls="--",color="gray",lw=1.2)
axes[0].axvline(med_sen,ls="--",color="gray",lw=1.2)
for txt,xp,yp in [("Low Vuln.",0.25,0.25),("High Exposure",0.25,0.75),
                   ("High Vuln.",0.75,0.75),("High Sensitivity",0.75,0.25)]:
    axes[0].text(xp,yp,txt,transform=axes[0].transAxes,ha="center",va="center",
                 fontsize=8,color="#555555",style="italic",
                 path_effects=[pe.withStroke(linewidth=2,foreground="white")])
axes[0].set_xlabel("Sensitivity (Dryness Index)",fontsize=10)
axes[0].set_ylabel("Exposure (Mean Fire Frequency)",fontsize=10)
axes[0].set_title("Pixel-Level Vulnerability Quadrant",fontweight="bold")
plt.colorbar(sc, ax=axes[0], label="Composite Risk"); axes[0].grid(alpha=0.2)

quad_map = np.zeros((H,W),dtype=int)
for i in range(H):
    for j in range(W):
        high_e = fire_map[i,j] >= med_exp; high_s = sm_idx[i,j] >= med_sen
        if high_e and high_s:   quad_map[i,j] = 3
        elif high_e:             quad_map[i,j] = 2
        elif high_s:             quad_map[i,j] = 1
q_colors = ["#1a9850","#fee08b","#fc8d59","#d73027"]
q_labels = ["Low Vuln.","High Sensitivity","High Exposure","High Vuln."]
im2 = axes[1].imshow(quad_map,cmap=ListedColormap(q_colors),origin="lower",
                      extent=EXT,aspect="auto",interpolation="nearest",vmin=-0.5,vmax=3.5)
axes[1].set_title("Vulnerability Class Map",fontweight="bold")
grid_ticks(axes[1]); add_aoi_border(axes[1])
axes[1].legend(handles=[mpatches.Patch(color=q_colors[i],label=q_labels[i]) for i in range(4)],
               loc="lower right",fontsize=8,framealpha=0.9)
plt.colorbar(im2, ax=axes[1], ticks=[0,1,2,3], boundaries=np.arange(-0.5,4.5))
plt.tight_layout()
savefig("application", "vulnerability_quadrant.png")

print(f"""
══════════════════════════════════════════════════════════════════════════════
  VISUALISATION PIPELINE COMPLETE                                             
══════════════════════════════════════════════════════════════════════════════
  Output directory : {OUTPUT_DIR:<54s}║
  01_study_area/      aoi_overview · fire_density · monthly_calendar         
  02_model_training/  loss_curves · smote_balance                            
  03_model_evaluation/ roc · pr · calibration · confusion_matrix             
                       horizon_metrics                                        
  04_spatial_forecast/ multihorizon_maps · error_maps                        
  05_interpretation/   feature_importance · wind_alignment                   
                       weather_vs_spread                                      
  06_application/      cumulative_burn · risk_composite                      
                       hotspot_ranking · vulnerability_quadrant               
  07_cv_results/       cv_fold_metrics · cv_horizon_bands · cv_train_sizes   
══════════════════════════════════════════════════════════════════════════════
""")