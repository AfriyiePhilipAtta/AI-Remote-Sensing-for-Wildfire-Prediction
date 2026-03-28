#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════════
  WILDFIRE SPREAD PREDICTION — CORE PIPELINE  (IMBALANCE-CORRECTED)         
  Upper West Ghana · ConvLSTM (PyTorch) · Gradient Boosting · Random Forest  
                                                                             
  Covers: Data loading, engineering, tensor construction,                    
          ConvLSTM, GB, RF, K-Fold Temporal CV (Stages 1–7)                 
                                                                              
  Outputs: cv_metrics.csv, horizon_metrics.csv, pipeline_metadata.json      
           (consumed by pipeline_viz.py for all figures)                     
══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              precision_recall_curve, roc_curve,
                              confusion_matrix, f1_score,
                              average_precision_score)
from sklearn.calibration import calibration_curve

# ── PyTorch imports ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("  [WARNING] imbalanced-learn not found. "
          "Install with: pip install imbalanced-learn")
    print("  [WARNING] Falling back to class_weight='balanced' only.")
    SMOTE_AVAILABLE = False

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────────────
#  SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH   = "upper_west_fire_spread.csv"
OUTPUT_DIR = "wildfire_outputs"

SUBDIRS = {
    "study_area":        os.path.join(OUTPUT_DIR, "01_study_area"),
    "model_training":    os.path.join(OUTPUT_DIR, "02_model_training"),
    "model_evaluation":  os.path.join(OUTPUT_DIR, "03_model_evaluation"),
    "spatial_forecast":  os.path.join(OUTPUT_DIR, "04_spatial_forecast"),
    "interpretation":    os.path.join(OUTPUT_DIR, "05_interpretation"),
    "application":       os.path.join(OUTPUT_DIR, "06_application"),
    "cv_results":        os.path.join(OUTPUT_DIR, "07_cv_results"),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
for d in SUBDIRS.values():
    os.makedirs(d, exist_ok=True)

GRID_RES      = 0.1
INTERVAL_DAYS = 3
SEQ_LEN       = 7
PRED_HORIZON  = 7

# ── K-Fold Temporal CV settings ───────────────────────────────────────────────
N_FOLDS = 5
CV_GAP  = SEQ_LEN + PRED_HORIZON   # = 14 sequences; prevents any look-ahead leakage

FOCAL_ALPHA = 0.90

FEATURES = [
    "ndvi", "nbr", "ndwi",
    "temp", "relative_humidity", "wind_speed", "wind_dir",
    "rainfall", "soil_moisture",
    "slope", "aspect", "elevation",
    "landcover",
]
FEATURE_LABELS = {
    "ndvi":              "NDVI",
    "nbr":               "NBR",
    "ndwi":              "NDWI",
    "temp":              "Temperature (°C)",
    "relative_humidity": "Rel. Humidity (%)",
    "wind_speed":        "Wind Speed (m/s)",
    "wind_dir":          "Wind Direction (°)",
    "rainfall":          "Rainfall (m)",
    "soil_moisture":     "Soil Moisture (m³/m³)",
    "slope":             "Slope (°)",
    "aspect":            "Aspect (°)",
    "elevation":         "Elevation (m)",
    "landcover":         "Land Cover Class",
}
C = len(FEATURES)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

HIDDEN   = 16
N_EPOCHS = 25
BATCH    = 4


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def valid_pairs(lst):
    return [(i + 1, v) for i, v in enumerate(lst) if not np.isnan(v)]


def norm01(a):
    mn, mx = np.nanmin(a), np.nanmax(a)
    return (a - mn) / (mx - mn + 1e-9)


def best_f1_threshold(y_true, y_prob, n_thresholds=200):
    """Return (threshold, f1) that maximises F1 score."""
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ──────────────────────────────────────────────────────────────────────────────
#  TEMPORAL K-FOLD SPLITTER
# ──────────────────────────────────────────────────────────────────────────────
def temporal_kfold_splits(n_sequences, n_folds=5, seq_len=7, horizon=7):
    """
    Expanding-window temporal K-Fold for sequence arrays.

    Design
    ------
    The timeline is divided into (n_folds + 1) equal blocks.
    Fold k trains on all sequences in blocks 0 … k and tests on block k+1.
    A mandatory gap of (seq_len + horizon) sequences sits between the last
    training index and the first test index, ensuring that no ConvLSTM input
    window in the test set overlaps with any training target window.
    This fully eliminates temporal look-ahead leakage.

    Yields
    ------
    fold_idx (int), train_indices (ndarray), test_indices (ndarray), gap (int)
    """
    gap      = seq_len + horizon      # = 14 for default settings
    block_sz = n_sequences // (n_folds + 1)

    for k in range(n_folds):
        train_end  = block_sz * (k + 1)
        test_start = train_end + gap
        test_end   = block_sz * (k + 2)

        if test_start >= n_sequences or test_end > n_sequences:
            print(f"  [CV] Fold {k+1} skipped — not enough sequences after the gap.")
            continue

        yield (k + 1,
               np.arange(0, train_end),
               np.arange(test_start, test_end),
               gap)


# ──────────────────────────────────────────────────────────────────────────────
#  SHARED HELPERS FOR STAGES 3-7
# ──────────────────────────────────────────────────────────────────────────────
def tabularise(X, Y, threshold=0.15):
    Xf = X.mean(axis=1)
    Xf = Xf.reshape(len(X), C, -1).mean(-1)
    Yf = (Y[:, 0].reshape(len(X), -1).mean(-1) > threshold).astype(int)
    return Xf, Yf


def find_tab_threshold(X_tr, Y_tr, X_te, Y_te):
    for thr in [0.15, 0.10, 0.05, 0.02, 0.01]:
        Xtr_, Ytr_ = tabularise(X_tr, Y_tr, thr)
        Xte_, Yte_ = tabularise(X_te, Y_te, thr)
        if len(np.unique(Ytr_)) >= 2 and len(np.unique(Yte_)) >= 2:
            return Xtr_, Ytr_, Xte_, Yte_, thr
    return None, None, None, None, None


# ──────────────────────────────────────────────────────────────────────────────
#  PYTORCH ConvLSTM MODEL
# ──────────────────────────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell operating on spatial feature maps."""

    def __init__(self, in_ch, hidden):
        super().__init__()
        # All 4 gates computed in one convolution for efficiency
        self.conv = nn.Conv2d(
            in_channels  = in_ch + hidden,
            out_channels = 4 * hidden,
            kernel_size  = 3,
            padding      = 1,   # keeps spatial dimensions the same
            bias         = True
        )
        self.hidden = hidden

        # Initialise forget gate bias to 1 (helps gradient flow early in training)
        nn.init.constant_(self.conv.bias[hidden : 2 * hidden], 1.0)

    def forward(self, x_t, h, c):
        """
        x_t : (batch, in_ch,  H, W)
        h   : (batch, hidden, H, W)
        c   : (batch, hidden, H, W)
        """
        combined = torch.cat([x_t, h], dim=1)   # (batch, in_ch+hidden, H, W)
        gates    = self.conv(combined)           # (batch, 4*hidden, H, W)

        H = self.hidden
        i_g = torch.sigmoid(gates[:, 0*H : 1*H])   # input gate
        f_g = torch.sigmoid(gates[:, 1*H : 2*H])   # forget gate
        o_g = torch.sigmoid(gates[:, 2*H : 3*H])   # output gate
        g   = torch.tanh(   gates[:, 3*H : 4*H])   # cell gate

        c_new = f_g * c + i_g * g
        h_new = o_g * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMModel(nn.Module):
    """
    Spatiotemporal wildfire spread predictor.

    Architecture
    ------------
    Input  : (batch, seq_len, C, H, W)  — C feature channels, spatial grid H×W
    Encoder: ConvLSTM cell unrolled over seq_len steps
    Decoder: 1×1 convolution mapping hidden state → horizon predictions
    Output : (batch, horizon, H, W)  — fire spread probability maps
    """

    def __init__(self, in_ch, hidden, horizon):
        super().__init__()
        self.cell    = ConvLSTMCell(in_ch, hidden)
        self.decoder = nn.Conv2d(hidden, horizon, kernel_size=1)  # 1×1 conv
        self.hidden  = hidden
        self.horizon = horizon

    def forward(self, X):
        """
        X : (batch, seq_len, C, H, W)
        Returns: (batch, horizon, H, W) — values in [0, 1]
        """
        B, T, C_, Hs, Ws = X.shape
        h = torch.zeros(B, self.hidden, Hs, Ws, device=X.device)
        c = torch.zeros(B, self.hidden, Hs, Ws, device=X.device)

        for t in range(T):
            h, c = self.cell(X[:, t], h, c)    # step through sequence

        out = self.decoder(h)                   # (batch, horizon, H, W)
        return torch.sigmoid(out)


# ──────────────────────────────────────────────────────────────────────────────
#  FOCAL LOSS (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary focal loss for class-imbalanced spatial predictions.

    Parameters
    ----------
    alpha : float
        Weight for the positive (fire) class.  0.90 = strong focus on fire.
    gamma : float
        Focusing parameter.  2.0 is the standard value from the original paper.
    """

    def __init__(self, alpha=0.90, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        pred   : (batch, horizon, H, W) — probabilities in [0, 1]
        target : (batch, horizon, H, W) — binary labels
        """
        eps = 1e-7
        p   = pred.clamp(eps, 1 - eps)

        # Standard binary cross-entropy weighted by alpha
        ce  = -(self.alpha       * target       * torch.log(p)
              + (1 - self.alpha) * (1 - target) * torch.log(1 - p))

        # Focal weight: down-weights easy examples
        pt  = torch.where(target == 1, p, 1 - p)
        loss = ce * (1 - pt) ** self.gamma

        return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
#  PYTORCH TRAINING HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def train_convlstm(model, X_train, Y_train, X_val, Y_val,
                   n_epochs=25, batch_size=4, lr=2e-3, focal_alpha=0.90):
    """
    Train ConvLSTMModel with focal loss and Adam optimizer.

    Parameters
    ----------
    model     : ConvLSTMModel instance
    X_train   : numpy array (N, seq_len, C, H, W)
    Y_train   : numpy array (N, horizon, H, W)
    X_val     : numpy array for validation loss monitoring
    Y_val     : numpy array for validation loss monitoring
    n_epochs  : int
    batch_size: int
    lr        : float — learning rate
    focal_alpha: float — passed to FocalLoss

    Returns
    -------
    train_losses, val_losses : lists of per-epoch losses
    """
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert numpy arrays to PyTorch tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    Y_tr = torch.tensor(Y_train, dtype=torch.float32)
    dataset    = TensorDataset(X_tr, Y_tr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Small validation set (up to 8 samples to keep it fast)
    X_v = torch.tensor(X_val[:8], dtype=torch.float32).to(DEVICE)
    Y_v = torch.tensor(Y_val[:8], dtype=torch.float32).to(DEVICE)

    train_losses, val_losses = [], []

    model.train()
    for epoch in range(1, n_epochs + 1):
        ep_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()

        # Validation loss
        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_v), Y_v).item()
        model.train()

        train_losses.append(ep_loss / len(dataloader))
        val_losses.append(vl)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  │  Epoch {epoch:2d}/{n_epochs}  "
                  f"train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

    return train_losses, val_losses


def predict_convlstm(model, X):
    """
    Run inference on numpy array X.
    Returns numpy array of shape (N, horizon, H, W).
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    return preds


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 1 — LOAD CSV                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  WILDFIRE SPREAD PREDICTION PIPELINE  (IMBALANCE-CORRECTED + TEMPORAL CV)  ║
║  Upper West Ghana · ConvLSTM (PyTorch) · Gradient Boosting · Random Forest  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("═" * 70)
print("  STAGE 1 · LOADING CSV DATA")
print("═" * 70)

print(f"  Loading: {CSV_PATH}")
df_raw = pd.read_csv(CSV_PATH, parse_dates=["date"])

if "geo" in df_raw.columns:
    df_raw = df_raw.drop(columns=["geo"])

required = FEATURES + ["date", "lat", "lon", "fire_t", "fire_t_plus1", "spread"]
missing  = [c for c in required if c not in df_raw.columns]
if missing:
    print(f"\n  ERROR: Missing columns: {missing}")
    print(f"  Available: {list(df_raw.columns)}")
    sys.exit(1)

df_raw = df_raw.dropna(subset=FEATURES)
df_raw["lat"] = (df_raw["lat"] / GRID_RES).round() * GRID_RES
df_raw["lon"] = (df_raw["lon"] / GRID_RES).round() * GRID_RES
df_raw = df_raw.sort_values(["date", "lat", "lon"]).reset_index(drop=True)
df_raw["month"] = df_raw["date"].dt.month
df_raw["year"]  = df_raw["date"].dt.year

lats = sorted(df_raw["lat"].unique())
lons = sorted(df_raw["lon"].unique())
H, W = len(lats), len(lons)
LON2D, LAT2D = np.meshgrid(lons, lats)
EXT = [min(lons), max(lons), min(lats), max(lats)]

print(f"  Rows       : {len(df_raw):,}")
print(f"  Dates      : {df_raw['date'].nunique()}")
print(f"  Grid       : {H} × {W}")
print(f"  Date range : {df_raw['date'].min().strftime('%Y-%m-%d')}"
      f" → {df_raw['date'].max().strftime('%Y-%m-%d')}")
print(f"  Fire rate  : {df_raw['fire_t'].mean()*100:.2f}%")
print(f"  Spread rate: {df_raw['spread'].mean()*100:.2f}%")
imbalance_ratio = (df_raw['spread'] == 0).sum() / max((df_raw['spread'] == 1).sum(), 1)
print(f"  Imbalance ratio (no-spread:spread): {imbalance_ratio:.1f}:1")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 2 — DATA ENGINEERING & TENSOR CONSTRUCTION                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("  STAGE 2 · DATA ENGINEERING")
print("═" * 70)

df        = df_raw.copy()
scaler    = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

dates_all   = sorted(df["date"].unique())
T           = len(dates_all)
date_to_idx = {d: i for i, d in enumerate(dates_all)}
lat_to_idx  = {v: i for i, v in enumerate(lats)}
lon_to_idx  = {v: i for i, v in enumerate(lons)}

feat_tensor   = np.zeros((T, C, H, W), dtype=np.float32)
fire_t_tensor = np.zeros((T, H, W),    dtype=np.float32)
spread_tensor = np.zeros((T, H, W),    dtype=np.float32)

for _, row in df.iterrows():
    t = date_to_idx[row["date"]]
    i = lat_to_idx[row["lat"]]
    j = lon_to_idx[row["lon"]]
    feat_tensor[t, :, i, j] = row[FEATURES].values.astype(np.float32)
    fire_t_tensor[t, i, j]  = float(row["fire_t"])
    spread_tensor[t, i, j]  = float(row["spread"])

print(f"  feat_tensor   : {feat_tensor.shape}  (T × C × H × W)")
print(f"  fire_tensor   : {fire_t_tensor.shape}")
print(f"  spread_tensor : {spread_tensor.shape}")

X_list, Y_list = [], []
for t in range(T - SEQ_LEN - PRED_HORIZON + 1):
    X_list.append(feat_tensor[t : t + SEQ_LEN])
    Y_list.append(spread_tensor[t + SEQ_LEN : t + SEQ_LEN + PRED_HORIZON])

X_all = np.array(X_list, dtype=np.float32)
Y_all = np.array(Y_list, dtype=np.float32)
N     = len(X_all)

print(f"  Total sequences : {N}")
print(f"  X shape         : {X_all.shape}")
print(f"  Y shape         : {Y_all.shape}")
print(f"\n  NOTE: All metrics now come from K-Fold Temporal CV ({N_FOLDS} folds).")
print(f"        The final fold's model objects are reused for spatial figures.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  STAGES 3–7 · K-FOLD TEMPORAL CROSS-VALIDATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("  STAGES 3–7 · K-FOLD TEMPORAL CROSS-VALIDATION")
print("═" * 70)
print(f"  Strategy : Expanding-window — fold k trains on blocks 0…k,")
print(f"             tests on block k+1")
print(f"  Folds    : {N_FOLDS}")
print(f"  Gap      : {CV_GAP} sequences = seq_len({SEQ_LEN}) + horizon({PRED_HORIZON})")
print(f"  Purpose  : Eliminates look-ahead leakage; no test window overlaps")
print(f"             any training target window\n")

cv_records = []

for fold, tr_idx, te_idx, gap in temporal_kfold_splits(N, N_FOLDS, SEQ_LEN, PRED_HORIZON):

    X_train = X_all[tr_idx]
    Y_train = Y_all[tr_idx]
    X_test  = X_all[te_idx]
    Y_test  = Y_all[te_idx]

    print(f"  ┌─ Fold {fold}/{N_FOLDS}  "
          f"train=[0…{tr_idx[-1]}]({len(tr_idx)} seqs)  "
          f"gap=[{tr_idx[-1]+1}…{te_idx[0]-1}]  "
          f"test=[{te_idx[0]}…{te_idx[-1]}]({len(te_idx)} seqs)")

    # ── PyTorch ConvLSTM ──────────────────────────────────────────────────────
    torch.manual_seed(42)
    clstm = ConvLSTMModel(in_ch=C, hidden=HIDDEN, horizon=PRED_HORIZON).to(DEVICE)

    train_losses_f, val_losses_f = train_convlstm(
        model       = clstm,
        X_train     = X_train,
        Y_train     = Y_train,
        X_val       = X_test,
        Y_val       = Y_test,
        n_epochs    = N_EPOCHS,
        batch_size  = BATCH,
        lr          = 2e-3,
        focal_alpha = FOCAL_ALPHA,
    )

    clstm_preds_f = predict_convlstm(clstm, X_test)
    print(f"  │  ConvLSTM done | preds: {clstm_preds_f.shape}")

    # ── Tabularise + SMOTE ────────────────────────────────────────────────────
    X_tr_t, Y_tr_t, X_te_t, Y_te_t, TAB_THRESH_F = find_tab_threshold(
        X_train, Y_train, X_test, Y_test)

    if X_tr_t is None:
        print(f"  │  [SKIP] No 2-class threshold found — fold skipped.\n")
        continue

    print(f"  │  Tabularisation threshold : {TAB_THRESH_F}")
    print(f"  │  Train class dist: {np.bincount(Y_tr_t)}  "
          f"(ratio {np.bincount(Y_tr_t)[0]/max(np.bincount(Y_tr_t)[1],1):.1f}:1)")

    if SMOTE_AVAILABLE and len(np.unique(Y_tr_t)) == 2 and np.bincount(Y_tr_t)[1] >= 2:
        k_nn   = min(5, max(1, np.bincount(Y_tr_t)[1] - 1))
        smote  = SMOTE(random_state=42, k_neighbors=k_nn)
        X_tr_s, Y_tr_s = smote.fit_resample(X_tr_t, Y_tr_t)
        print(f"  │  SMOTE k={k_nn}: {np.bincount(Y_tr_t)} → {np.bincount(Y_tr_s)}")
    else:
        X_tr_s, Y_tr_s = X_tr_t, Y_tr_t

    # ── Gradient Boosting ─────────────────────────────────────────────────────
    gb_f = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=5, random_state=42)
    n0_s, n1_s = np.bincount(Y_tr_s)
    sw = np.where(Y_tr_s == 1,
                  len(Y_tr_s) / (2 * n1_s),
                  len(Y_tr_s) / (2 * n0_s))
    gb_f.fit(X_tr_s, Y_tr_s, sample_weight=sw)
    gb_prob_f = gb_f.predict_proba(X_te_t)[:, 1]
    gb_auc_f  = roc_auc_score(Y_te_t, gb_prob_f) if len(np.unique(Y_te_t)) > 1 else np.nan

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_f = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1)
    rf_f.fit(X_tr_s, Y_tr_s)
    rf_prob_f = rf_f.predict_proba(X_te_t)[:, 1]
    rf_auc_f  = roc_auc_score(Y_te_t, rf_prob_f) if len(np.unique(Y_te_t)) > 1 else np.nan

    print(f"  │  GB AUC={gb_auc_f:.4f}  RF AUC={rf_auc_f:.4f}")

    # ── Weighted ensemble ─────────────────────────────────────────────────────
    clstm_tab_f = minmax_scale(
        clstm_preds_f[:, 0].reshape(len(X_test), -1).mean(axis=-1))
    ensemble_f  = 0.50 * clstm_tab_f + 0.30 * gb_prob_f + 0.20 * rf_prob_f
    ens_auc_f   = roc_auc_score(Y_te_t, ensemble_f) if len(np.unique(Y_te_t)) > 1 else np.nan
    ens_thr_f, ens_f1_f = best_f1_threshold(Y_te_t, ensemble_f)
    print(f"  │  Ensemble AUC={ens_auc_f:.4f}  F1={ens_f1_f:.4f} (thr={ens_thr_f:.2f})")

    # ── Per-horizon metrics ───────────────────────────────────────────────────
    fold_auc, fold_f1, fold_ap, fold_brier, fold_thr = [], [], [], [], []

    for h in range(PRED_HORIZON):
        y_t = (Y_test[:, h].reshape(len(X_test), -1).mean(axis=-1)
               > TAB_THRESH_F).astype(int)
        y_p = np.clip(
            minmax_scale(clstm_preds_f[:, h].reshape(len(X_test), -1).mean(axis=-1)),
            0.0, 1.0)

        if len(np.unique(y_t)) < 2:
            fold_auc.append(np.nan); fold_f1.append(np.nan)
            fold_ap.append(np.nan);  fold_brier.append(np.nan)
            fold_thr.append(0.5)
            continue

        opt_t, opt_f1_h = best_f1_threshold(y_t, y_p)
        fold_thr.append(opt_t)
        fold_auc.append(roc_auc_score(y_t, y_p))
        fold_f1.append(opt_f1_h)
        fold_ap.append(average_precision_score(y_t, y_p))
        fold_brier.append(brier_score_loss(y_t, y_p))

    cv_records.append({
        "fold":               fold,
        "train_size":         len(tr_idx),
        "test_size":          len(te_idx),
        "tab_threshold":      TAB_THRESH_F,
        "horizon_auc":        fold_auc,
        "horizon_f1":         fold_f1,
        "horizon_ap":         fold_ap,
        "horizon_brier":      fold_brier,
        "horizon_thr":        fold_thr,
        "ensemble_auc":       ens_auc_f,
        "ensemble_f1":        ens_f1_f,
        "ensemble_threshold": ens_thr_f,
        "train_losses":       train_losses_f,
        "val_losses":         val_losses_f,
        # ── model objects kept for final-fold figure generation ──
        "_clstm":             clstm,
        "_clstm_preds":       clstm_preds_f,
        "_gb":                gb_f,
        "_rf":                rf_f,
        "_X_test":            X_test,
        "_Y_test":            Y_test,
        "_Y_te_t":            Y_te_t,
        "_ensemble_p":        ensemble_f,
        "_gb_prob":           gb_prob_f,
        "_rf_prob":           rf_prob_f,
        "_X_tr_smote":        X_tr_s,
        "_Y_tr_smote":        Y_tr_s,
        "_Y_tr_t":            Y_tr_t,
    })
    print(f"  └─ Fold {fold} complete\n")


# ── Aggregate & print CV summary ──────────────────────────────────────────────
print("═" * 70)
print("  CROSS-VALIDATION SUMMARY")
print("═" * 70)

cv_auc_mat   = np.array([r["horizon_auc"]   for r in cv_records])
cv_f1_mat    = np.array([r["horizon_f1"]    for r in cv_records])
cv_ap_mat    = np.array([r["horizon_ap"]    for r in cv_records])
cv_brier_mat = np.array([r["horizon_brier"] for r in cv_records])

print(f"\n  {'Day':<6} {'AUC mean±std':<22} {'F1 mean±std':<22} "
      f"{'AP mean±std':<22} {'Brier mean±std'}")
print("  " + "-" * 90)
for h in range(PRED_HORIZON):
    def _fmt(v):
        v = v[~np.isnan(v)]
        return f"{np.mean(v):.4f} ± {np.std(v):.4f}" if len(v) else "  n/a            "
    print(f"  Day+{h+1:<2}  "
          f"{_fmt(cv_auc_mat[:,h]):<22} "
          f"{_fmt(cv_f1_mat[:,h]):<22} "
          f"{_fmt(cv_ap_mat[:,h]):<22} "
          f"{_fmt(cv_brier_mat[:,h])}")

ens_aucs = np.array([r["ensemble_auc"] for r in cv_records])
ens_f1s  = np.array([r["ensemble_f1"]  for r in cv_records])
print(f"\n  Ensemble (Day+1):  "
      f"AUC = {np.nanmean(ens_aucs):.4f} ± {np.nanstd(ens_aucs):.4f}   "
      f"F1 = {np.nanmean(ens_f1s):.4f} ± {np.nanstd(ens_f1s):.4f}")

# Save CV metrics CSV
cv_rows = []
for r in cv_records:
    for h in range(PRED_HORIZON):
        cv_rows.append({
            "fold":            r["fold"],
            "horizon":         h + 1,
            "auc":             r["horizon_auc"][h],
            "f1":              r["horizon_f1"][h],
            "avg_prec":        r["horizon_ap"][h],
            "brier":           r["horizon_brier"][h],
            "opt_threshold":   r["horizon_thr"][h],
            "ensemble_auc":    r["ensemble_auc"],
            "ensemble_f1":     r["ensemble_f1"],
            "ens_threshold":   r["ensemble_threshold"],
        })
cv_df = pd.DataFrame(cv_rows)
cv_csv = os.path.join(OUTPUT_DIR, "cv_metrics.csv")
cv_df.to_csv(cv_csv, index=False)
print(f"\n  CV metrics saved → {cv_csv}")

# ── Unpack final fold for all downstream figure generation ────────────────────
final          = cv_records[-1]
clstm          = final["_clstm"]
clstm_preds    = final["_clstm_preds"]
gb             = final["_gb"]
rf             = final["_rf"]
X_test         = final["_X_test"]
Y_test         = final["_Y_test"]
Y_te_t         = final["_Y_te_t"]
ensemble_p     = final["_ensemble_p"]
gb_prob        = final["_gb_prob"]
rf_prob        = final["_rf_prob"]
OPT_THRESH_ENS = final["ensemble_threshold"]
OPT_F1_ENS     = final["ensemble_f1"]
TAB_THRESH     = final["tab_threshold"]
train_losses   = final["train_losses"]
val_losses     = final["val_losses"]
auc_h          = final["horizon_auc"]
brier_h        = final["horizon_brier"]
f1_h           = final["horizon_f1"]
ap_h           = final["horizon_ap"]
gb_auc = roc_auc_score(Y_te_t, gb_prob) if len(np.unique(Y_te_t)) > 1 else np.nan
rf_auc = roc_auc_score(Y_te_t, rf_prob) if len(np.unique(Y_te_t)) > 1 else np.nan

pd.DataFrame({
    "horizon":  range(1, PRED_HORIZON + 1),
    "auc":      auc_h,
    "brier":    brier_h,
    "f1":       f1_h,
    "avg_prec": ap_h,
}).to_csv(f"{OUTPUT_DIR}/horizon_metrics.csv", index=False)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SAVE METADATA & SUMMARY                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("  SAVING METADATA & SUMMARY")
print("═" * 70)

m_probas = [np.clip(minmax_scale(clstm_preds[:,0].reshape(len(X_test),-1).mean(axis=-1)),0,1),
            gb_prob, rf_prob, ensemble_p]
m_aucs   = [roc_auc_score(Y_te_t,p) if len(np.unique(Y_te_t))>1 else np.nan
            for p in m_probas]

meta = {
    "csv_path":        CSV_PATH,
    "grid_resolution": GRID_RES,
    "seq_len":         SEQ_LEN,
    "pred_horizon":    PRED_HORIZON,
    "n_features":      C,
    "features":        FEATURES,
    "grid_shape":      [H, W],
    "n_dates":         T,
    "tab_threshold":   TAB_THRESH,
    "convlstm_backend": "PyTorch",
    "device":           str(DEVICE),
    "imbalance_fixes": {
        "smote_applied":        bool(SMOTE_AVAILABLE),
        "focal_alpha":          FOCAL_ALPHA,
        "gb_sample_weight":     "inverse class frequency",
        "rf_class_weight":      "balanced",
        "opt_threshold_method": "max-F1 search",
        "opt_threshold_ens":    round(float(OPT_THRESH_ENS), 3),
    },
    "temporal_cv": {
        "n_folds":          N_FOLDS,
        "gap_sequences":    CV_GAP,
        "strategy":         "expanding-window, no look-ahead leakage",
        "folds_completed":  len(cv_records),
        "cv_mean_auc_d1":   round(float(np.nanmean([r["horizon_auc"][0] for r in cv_records])), 4),
        "cv_std_auc_d1":    round(float(np.nanstd( [r["horizon_auc"][0] for r in cv_records])), 4),
        "cv_mean_f1_d1":    round(float(np.nanmean([r["horizon_f1"][0]  for r in cv_records])), 4),
        "cv_std_f1_d1":     round(float(np.nanstd( [r["horizon_f1"][0]  for r in cv_records])), 4),
        "cv_mean_ap_d1":    round(float(np.nanmean([r["horizon_ap"][0]  for r in cv_records])), 4),
        "cv_std_ap_d1":     round(float(np.nanstd( [r["horizon_ap"][0]  for r in cv_records])), 4),
        "ens_mean_auc_d1":  round(float(np.nanmean(ens_aucs)), 4),
        "ens_std_auc_d1":   round(float(np.nanstd(ens_aucs)),  4),
    },
    "convlstm_final_fold": {
        "hidden":           HIDDEN,
        "epochs":           N_EPOCHS,
        "batch":            BATCH,
        "focal_alpha":      FOCAL_ALPHA,
        "final_train_loss": round(float(train_losses[-1]), 5),
        "final_val_loss":   round(float(val_losses[-1]),   5),
    },
    "model_auc_final_fold": {
        "convlstm":       round(float(m_aucs[0]), 4),
        "gradient_boost": round(float(m_aucs[1]), 4),
        "random_forest":  round(float(m_aucs[2]), 4),
        "ensemble":       round(float(m_aucs[3]), 4),
    },
    "ensemble_f1_final_fold": round(float(OPT_F1_ENS), 4),
    "date_range": {"start": df_raw["date"].min().strftime("%Y-%m-%d"),
                   "end":   df_raw["date"].max().strftime("%Y-%m-%d")},
    "extent": {"lon_min":EXT[0],"lon_max":EXT[1],"lat_min":EXT[2],"lat_max":EXT[3]},
    "output_structure": {
        "01_study_area":       ["aoi_overview.png","fire_density.png","monthly_calendar.png"],
        "02_model_training":   ["loss_curves.png","smote_balance.png"],
        "03_model_evaluation": ["roc.png","pr.png","calibration.png",
                                "confusion_matrix.png","horizon_metrics.png"],
        "04_spatial_forecast": ["multihorizon_maps.png","error_maps.png"],
        "05_interpretation":   ["feature_importance.png","wind_alignment.png",
                                "weather_vs_spread.png"],
        "06_application":      ["cumulative_burn.png","risk_composite.png",
                                "hotspot_ranking.png","vulnerability_quadrant.png"],
        "07_cv_results":       ["cv_fold_metrics.png","cv_horizon_bands.png",
                                "cv_train_sizes.png"],
    },
}


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


meta_path = f"{OUTPUT_DIR}/pipeline_metadata.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2, cls=_NumpyEncoder)
print(f"  Metadata saved → {meta_path}")

cv_auc_mean = meta["temporal_cv"]["cv_mean_auc_d1"]
cv_auc_std  = meta["temporal_cv"]["cv_std_auc_d1"]
cv_f1_mean  = meta["temporal_cv"]["cv_mean_f1_d1"]
cv_f1_std   = meta["temporal_cv"]["cv_std_f1_d1"]
cv_ap_mean  = meta["temporal_cv"]["cv_mean_ap_d1"]
cv_ap_std   = meta["temporal_cv"]["cv_std_ap_d1"]

print(f"""
══════════════════════════════════════════════════════════════════════════════
  CORE PIPELINE COMPLETE  (IMBALANCE-CORRECTED + TEMPORAL CV)                
══════════════════════════════════════════════════════════════════════════════
  Imbalance fixes applied:                                                    
    1. SMOTE oversampling on tabular training split (per fold)                
    2. GradientBoosting: sample_weight = inverse class frequency              
      RandomForest: class_weight='balanced'                                  
    3. ConvLSTM focal-loss alpha raised  0.75 → {FOCAL_ALPHA}                         
    4. Decision threshold optimised (max-F1), not hard 0.5                   
    5. F1 & Average Precision now primary metrics; AUC is reference           
    6. Tabularisation threshold search extended to 0.01                      
    7. K-Fold Temporal CV: {N_FOLDS} folds · gap={CV_GAP} seqs · expanding window        
    8. ConvLSTM backend: PyTorch (replaces manual NumPy implementation)      
══════════════════════════════════════════════════════════════════════════════
  Cross-Validation Results (Day+1)                                            
    AUC : {cv_auc_mean:.4f} ± {cv_auc_std:.4f}                                         
    F1  : {cv_f1_mean:.4f} ± {cv_f1_std:.4f}                                         
    AP  : {cv_ap_mean:.4f} ± {cv_ap_std:.4f}                                         
══════════════════════════════════════════════════════════════════════════════
  Run pipeline_viz.py next to generate all figures.                          
══════════════════════════════════════════════════════════════════════════════
""")