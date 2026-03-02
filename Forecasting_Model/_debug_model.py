import torch
import numpy as np
import joblib
from src.config import load_default_config, PROJECT_ROOT
from src.pipeline.ingestion.loader import FEATURE_COLUMNS
from src.pipeline.training.model import build_model
from src.api.weather_fetcher import fetch_forecast_window
from src.api.inference_forecast import _engineer_features

cfg    = load_default_config()
device = torch.device("cpu")
ckpt   = PROJECT_ROOT / "models" / "best.pt"
payload = torch.load(ckpt, map_location=device, weights_only=True)
model   = build_model(cfg, num_features=len(FEATURE_COLUMNS)).to(device)
model.load_state_dict(payload["model_state_dict"])
model.eval()
scaler = joblib.load(PROJECT_ROOT / "artifacts" / "scaler.joblib")

# Fiji — has real rain
lat, lon = -17.368, 178.143
raw_df = fetch_forecast_window(lat, lon, past_hours=24, forecast_hours=24)
is_fc  = raw_df["is_forecast"].values.copy()
ts     = raw_df["Timestamp"].values.copy()
feat_df = _engineer_features(raw_df.drop(columns=["is_forecast"]))
n_dropped = len(raw_df) - len(feat_df)
feat_df["is_forecast"] = is_fc[n_dropped:]

X_all    = feat_df[FEATURE_COLUMNS].values.astype(np.float32)
X_scaled = scaler.transform(X_all).astype(np.float32)

future_idx = feat_df.index[feat_df["is_forecast"].astype(bool)].tolist()
print(f"Future rows: {len(future_idx)}  |  Total feat_df rows: {len(feat_df)}")

# Check raw model output for first 3 future windows
W = 24
for i, idx in enumerate(future_idx[:3]):
    start  = max(0, idx - W + 1)
    window = X_scaled[start: idx + 1]
    if len(window) < W:
        pad    = np.repeat(window[:1], W - len(window), axis=0)
        window = np.concatenate([pad, window], axis=0)
    tensor = torch.from_numpy(window).unsqueeze(0)
    with torch.no_grad():
        raw_out = model(tensor)
        logit   = raw_out.item()
        prob    = torch.sigmoid(raw_out).item()
    print(f"  future[{i}] idx={idx}  window_shape={window.shape}  logit={logit:.6f}  prob={prob:.6f}")

# Also check: are all scaled inputs near zero?
print(f"\nX_scaled stats (future rows):")
future_X = X_scaled[future_idx]
print(f"  mean={future_X.mean():.4f}  std={future_X.std():.4f}  min={future_X.min():.4f}  max={future_X.max():.4f}")
print(f"  Rain_24h col index: {FEATURE_COLUMNS.index('Rain_24h')}")
print(f"  Rain_24h scaled values: {future_X[:, FEATURE_COLUMNS.index('Rain_24h')][:5]}")
