import torch
import numpy as np
import joblib
from src.config import load_default_config, PROJECT_ROOT
from src.pipeline.ingestion.loader import FEATURE_COLUMNS, load_csv
from src.pipeline.training.model import build_model
from src.pipeline.training.run_train_forecast import make_sequences

cfg = load_default_config()
payload = torch.load("models/forecast_24h.pt", map_location="cpu", weights_only=True)
model = build_model(cfg, num_features=len(FEATURE_COLUMNS))
model.load_state_dict(payload["model_state_dict"])
model.eval()
scaler = joblib.load(PROJECT_ROOT / "artifacts" / "scaler.joblib")

# Load test split (last 20% of data)
df = load_csv(cfg)
label_col = cfg["data"]["label_column"]
n_train = int(len(df) * 0.8)

feat = df[FEATURE_COLUMNS].values.astype(np.float32)
lbl  = df[label_col].values.astype(np.float32)
feat[n_train:] = scaler.transform(feat[n_train:])

X_te, y_te = make_sequences(feat[n_train:], lbl[n_train:], 24)

# Sample 5000 random non-flood and 500 flood windows
rng = np.random.default_rng(42)
neg_idx = np.where(y_te == 0)[0]
pos_idx = np.where(y_te == 1)[0]
neg_sample = rng.choice(neg_idx, size=min(5000, len(neg_idx)), replace=False)
pos_sample = rng.choice(pos_idx, size=min(500,  len(pos_idx)), replace=False)

def run_batch(indices, X):
    probs = []
    with torch.no_grad():
        for i in range(0, len(indices), 256):
            batch = torch.from_numpy(X[indices[i:i+256]])
            p = torch.sigmoid(model(batch)).squeeze(-1).numpy()
            probs.extend(p.tolist())
    return np.array(probs)

neg_probs = run_batch(neg_sample, X_te)
pos_probs = run_batch(pos_sample, X_te)

print("=== NON-FLOOD samples (should be << 0.5) ===")
print(f"  mean={neg_probs.mean():.5f}  min={neg_probs.min():.5f}  max={neg_probs.max():.5f}")
print(f"  <0.5: {(neg_probs < 0.5).sum()}/{len(neg_probs)} ({100*(neg_probs<0.5).mean():.1f}%)")
print(f"  <0.1: {(neg_probs < 0.1).sum()}/{len(neg_probs)}")

print("\n=== FLOOD samples (should be >> 0.5) ===")
print(f"  mean={pos_probs.mean():.5f}  min={pos_probs.min():.5f}  max={pos_probs.max():.5f}")
print(f"  >0.5: {(pos_probs > 0.5).sum()}/{len(pos_probs)} ({100*(pos_probs>0.5).mean():.1f}%)")

print("\n=== LOGIT distribution for non-flood ===")
neg_logits = np.log(neg_probs / (1 - neg_probs + 1e-9))
print(f"  logit mean={neg_logits.mean():.5f}  min={neg_logits.min():.5f}  max={neg_logits.max():.5f}")
print(f"  logit < 0: {(neg_logits < 0).sum()}/{len(neg_logits)} ({100*(neg_logits<0).mean():.1f}%)")
