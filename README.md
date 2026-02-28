# Satellite Insight Engine v1

Hybrid CNN-LSTM flood risk predictor (`T+24h`) with TorchGeo ingestion scaffold, `uv`-first environment setup, MLflow tracking, and DVC artifact stages.

## 1) Bootstrap environment (GPU-first, background)

Run from repo root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_bootstrap_job.ps1
```

Check progress:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_bootstrap.ps1
```

Validate CUDA after completion:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\validate_gpu.ps1
```

## 2) Run pipeline

```powershell
.\.venv\Scripts\python -m src.ingest
.\.venv\Scripts\python -m src.train
.\.venv\Scripts\python -m src.risk_mapper
.\.venv\Scripts\python -m src.run_pipeline
```

## 3) Notes

- `configs/config.yaml` drives everything; no CLI arguments are required.
- If you need a different config path, set env var `PIPELINE_CONFIG_PATH`.
- Default ingest mode is `gee` with `materialization_after_gee: authentic`, which builds a real dataset from flood events.
- Scale dataset size from `gee.max_events`, `gee.positive_points_per_event`, `gee.negative_points_per_event`, and `gee.max_samples`.
- Risk report class mapping:
  - `Low`: `<0.30`
  - `Moderate`: `0.30-0.70`
  - `High`: `>0.70`
- Confidence formula: `abs(probability - 0.5) * 2`.
