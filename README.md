# COSMEON — Climate Risk Intelligence

End-to-end flood detection engine that converts raw Sentinel-1 SAR satellite imagery into per-district flood risk scores. Built for the Bangladesh Delta, targeting emergency response teams, governments, and insurers who need structured, queryable flood intelligence — not raw rasters.

---

## What it does

- Ingests Sentinel-1 GRD scenes from Google Earth Engine / Planetary Computer
- Preprocesses SAR imagery (orbit correction, calibration, speckle filter, terrain correction)
- Detects flood extent via Otsu thresholding + optional U-Net ONNX inference
- Scores 20+ districts on flood coverage, exposed population, and composite risk level
- Serves results via a REST API with per-district GeoJSON and flood masks on S3

Detection latency: **< 30 minutes** from scene ingestion to queryable district scores.

---

## Project structure

```
HackX/
└── frontend/          # React dashboard (Vite + Tailwind + Zustand)
    ├── src/
    │   ├── components/
    │   │   ├── common/        # Nav, RiskBadge, StatCounter, LogFeed
    │   │   ├── dashboard/     # SatelliteOrbit canvas animation
    │   │   └── district/      # DistrictHexMap SVG, SparkChart
    │   ├── pages/
    │   │   ├── Dashboard.jsx          # Hero + stats + tech stack
    │   │   ├── MissionControl.jsx     # Pipeline simulator
    │   │   ├── DistrictIntelligence.jsx  # Interactive hex map + analytics
    │   │   └── ApiTerminal.jsx        # API docs + live tester
    │   ├── stores/            # Zustand state (app, api, map, pipeline)
    │   └── data/              # Mock data (districts, endpoints, pipeline stages)
    ├── tailwind.config.js
    ├── vite.config.js
    └── package.json
```

---

## Tech stack

**Frontend**

| Layer | Library |
|---|---|
| UI framework | React 18 |
| Styling | Tailwind CSS 3 |
| Animations | Framer Motion |
| Charts | Recharts |
| State | Zustand |
| Build | Vite 5 |

**Backend (pipeline)**

| Layer | Tools |
|---|---|
| Runtime | Python 3.11 · FastAPI · Uvicorn |
| Geospatial | GDAL 3.8 · Rasterio · GeoPandas · PostGIS |
| Satellite data | earthengine-api · pystac-client · sentinelhub |
| Detection | scikit-image · OpenCV · ONNX Runtime · PyTorch |
| Infrastructure | Celery + Redis · PostgreSQL 16 · MinIO/S3 · Docker |
| Observability | structlog · Prometheus · Sentry · OpenTelemetry |

---

## Pipeline stages

```
01  Scene Ingestion       Query STAC catalog, fetch Sentinel-1 GRD GeoTIFFs
02  SAR Preprocessing     Calibration, terrain correction, speckle filter, tiling
03  Flood Detection       Otsu threshold + U-Net ONNX inference, JRC water subtraction
04  Risk Scoring          Spatial join with GADM L2, WorldPop intersection, risk formula
05  Persist & Serve       Write to PostgreSQL/PostGIS, update run status, S3 presigned URLs
```

---

## Getting started

```bash
cd frontend
npm install
npm run dev       # dev server on http://localhost:5173
npm run build     # production build → dist/
```

---

## API overview

Base URL: `https://api.cosmeon.io/api/v1`

| Method | Path | Description |
|---|---|---|
| `POST` | `/runs` | Submit a new flood detection run |
| `GET` | `/runs/{run_id}` | Poll run execution status |
| `GET` | `/floods/{run_id}/districts` | Per-district risk results |
| `GET` | `/health` | Service health check |

Authentication: `X-API-Key` header. Rate limit: 10 req/min.

---

## Data coverage

- **AOI**: Bangladesh Delta (BD-DELTA-01), 144,000 km²
- **Districts**: 20 GADM Level-2 districts
- **Sensor**: Sentinel-1 GRD (VV + VH polarization)
- **Reference run**: 2024-11-04 flood event — 18,400 km² inundated, 4.2M exposed
