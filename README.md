# COSMEON — Satellite Data to Insight Engine for Climate Risk

**Transforming Earth observation data into real-time flood risk intelligence for governments, insurers, and emergency responders.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Built with React](https://img.shields.io/badge/Built_with-React-blue.svg)](https://reactjs.org)
[![Python Backend](https://img.shields.io/badge/Backend-Python_FastAPI-green.svg)](https://fastapi.tiangolo.com)
[![Geospatial](https://img.shields.io/badge/Geospatial-GDAL_|_Rasterio-orange.svg)](https://rasterio.readthedocs.io)
[![3D Globe](https://img.shields.io/badge/3D-Cesium_JS-lightblue.svg)](https://cesium.com)

---

## What This Solves

**The Problem:** Governments, insurers, and disaster management agencies detect flood events days or weeks after they occur—if at all. Raw satellite imagery sits in public archives, underutilized because converting pixels to actionable intelligence requires advanced geospatial processing, ML expertise, and expensive infrastructure.

**The Solution:** COSMEON is a fully automated pipeline that:
- **Ingests** open Sentinel-1 SAR satellite imagery in near real-time
- **Detects** flood-affected areas via change detection and ML-powered image processing
- **Scores** geographic regions (districts, municipalities) with quantified flood risk, affected population estimates, and confidence metrics
- **Delivers** structured, API-accessible intelligence—not raw rasters—ready for immediate decision-making

**Impact:** From satellite scene acquisition to queryable district-level risk scores in **< 30 minutes**, with no proprietary data costs or vendor lock-in.

---

## Why This Solution Stands Out

### 1. **End-to-End Automation**
Most geospatial tools require manual intervention—downloading imagery, running preprocessing, tuning detection parameters. COSMEON orchestrates the entire pipeline (ingestion → preprocessing → detection → risk scoring → storage) with minimal configuration. Submit a location and date range; get structured results.

### 2. **Open Data, No Vendor Lock-in**
Built entirely on **free, public satellite data** (Sentinel-1/Sentinel-2 from Copernicus/USGS). No expensive proprietary imagery subscriptions. Scalable globally without dependency on commercial data providers.

### 3. **Decision-Ready Output, Not Just Visualization**
Unlike GIS dashboards that display rasters, COSMEON outputs:
- **Per-district flood area** (km²) and percentage coverage
- **Exposed population** estimates via spatial join with WorldPop
- **Risk classification** (Critical/High/Medium/Low) with confidence scoring
- **Historical trend** analysis comparing pre/post-event periods

Governments can immediately alert officials. Insurers can assess exposure. Urban planners can allocate resources.

### 4. **Temporal Change Detection**
The system doesn't analyze a single snapshot. It compares satellite imagery from before and after an event (or across time periods) using SAR backscatter change metrics. This enables:
- **Early warning** by detecting water surface changes before official reports
- **Trend analysis** to forecast future risk based on historical patterns
- **Nuanced risk** scoring that accounts for water permanence, depth proxies (SAR dB drop), and false positives

### 5. **Confidence-Aware Risk Classification**
Flood detections include confidence scores (High/Medium/Low) based on SAR signal quality, cloud cover, and detection algorithm certainty. Stakeholders can filter by confidence to prioritize high-certainty alerts and reduce false-alarm response costs.

### 6. **Multi-Stakeholder Ready**
- **Governments**: District-level summaries for emergency response coordination
- **Insurers**: Risk labels and exposure metrics for claims assessment and premium modeling
- **Urban Planners**: Spatial distribution of affected zones for infrastructure investment
- **Agricultural Users**: Environmental zone monitoring for crop risk assessment

### 7. **Predictive, Not Just Reactive**
Includes historical trend forecasting to predict future flood-prone areas based on past inundation patterns—shifting the tool from passive reporting to proactive early warning.

### 8. **API-First Design**
All processed intelligence is accessible programmatically via REST endpoints. Third-party systems (government portals, insurance platforms, disaster management tools) can integrate climate risk data directly into existing workflows.

---

## Quick Start

### Prerequisites
- **Node.js** 18+ (frontend)
- **Python** 3.11+ (backend)
- **Docker** (optional, for containerized deployment)
- **MongoDB Atlas** account (or local MongoDB)
- **Cesium Ion token** (optional, for satellite imagery; free tier available)

### Installation

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev          # Start dev server on http://localhost:5173
npm run build        # Production build
```

#### Backend Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Set up .env (see Environment Variables section)
cp .env.example .env
# Edit .env with your MongoDB connection string, OAuth credentials, etc.

python mongo/run.py   # Start FastAPI server on http://localhost:5000
```

---

## Environment Variables

### Frontend (`.env` in `frontend/` directory)

| Variable | Purpose | Example |
|----------|---------|---------|
| `VITE_API_URL` | Backend API base URL | `http://localhost:5000` |
| `VITE_INSIGHTS_API_URL` | Insights analysis service | `https://your-insights-api.com` |
| `VITE_RISK_API_URL` | Risk modeling service | `https://your-risk-api.com` |
| `VITE_LIFELINE_API_URL` | Infrastructure impact service | `https://your-lifeline-api.com` |
| `VITE_FORECAST_API_URL` | Flood forecast service | `https://your-forecast-api.com` |
| `VITE_CESIUM_TOKEN` | Cesium Ion access token (optional) | `eyJhbGci...` |
| `VITE_MOCK_MODE` | Use mock data in development | `true` or `false` |

### Backend (`.env` in project root)

| Variable | Purpose | Example |
|----------|---------|---------|
| `MONGO_DB_CONNECTION_STRING` | MongoDB Atlas connection URI | `mongodb+srv://user:pass@cluster.mongodb.net/?appName=...` |
| `JWT_SECRET` | Secret key for JWT signing (change in production!) | `your-secret-key-here` |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID | `123456789-abc.apps.googleusercontent.com` |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret | `GOCSPX-...` |
| `GOOGLE_REDIRECT_URI` | OAuth callback redirect URL | `http://localhost:5000/auth/google/callback` |
| `FRONTEND_URL` | Frontend URL (for OAuth redirects) | `http://localhost:5173` |

---

## Project Structure

```
HackX/
├── frontend/                    # React application (Vite)
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/          # Shared components (Nav, RiskBadge, StatCounter)
│   │   │   ├── globe/           # 3D Cesium globe visualization
│   │   │   ├── district/        # District hexagon map & analytics
│   │   │   ├── dashboard/       # Satellite orbit animations
│   │   │   └── ui/              # Form inputs (GeoSearchInput, CalendarPicker)
│   │   ├── pages/
│   │   │   ├── Landing.jsx      # Marketing landing page
│   │   │   ├── Login.jsx        # Authentication page
│   │   │   ├── MissionControl.jsx # Pipeline status monitor
│   │   │   ├── GlobeAnalysis.jsx  # Interactive flood detection interface
│   │   │   └── FloodInsights.jsx  # Historical analysis results viewer
│   │   ├── api/                 # API client modules
│   │   ├── stores/              # Zustand state management
│   │   ├── hooks/               # Custom React hooks
│   │   ├── utils/               # Utility functions (report generation, etc.)
│   │   └── data/                # Static data (districts, mock responses)
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── .env.example
│
├── mongo/                       # FastAPI authentication service
│   ├── app.py                   # FastAPI application
│   ├── routes.py                # Authentication endpoints
│   ├── models.py                # User model & MongoDB operations
│   ├── auth.py                  # JWT & OAuth logic
│   ├── client.py                # MongoDB connection
│   ├── run.py                   # Server entry point
│   └── seed.py                  # Database initialization
│
├── .env                         # Environment variables (DO NOT COMMIT)
├── .env.example                 # Environment template
├── .gitignore
└── README.md
```

### Directory Explanations

| Directory | Purpose |
|-----------|---------|
| `frontend/src/components/` | Reusable React components organized by function (globe, ui, common) |
| `frontend/src/pages/` | Full-page views; each represents a main section of the app |
| `frontend/src/stores/` | Zustand store modules for state management (globe, risk, lifeline, insights) |
| `frontend/src/api/` | API client wrappers for external services (insights, risk, lifeline, forecast) |
| `mongo/` | FastAPI backend handling authentication, JWT tokens, and Google OAuth |

---

## Running Locally

### Development Mode

**Terminal 1 — Backend (http://localhost:5000)**
```bash
cd /path/to/HackX
source .venv/bin/activate
python mongo/run.py
```

**Terminal 2 — Frontend (http://localhost:5173)**
```bash
cd /path/to/HackX/frontend
npm run dev
```

Open http://localhost:5173 in your browser.

### Production Build

```bash
cd frontend
npm run build         # Creates optimized dist/ folder
# Deploy dist/ to a static hosting service (Vercel, Netlify, CloudFlare, etc.)

# For backend, use a production ASGI server:
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker mongo.app:app --bind 0.0.0.0:8000
```

---

## Technology Stack

### Frontend
| Layer | Technology |
|-------|-----------|
| **Framework** | React 18 |
| **Build** | Vite 5 |
| **Styling** | Tailwind CSS 3 |
| **3D Visualization** | Cesium.js (satellite imagery, vector layers) |
| **State Management** | Zustand |
| **Animations** | Framer Motion |
| **Charts** | Recharts |
| **Routing** | URL hash-based (custom) |
| **HTTP Client** | Fetch API |
| **Reporting** | jsPDF (export to PDF) |

### Backend
| Layer | Technology |
|-------|-----------|
| **Framework** | FastAPI (Python 3.11) |
| **Server** | Uvicorn |
| **Database** | MongoDB (Atlas) |
| **Authentication** | JWT + Google OAuth 2.0 |
| **Hashing** | bcrypt |
| **Environment** | python-dotenv |

### External Services (Microservices)
| Service | Purpose | Base URL |
|---------|---------|----------|
| **Insights API** | Flood detection & change analysis | Configurable via `VITE_INSIGHTS_API_URL` |
| **Risk API** | District-level risk scoring | Configurable via `VITE_RISK_API_URL` |
| **Lifeline API** | Infrastructure exposure analysis | Configurable via `VITE_LIFELINE_API_URL` |
| **Forecast API** | Flood forecasting | Configurable via `VITE_FORECAST_API_URL` |

---

## Core Workflows

### 1. User Authentication
1. User lands on `/login`
2. Chooses "Email & Password" or "Sign in with Google"
3. If Google: redirected to Google OAuth consent screen
4. Backend exchanges code for tokens, creates/finds user in MongoDB
5. Returns JWT token to frontend
6. Frontend stores token, uses it for API requests via `Authorization: Bearer <token>` header

### 2. Flood Detection Analysis
1. User navigates to **Globe Analysis** page
2. Fills **RegionForm**: selects location (via Nominatim geocoding), confirms region boundary on globe
3. Clicks **"Start Analysis"** → triggers `POST /analyze` on Insights API with location + date range
4. **ProgressOverlay** shows pipeline stages (queued → preprocessing → detecting → scoring → completed)
5. Results render on **ResultsPanel**: flood area (km²), affected patches, risk classification
6. User can switch to **Risk Dashboard** or **Lifeline Panel** for deeper analysis

### 3. Viewing Historical Insights
1. User navigates to **Flood Insights** page
2. System fetches historical analysis runs via `GET /runs` from Insights API
3. User selects a run from the list
4. Detailed view shows:
   - Key metrics (flood area, percentage, severity, risk)
   - Change detection imagery (SAR before/after)
   - Per-patch flood statistics (area, location, risk)
   - AI-generated insights (Gemini analysis)
   - Location trend chart (if 3+ runs exist for that location)

### 4. Risk Assessment
1. From **Globe Analysis**, user switches to **Risk** tab
2. Triggers `POST /analyze/risk` on Risk API
3. Receives district-level summaries: flood coverage %, exposed population, risk labels
4. **RiskDashboardPanel** visualizes on globe with color-coded districts (red=critical, yellow=medium, etc.)

### 5. Lifeline Infrastructure Analysis
1. From **Globe Analysis**, user switches to **Lifeline** tab
2. Triggers `POST /flood-infrastructure` on Lifeline API
3. Receives list of affected infrastructure (hospitals, power plants, water systems)
4. **LifelinePanel** lists impact with severity ratings

---

## API Documentation

### Authentication Endpoints

**POST `/auth/login`**
- **Body**: `{ "email": "user@example.com", "password": "..." }`
- **Response**: `{ "token": "jwt...", "user": { ... } }`
- **Purpose**: Local email/password login

**GET `/auth/google`**
- **Purpose**: Redirect to Google OAuth consent screen

**GET `/auth/google/callback`**
- **Query**: `code=...&state=...` (from Google)
- **Response**: Redirect to frontend with `?token=jwt...`
- **Purpose**: Handle Google OAuth callback

**POST `/auth/logout`**
- **Headers**: `Authorization: Bearer <token>`
- **Response**: `{ "message": "Logged out successfully" }`
- **Purpose**: Invalidate JWT token

**GET `/auth/me`**
- **Headers**: `Authorization: Bearer <token>`
- **Response**: `{ "user": { id, email, subscription_level, auth_provider, created_at } }`
- **Purpose**: Fetch current authenticated user

**POST `/auth/refresh-token`**
- **Headers**: `Authorization: Bearer <token>`
- **Response**: `{ "token": "new_jwt...", "user": { ... } }`
- **Purpose**: Refresh an expiring token

### Insights Analysis Endpoints

> **Note**: These endpoints are provided by external Insights API service, not the local FastAPI backend.

**POST `/analyze`**
- **Host**: `VITE_INSIGHTS_API_URL`
- **Body**:
  ```json
  {
    "location": "Dhaka, Bangladesh",
    "pre_start": "2024-10-01",
    "pre_end": "2024-11-01",
    "post_start": "2024-11-01",
    "post_end": "2024-11-15",
    "threshold": -1.25
  }
  ```
- **Response**:
  ```json
  {
    "run_id": "RUN-20241105-BD-001",
    "location_name": "Dhaka, Bangladesh",
    "flood_area_km2": 1234.56,
    "flood_percentage": 8.5,
    "severity": "High",
    "risk_label": "High",
    "total_patches": 45,
    "patches": [
      {
        "patch_id": 1,
        "area_km2": 123.45,
        "centroid_lat": 23.8,
        "centroid_lon": 90.4,
        "risk_label": "High"
      }
    ],
    "ai_insight": "SAR analysis detected significant inundation...",
    "panel_png_path": "s3://...",
    "aoi_bbox": [90.1, 23.5, 91.0, 24.0],
    "resolution_m": 10,
    "timestamp": "2024-11-05T10:30:00Z"
  }
  ```

**GET `/runs`**
- **Host**: `VITE_INSIGHTS_API_URL`
- **Response**: `{ "runs": [ { run_id, location_name, flood_area_km2, ... }, ... ] }`
- **Purpose**: Fetch historical analysis runs

**GET `/runs/{run_id}`**
- **Host**: `VITE_INSIGHTS_API_URL`
- **Response**: Full run details (same as analyze response)
- **Purpose**: Fetch detailed results for a specific run

---

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Commit** with clear messages: `git commit -m "Add feature: description"`
4. **Push** to your branch: `git push origin feature/your-feature-name`
5. **Open a Pull Request** with a description of changes

### Code Standards
- **Frontend**: ESLint-compliant React/JSX, consistent with existing component patterns
- **Backend**: PEP 8 Python, use type hints, include docstrings
- **Git**: Use conventional commit messages (`feat:`, `fix:`, `docs:`, etc.)

---

## Troubleshooting

### "MONGO_DB_CONNECTION_STRING not found"
- Ensure `.env` file exists in project root with `MONGO_DB_CONNECTION_STRING=...`
- Check MongoDB Atlas connection string is correct

### "Cesium Ion token not configured"
- Optional; app works without it (uses bundled NaturalEarthII tiles)
- To enable satellite imagery, register at https://cesium.com/ion and add token to `.env`

### Frontend not connecting to backend
- Ensure backend is running: `python mongo/run.py`
- Check `VITE_API_URL` in `frontend/.env` matches backend address
- Check CORS configuration in `mongo/app.py` includes frontend URL

### "Failed to fetch runs"
- Ensure `VITE_INSIGHTS_API_URL` is set and pointing to a running Insights API service
- Check network tab in browser DevTools for detailed error

---

## License

MIT License — see LICENSE file for details.

---

## Support

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Documentation**: See `FEATURES.md`, `SYSTEM_ARCHITECTURE.md`, `IMPLEMENTATION.md`
- **Questions**: Reach out via GitHub Discussions

---

## Acknowledgments

Built to address **Problem Statement 6: "Satellite Data to Insight Engine for Climate Risk"** by COSMEON. Leverages open data from:
- **Sentinel-1/Sentinel-2**: Copernicus program (ESA)
- **Landsat**: USGS Earth Explorer
- **Population data**: WorldPop project
- **Administrative boundaries**: GADM

Thank you to all contributors and the open-source geospatial community.
