export const BASE_URL = 'https://api.cosmeon.io/api/v1'

export const ENDPOINTS = [
  {
    id: 'submit',
    method: 'POST',
    path: '/runs',
    summary: 'Submit Analysis Run',
    description: 'Queue a new flood-detection pipeline run. Returns immediately with a run_id; poll GET /runs/{run_id} for status.',
    statusCode: 202,
    statusText: 'Accepted',
    params: [
      { name: 'aoi_polygon', type: 'GeoJSON Polygon', required: true,  desc: 'AOI boundary. Max 5° × 5°.' },
      { name: 'start_date',  type: 'ISO 8601 date',   required: true,  desc: 'Event window start (YYYY-MM-DD)' },
      { name: 'end_date',    type: 'ISO 8601 date',   required: true,  desc: 'Event window end. Max range 30 days.' },
      { name: 'sensor',      type: 'string',           required: false, desc: 'S1_GRD | S2_L2A | LANDSAT_8 (default: S1_GRD)' },
      { name: 'detector',    type: 'string',           required: false, desc: 'otsu | rf | unet (default: otsu)' },
      { name: 'gadm_level',  type: 'integer',          required: false, desc: '1=country 2=province 3=district (default: 2)' },
    ],
    curlExample: `curl -X POST https://api.cosmeon.io/api/v1/runs \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your_key_here" \\
  -d '{
    "aoi_polygon": {
      "type": "Polygon",
      "coordinates": [[[88.0,21.5],[92.7,21.5],[92.7,26.6],[88.0,26.6],[88.0,21.5]]]
    },
    "start_date": "2024-10-25",
    "end_date":   "2024-11-04",
    "sensor":     "S1_GRD",
    "detector":   "unet"
  }'`,
    pythonExample: `import httpx, json

client = httpx.Client(base_url="https://api.cosmeon.io/api/v1",
                      headers={"X-API-Key": "your_key_here"})

payload = {
    "aoi_polygon": {
        "type": "Polygon",
        "coordinates": [[[88.0,21.5],[92.7,21.5],
                          [92.7,26.6],[88.0,26.6],[88.0,21.5]]]
    },
    "start_date": "2024-10-25",
    "end_date":   "2024-11-04",
    "sensor":     "S1_GRD",
    "detector":   "unet",
}

r = client.post("/runs", json=payload)
print(r.json()["run_id"])`,
    response: {
      run_id: "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      status: "queued",
      sensor: "S1_GRD",
      detector: "unet",
      aoi_id: "aoi_bd_delta_001",
      created_at: "2024-11-05T02:14:07.331Z",
      eta_minutes: 35
    }
  },
  {
    id: 'status',
    method: 'GET',
    path: '/runs/{run_id}',
    summary: 'Get Run Status',
    description: 'Poll pipeline execution state. Status progresses: queued → preprocessing → detecting → scoring → completed | failed.',
    statusCode: 200,
    statusText: 'OK',
    params: [
      { name: 'run_id', type: 'UUID path', required: true, desc: 'UUID returned by POST /runs' },
    ],
    curlExample: `curl https://api.cosmeon.io/api/v1/runs/3fa85f64 \\
  -H "X-API-Key: your_key_here"`,
    pythonExample: `import time

run_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"

while True:
    r = client.get(f"/runs/{run_id}")
    data = r.json()
    print(data["status"])
    if data["status"] in ("completed", "failed"):
        break
    time.sleep(10)`,
    response: {
      run_id: "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      status: "completed",
      scene_id: "S1_GRD_20241104T001023_IW_GRDH",
      started_at: "2024-11-05T02:14:09.112Z",
      completed_at: "2024-11-05T02:52:21.890Z",
      duration_s: 2292,
      districts_count: 64,
      flood_area_km2: 18400.2,
      confidence_avg: 0.874,
      geotiff_url: "https://s3.amazonaws.com/cosmeon/scenes/BD-DELTA-01/RUN-BD-001/flood_mask.tif?X-Amz-Expires=900"
    }
  },
  {
    id: 'districts',
    method: 'GET',
    path: '/floods/{run_id}/districts',
    summary: 'District Risk Results',
    description: 'Paginated per-district flood statistics for a completed run. Filter by risk_level, sort by flood_pct or pop_exposed.',
    statusCode: 200,
    statusText: 'OK',
    params: [
      { name: 'run_id',     type: 'UUID path', required: true,  desc: 'Completed run UUID' },
      { name: 'page',       type: 'integer',   required: false, desc: 'Page number (default: 1)' },
      { name: 'page_size',  type: 'integer',   required: false, desc: 'Max 100 (default: 20)' },
      { name: 'risk_level', type: 'string',    required: false, desc: 'Low | Medium | High | Critical' },
      { name: 'sort_by',    type: 'string',    required: false, desc: 'flood_pct | pop_exposed | gadm_id' },
    ],
    curlExample: `curl "https://api.cosmeon.io/api/v1/floods/3fa85f64/districts?risk_level=Critical&sort_by=pop_exposed" \\
  -H "X-API-Key: your_key_here"`,
    pythonExample: `r = client.get(
    f"/floods/{run_id}/districts",
    params={"risk_level": "Critical", "sort_by": "pop_exposed"}
)
for d in r.json()["districts"]:
    print(d["district_name"], d["flood_pct"], d["risk_level"])`,
    response: {
      run_id: "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      page: 1, page_size: 20, total: 64,
      districts: [
        { gadm_id: "BGD.11.1", district_name: "Sylhet",    flood_pct: 67.2, flood_area_km2: 2319.4, pop_exposed: 842000,  risk_level: "Critical", risk_score: 94, confidence: 0.91 },
        { gadm_id: "BGD.11.2", district_name: "Sunamganj", flood_pct: 54.1, flood_area_km2: 1537.0, pop_exposed: 620000,  risk_level: "Critical", risk_score: 88, confidence: 0.88 },
        { gadm_id: "BGD.11.3", district_name: "Netrokona", flood_pct: 48.3, flood_area_km2:  956.3, pop_exposed: 440000,  risk_level: "Critical", risk_score: 81, confidence: 0.85 },
      ]
    }
  },
  {
    id: 'health',
    method: 'GET',
    path: '/health',
    summary: 'System Health Check',
    description: 'Returns service connectivity status for all subsystems: database, Redis broker, Celery workers.',
    statusCode: 200,
    statusText: 'OK',
    params: [],
    curlExample: `curl https://api.cosmeon.io/api/v1/health`,
    pythonExample: `r = httpx.get("https://api.cosmeon.io/api/v1/health")
assert r.json()["status"] == "ok"`,
    response: {
      status: "ok",
      version: "1.0.0",
      database: "connected",
      redis: "connected",
      workers: 4,
      timestamp: "2024-11-05T03:01:22.004Z"
    }
  },
]

export const ERROR_CODES = [
  { code: 'AOI_TOO_LARGE',      status: 400, desc: 'AOI exceeds 5° × 5° bounding box limit' },
  { code: 'DATE_RANGE_INVALID', status: 400, desc: 'Range > 30 days or end_date < start_date' },
  { code: 'NO_SCENES_FOUND',    status: 404, desc: 'No Sentinel-1 scenes found for AOI + date' },
  { code: 'RUN_NOT_FOUND',      status: 404, desc: 'run_id does not exist in database' },
  { code: 'RUN_NOT_COMPLETE',   status: 409, desc: 'Results unavailable — pipeline not finished' },
  { code: 'PIPELINE_FAILED',    status: 500, desc: 'Internal pipeline error, logged to Sentry' },
]
