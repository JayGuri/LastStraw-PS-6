// All 20 Bangladesh districts with real data from the project
export const DISTRICTS = [
  { id: 'BGD.11.1', name: 'Sylhet',     floodPct: 67.2, pop: 842000,   area: 3452, conf: 0.91, risk: 'Critical', col: 4, row: 0 },
  { id: 'BGD.11.2', name: 'Sunamganj',  floodPct: 54.1, pop: 620000,   area: 2841, conf: 0.88, risk: 'Critical', col: 3, row: 0 },
  { id: 'BGD.11.3', name: 'Netrokona',  floodPct: 48.3, pop: 440000,   area: 1980, conf: 0.85, risk: 'Critical', col: 3, row: 1 },
  { id: 'BGD.4.1',  name: 'Mymensingh', floodPct: 38.7, pop: 1100000,  area: 1640, conf: 0.85, risk: 'High',     col: 4, row: 1 },
  { id: 'BGD.3.1',  name: 'Jamalpur',   floodPct: 36.2, pop: 380000,   area: 1380, conf: 0.83, risk: 'High',     col: 2, row: 1 },
  { id: 'BGD.7.1',  name: 'Sirajganj',  floodPct: 31.4, pop: 520000,   area: 1190, conf: 0.86, risk: 'High',     col: 2, row: 2 },
  { id: 'BGD.1.1',  name: 'Bogura',     floodPct: 24.8, pop: 480000,   area: 940,  conf: 0.82, risk: 'High',     col: 1, row: 1 },
  { id: 'BGD.2.1',  name: 'Kurigram',   floodPct: 22.1, pop: 290000,   area: 860,  conf: 0.80, risk: 'High',     col: 2, row: 0 },
  { id: 'BGD.5.1',  name: 'Gaibandha',  floodPct: 19.6, pop: 310000,   area: 740,  conf: 0.81, risk: 'Medium',   col: 1, row: 0 },
  { id: 'BGD.6.1',  name: 'Dhaka',      floodPct: 14.3, pop: 9200000,  area: 1360, conf: 0.87, risk: 'Medium',   col: 4, row: 2 },
  { id: 'BGD.8.1',  name: 'Manikganj',  floodPct: 12.8, pop: 260000,   area: 490,  conf: 0.84, risk: 'Medium',   col: 3, row: 2 },
  { id: 'BGD.9.1',  name: 'Faridpur',   floodPct: 11.2, pop: 340000,   area: 420,  conf: 0.82, risk: 'Medium',   col: 2, row: 3 },
  { id: 'BGD.10.1', name: 'Tangail',    floodPct:  9.4, pop: 430000,   area: 355,  conf: 0.83, risk: 'Medium',   col: 3, row: 1 },
  { id: 'BGD.12.1', name: 'Rajshahi',   floodPct:  4.2, pop: 620000,   area: 158,  conf: 0.90, risk: 'Low',      col: 0, row: 2 },
  { id: 'BGD.13.1', name: 'Pabna',      floodPct:  3.8, pop: 380000,   area: 142,  conf: 0.88, risk: 'Low',      col: 0, row: 1 },
  { id: 'BGD.14.1', name: 'Natore',     floodPct:  2.9, pop: 320000,   area: 110,  conf: 0.89, risk: 'Low',      col: 1, row: 2 },
  { id: 'BGD.15.1', name: 'Khulna',     floodPct:  1.4, pop: 680000,   area: 52,   conf: 0.90, risk: 'None',     col: 1, row: 4 },
  { id: 'BGD.16.1', name: 'Barishal',   floodPct:  2.1, pop: 510000,   area: 80,   conf: 0.91, risk: 'None',     col: 3, row: 4 },
  { id: 'BGD.17.1', name: 'Jashore',    floodPct:  1.8, pop: 340000,   area: 68,   conf: 0.88, risk: 'None',     col: 0, row: 3 },
  { id: 'BGD.18.1', name: 'Chattogram', floodPct:  0.9, pop: 3400000,  area: 34,   conf: 0.93, risk: 'None',     col: 4, row: 4 },
]

export const RISK_COLORS = {
  Critical: { hex: '#d84040', dim: 'rgba(216,64,64,0.15)',  label: 'Critical', ring: true  },
  High:     { hex: '#d06828', dim: 'rgba(208,104,40,0.15)', label: 'High',     ring: false },
  Medium:   { hex: '#c8a018', dim: 'rgba(200,160,24,0.15)', label: 'Medium',   ring: false },
  Low:      { hex: '#38a058', dim: 'rgba(56,160,88,0.15)',  label: 'Low',      ring: false },
  None:     { hex: '#2a3f58', dim: 'rgba(42,63,88,0.3)',    label: 'No Risk',  ring: false },
}

export const RUN_META = {
  runId:       'RUN-20241105-BD-001',
  sceneId:     'S1_GRD_20241104T001023',
  aoiId:       'BD-DELTA-01',
  sceneDate:   '2024-11-04',
  aoiArea:     144000,
  districts:   64,
  duration:    38,
  confidence:  87.4,
  criticalDist: 8,
  highDist:     12,
  mediumDist:   19,
  popExposed:   4.2,
  floodedArea:  18400,
}

// Generate mock historical time-series per district
export function getDistrictHistory(districtId) {
  const base = DISTRICTS.find(d => d.id === districtId)
  if (!base) return []
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
  return months.map((m, i) => {
    const seasonal = Math.sin((i / 12) * Math.PI * 2 - 1) * 0.5 + 0.5
    const noise = (Math.random() - 0.5) * 0.2
    const pct = Math.max(0, base.floodPct * (0.3 + seasonal * 0.7 + noise))
    const expPop = Math.round(base.pop * (pct / 100) * 0.8)
    return { month: m, floodPct: +pct.toFixed(1), population: expPop }
  })
}
