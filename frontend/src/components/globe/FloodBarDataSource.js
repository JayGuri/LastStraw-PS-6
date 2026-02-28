import * as Cesium from 'cesium'

/**
 * FloodBarDataSource — Renders vertical bars on the Cesium globe.
 *
 * Data format (from API grid_points):
 * [
 *   ["flood_depth",  [lat, lon, value, lat, lon, value, ...]],
 *   ["pop_density",  [lat, lon, value, lat, lon, value, ...]],
 *   ["risk_score",   [lat, lon, value, lat, lon, value, ...]]
 * ]
 *
 * Each triplet (lat, lon, value) produces a vertical polyline bar.
 * Height = value * heightScale.  Color = lerp based on normalized value.
 */

// COSMEON risk color stops
const COLOR_STOPS = [
  { t: 0.0,  color: new Cesium.Color(0.22, 0.63, 0.35, 1.0) },  // #38a058 (low)
  { t: 0.33, color: new Cesium.Color(0.78, 0.63, 0.09, 1.0) },  // #c8a018 (medium)
  { t: 0.66, color: new Cesium.Color(0.82, 0.41, 0.16, 1.0) },  // #d06828 (high)
  { t: 1.0,  color: new Cesium.Color(0.85, 0.25, 0.25, 1.0) },  // #d84040 (critical)
]

function lerpColor(t) {
  const clamped = Math.max(0, Math.min(1, t))
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const a = COLOR_STOPS[i]
    const b = COLOR_STOPS[i + 1]
    if (clamped >= a.t && clamped <= b.t) {
      const f = (clamped - a.t) / (b.t - a.t)
      return Cesium.Color.lerp(a.color, b.color, f, new Cesium.Color())
    }
  }
  return COLOR_STOPS[COLOR_STOPS.length - 1].color.clone()
}

// Height scaling: value * scale = meters above ground
const HEIGHT_SCALES = {
  flood_depth: 50000,
  pop_density: 0.05,
  risk_score:  5000,
}

// Max value per metric for normalization (0..1)
const MAX_VALUES = {
  flood_depth: 5.0,
  pop_density: 10000,
  risk_score:  100,
}

export class FloodBarDataSource {
  constructor(name) {
    this._name = name
    this._entityCollection = new Cesium.EntityCollection()
    this._changed = new Cesium.Event()
    this._error = new Cesium.Event()
    this._loading = new Cesium.Event()
    this._isLoading = false
    this._entityCluster = new Cesium.EntityCluster()
  }

  get name() { return this._name }
  get clock() { return undefined }
  get entities() { return this._entityCollection }
  get isLoading() { return this._isLoading }
  get changedEvent() { return this._changed }
  get errorEvent() { return this._error }
  get loadingEvent() { return this._loading }

  get show() { return this._entityCollection.show }
  set show(v) { this._entityCollection.show = v }

  get clustering() { return this._entityCluster }
  set clustering(v) { this._entityCluster = v }

  /**
   * Load from the API grid_points response.
   * @param {Array} gridPoints  [["series_name", [lat, lon, val, ...]], ...]
   * @param {string} metric     which series to render
   */
  loadFromFloodResponse(gridPoints, metric = 'flood_depth') {
    this._setLoading(true)
    const entities = this._entityCollection
    entities.suspendEvents()
    entities.removeAll()

    const series = gridPoints.find(s => s[0] === metric)
    if (!series) {
      entities.resumeEvents()
      this._setLoading(false)
      return
    }

    const [, values] = series
    const heightScale = HEIGHT_SCALES[metric] ?? 50000
    const maxVal = MAX_VALUES[metric] ?? 1

    for (let i = 0; i < values.length; i += 3) {
      const lat = values[i]
      const lon = values[i + 1]
      const val = values[i + 2]

      if (val <= 0) continue

      const height = val * heightScale
      const normalized = val / maxVal
      const color = lerpColor(normalized)

      const bottom = Cesium.Cartesian3.fromDegrees(lon, lat, 0)
      const top = Cesium.Cartesian3.fromDegrees(lon, lat, height)

      const polyline = new Cesium.PolylineGraphics()
      polyline.material = new Cesium.ColorMaterialProperty(color)
      polyline.width = new Cesium.ConstantProperty(3)
      polyline.arcType = new Cesium.ConstantProperty(Cesium.ArcType.NONE)
      polyline.positions = new Cesium.ConstantProperty([bottom, top])

      entities.add(new Cesium.Entity({
        id: `${metric}_${i}`,
        polyline,
      }))
    }

    entities.resumeEvents()
    this._changed.raiseEvent(this)
    this._setLoading(false)
  }

  _setLoading(isLoading) {
    if (this._isLoading !== isLoading) {
      this._isLoading = isLoading
      this._loading.raiseEvent(this, isLoading)
    }
  }

  update() { return true }
}
