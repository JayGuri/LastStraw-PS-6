import * as Cesium from 'cesium'

/**
 * FloodBarDataSource — Renders glowing vertical bars on the Cesium globe.
 *
 * Data format (from API grid_points):
 *   [["flood_depth",  [lat, lon, value, ...]],
 *    ["pop_density",  [lat, lon, value, ...]],
 *    ["risk_score",   [lat, lon, value, ...]]]
 *
 * Each bar = a PolylineGlow entity (shaft) + a Point entity (cap dot).
 */

// COSMEON risk color stops (low → critical)
const COLOR_STOPS = [
  { t: 0.0,  r: 0.22, g: 0.63, b: 0.35 },  // #38a058 low-green
  { t: 0.33, r: 0.78, g: 0.63, b: 0.09 },  // #c8a018 medium-amber
  { t: 0.66, r: 0.82, g: 0.41, b: 0.16 },  // #d06828 high-orange
  { t: 1.0,  r: 0.85, g: 0.25, b: 0.25 },  // #d84040 critical-red
]

function lerpColor(t) {
  const clamped = Math.max(0, Math.min(1, t))
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const a = COLOR_STOPS[i]
    const b = COLOR_STOPS[i + 1]
    if (clamped >= a.t && clamped <= b.t) {
      const f = (clamped - a.t) / (b.t - a.t)
      return new Cesium.Color(
        a.r + f * (b.r - a.r),
        a.g + f * (b.g - a.g),
        a.b + f * (b.b - a.b),
        1.0
      )
    }
  }
  const last = COLOR_STOPS[COLOR_STOPS.length - 1]
  return new Cesium.Color(last.r, last.g, last.b, 1.0)
}

// Height scaling per metric — tuned so bars look meaningful at city zoom (~50-200km alt)
const HEIGHT_SCALES = {
  flood_depth: 35000,    // 1 m depth  → 35 km bar
  pop_density: 0.04,     // 1 k/km²   → 40 m  (scaled via maxVal)
  risk_score:  3000,     // score 100  → 300 km bar
}

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
    this._error   = new Cesium.Event()
    this._loading = new Cesium.Event()
    this._isLoading = false
    this._entityCluster = new Cesium.EntityCluster()
  }

  get name()         { return this._name }
  get clock()        { return undefined }
  get entities()     { return this._entityCollection }
  get isLoading()    { return this._isLoading }
  get changedEvent() { return this._changed }
  get errorEvent()   { return this._error }
  get loadingEvent() { return this._loading }
  get show()         { return this._entityCollection.show }
  set show(v)        { this._entityCollection.show = v }
  get clustering()   { return this._entityCluster }
  set clustering(v)  { this._entityCluster = v }

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
    const heightScale = HEIGHT_SCALES[metric] ?? 35000
    const maxVal      = MAX_VALUES[metric] ?? 1

    for (let i = 0; i < values.length; i += 3) {
      const lat = values[i]
      const lon = values[i + 1]
      const val = values[i + 2]
      if (val <= 0) continue

      const height     = val * heightScale
      const normalized = val / maxVal
      const color      = lerpColor(normalized)
      const glowColor  = color.withAlpha(1.0)

      const bottom = Cesium.Cartesian3.fromDegrees(lon, lat, 0)
      const top    = Cesium.Cartesian3.fromDegrees(lon, lat, height)

      // ── Glowing polyline shaft ──
      const polyline = new Cesium.PolylineGraphics()
      polyline.positions = new Cesium.ConstantProperty([bottom, top])
      polyline.width     = new Cesium.ConstantProperty(6)
      polyline.arcType   = new Cesium.ConstantProperty(Cesium.ArcType.NONE)
      polyline.material  = new Cesium.PolylineGlowMaterialProperty({
        glowPower:  0.4,
        taperPower: 0.7,
        color:      glowColor,
      })

      entities.add(new Cesium.Entity({
        id:       `${metric}_shaft_${i}`,
        polyline,
      }))

      // ── Cap dot at top of bar ──
      entities.add(new Cesium.Entity({
        id:       `${metric}_cap_${i}`,
        position: top,
        point: {
          pixelSize:    6,
          color:        Cesium.Color.WHITE.withAlpha(0.9),
          outlineColor: glowColor,
          outlineWidth: 3,
          disableDepthTestDistance: 5000000,
        },
      }))

      // ── Base glow disk (wide, very transparent) ──
      entities.add(new Cesium.Entity({
        id:       `${metric}_base_${i}`,
        position: bottom,
        point: {
          pixelSize:       12,
          color:           glowColor.withAlpha(0.12),
          outlineColor:    glowColor.withAlpha(0.35),
          outlineWidth:    4,
          heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      }))
    }

    entities.resumeEvents()
    this._changed.raiseEvent(this)
    this._setLoading(false)
  }

  _setLoading(v) {
    if (this._isLoading !== v) {
      this._isLoading = v
      this._loading.raiseEvent(this, v)
    }
  }

  update() { return true }
}
