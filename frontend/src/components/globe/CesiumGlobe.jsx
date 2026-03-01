import React, { useEffect, useRef } from 'react'
import * as Cesium from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'
import { useGlobeStore } from '../../stores/globeStore.js'

const SEVERITY_CFG = {
  critical: {
    fill:    new Cesium.Color(0.85, 0.25, 0.25, 0.55),
    outline: new Cesium.Color(0.85, 0.25, 0.25, 1.0),
    extrude: 3500,
  },
  high: {
    fill:    new Cesium.Color(0.82, 0.41, 0.16, 0.50),
    outline: new Cesium.Color(0.82, 0.41, 0.16, 1.0),
    extrude: 2500,
  },
  medium: {
    fill:    new Cesium.Color(0.78, 0.63, 0.09, 0.42),
    outline: new Cesium.Color(0.78, 0.63, 0.09, 1.0),
    extrude: 1500,
  },
  low: {
    fill:    new Cesium.Color(0.22, 0.63, 0.35, 0.35),
    outline: new Cesium.Color(0.22, 0.63, 0.35, 1.0),
    extrude: 600,
  },
}

export default function CesiumGlobe() {
  const containerRef   = useRef(null)
  const viewerRef      = useRef(null)
  const isRotatingRef  = useRef(true)

  const geocoded = useGlobeStore(s => s.geocoded)
  const result   = useGlobeStore(s => s.result)

  // ──────────────────────────────────────────────────────────
  // 1. INIT VIEWER
  // ──────────────────────────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current
    if (!container || viewerRef.current) return

    let cancelled = false
    let resizeObs = null
    let stopSpin  = () => {}

    async function initViewer() {
      if (viewerRef.current || cancelled) return
      const rect = container.getBoundingClientRect()
      if (rect.width < 1 || rect.height < 1) return

      // ── Cesium Ion token (optional — enables Bing Maps satellite) ──
      const token = import.meta.env.VITE_CESIUM_TOKEN || ''
      const hasToken = token.length > 10 && token !== 'your_cesium_ion_token_here'
      if (hasToken) Cesium.Ion.defaultAccessToken = token

      // ── Build the base imagery layer ──
      // Cesium 1.104+ uses ImageryLayer.fromWorldImagery / baseLayer instead
      // of the deprecated createWorldImagery / imageryProvider constructor opts.
      let baseLayer = false
      if (hasToken) {
        try {
          baseLayer = Cesium.ImageryLayer.fromWorldImagery({
            style: Cesium.IonWorldImageryStyle.AERIAL_WITH_LABELS,
          })
        } catch { baseLayer = false }
      }

      // If no Ion token or Ion failed, use the bundled NaturalEarthII tiles
      if (!baseLayer) {
        try {
          const provider = await Cesium.TileMapServiceImageryProvider.fromUrl(
            Cesium.buildModuleUrl('Assets/Textures/NaturalEarthII'),
          )
          if (cancelled) return
          baseLayer = new Cesium.ImageryLayer(provider)
        } catch {
          baseLayer = false
        }
      }
      if (cancelled) return

      const viewer = new Cesium.Viewer(container, {
        baseLayer,
        baseLayerPicker:      false,
        geocoder:             false,
        homeButton:           false,
        sceneModePicker:      false,
        navigationHelpButton: false,
        animation:            false,
        timeline:             false,
        fullscreenButton:     false,
        selectionIndicator:   false,
        infoBox:              false,
        creditContainer:      (() => {
          const el = document.createElement('div')
          el.style.display = 'none'
          return el
        })(),
        scene3DOnly: true,
        orderIndependentTranslucency: false,
      })

      if (cancelled) { viewer.destroy(); return }

      const scene = viewer.scene
      const globe = scene.globe

      // ── Space background ──
      scene.backgroundColor = new Cesium.Color(0.027, 0.035, 0.055, 1.0)
      scene.moon = new Cesium.Moon({ show: false })

      // ── Globe appearance ──
      globe.show                    = true
      globe.showWaterEffect         = true
      globe.enableLighting          = true
      globe.dynamicAtmosphereLighting        = true
      globe.dynamicAtmosphereLightingFromSun = true
      globe.atmosphereLightIntensity         = 8.0
      globe.atmosphereRayleighCoefficient    = new Cesium.Cartesian3(5.5e-6, 13.0e-6, 28.4e-6)
      globe.atmosphereMieCoefficient         = new Cesium.Cartesian3(21e-6, 21e-6, 21e-6)
      globe.nightFadeOutDistance   = 1e10
      globe.nightFadeInDistance    = 5e8
      globe.translucency.enabled   = false

      // ── Atmosphere glow ──
      scene.skyAtmosphere.show = true
      scene.skyAtmosphere.atmosphereLightIntensity        = 25.0
      scene.skyAtmosphere.atmosphereRayleighCoefficient   = new Cesium.Cartesian3(5.5e-6, 13.0e-6, 28.4e-6)
      scene.skyAtmosphere.atmosphereMieCoefficient        = new Cesium.Cartesian3(21e-6, 21e-6, 21e-6)
      scene.skyAtmosphere.atmosphereMieAnisotropy         = 0.9
      scene.skyAtmosphere.hueShift        =  0.0
      scene.skyAtmosphere.saturationShift =  0.0
      scene.skyAtmosphere.brightnessShift = -0.05

      // ── Star field ──
      scene.skyBox = new Cesium.SkyBox({
        sources: {
          positiveX: Cesium.buildModuleUrl('Assets/Textures/SkyBox/tycho2t3_80_px.jpg'),
          negativeX: Cesium.buildModuleUrl('Assets/Textures/SkyBox/tycho2t3_80_mx.jpg'),
          positiveY: Cesium.buildModuleUrl('Assets/Textures/SkyBox/tycho2t3_80_py.jpg'),
          negativeY: Cesium.buildModuleUrl('Assets/Textures/SkyBox/tycho2t3_80_my.jpg'),
          positiveZ: Cesium.buildModuleUrl('Assets/Textures/SkyBox/tycho2t3_80_pz.jpg'),
          negativeZ: Cesium.buildModuleUrl('Assets/Textures/SkyBox/tycho2t3_80_mz.jpg'),
        },
      })

      // ── Clock: freeze at noon UTC so the lit side faces camera ──
      const now = new Date()
      now.setUTCHours(12, 0, 0, 0)
      viewer.clock.currentTime   = Cesium.JulianDate.fromDate(now)
      viewer.clock.shouldAnimate = false

      // ── Render quality ──
      scene.postProcessStages.fxaa.enabled = true
      viewer.resolutionScale = window.devicePixelRatio > 1 ? 1.5 : 1.0
      scene.highDynamicRange = false

      // ── Auto-rotation ──
      scene.preRender.addEventListener(() => {
        if (isRotatingRef.current) {
          scene.camera.rotate(Cesium.Cartesian3.UNIT_Z, -0.00012)
        }
      })

      // ── Initial camera ──
      viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(30, 15, 22_000_000),
        orientation: {
          heading: Cesium.Math.toRadians(0),
          pitch:   Cesium.Math.toRadians(-90),
          roll:    0,
        },
      })

      // ── Stop rotation on user interaction ──
      stopSpin = () => { isRotatingRef.current = false }
      scene.screenSpaceCameraController.inertiaSpin       = 0.9
      scene.screenSpaceCameraController.inertiaZoom        = 0.8
      scene.screenSpaceCameraController.inertiaTranslate   = 0.9
      viewer.camera.moveStart.addEventListener(stopSpin)

      viewerRef.current = viewer
    }

    // Defer one frame so the container is laid out, then use ResizeObserver as fallback
    const rafId = requestAnimationFrame(() => { initViewer() })
    resizeObs = new ResizeObserver(() => {
      if (!viewerRef.current) {
        initViewer()
      } else if (!viewerRef.current.isDestroyed()) {
        viewerRef.current.resize()
      }
    })
    resizeObs.observe(container)

    return () => {
      cancelled = true
      cancelAnimationFrame(rafId)
      resizeObs?.disconnect()
      const v = viewerRef.current
      if (v) {
        try { v.camera.moveStart.removeEventListener(stopSpin) } catch (_) {}
        if (!v.isDestroyed()) v.destroy()
        viewerRef.current = null
      }
    }
  }, [])

  // ──────────────────────────────────────────────────────────
  // 2. FLY TO GEOCODED REGION
  // ──────────────────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || viewer.isDestroyed()) return

    ;['region-boundary', 'region-pin'].forEach(name => {
      viewer.dataSources.getByName(name).forEach(ds => viewer.dataSources.remove(ds))
    })
    viewer.entities.removeById('region-center-pin')
    viewer.entities.removeById('region-boundary-bbox')

    if (!geocoded) {
      isRotatingRef.current = true
      return
    }

    isRotatingRef.current = false

    const { lat, lon, bbox } = geocoded
    const flyOptions = {
      duration: 2.2,
      easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
    }

    if (bbox?.length === 4) {
      viewer.camera.flyTo({
        ...flyOptions,
        destination: Cesium.Rectangle.fromDegrees(
          bbox[0] - 0.5, bbox[1] - 0.5,
          bbox[2] + 0.5, bbox[3] + 0.5,
        ),
      })
    } else {
      viewer.camera.flyTo({
        ...flyOptions,
        destination: Cesium.Cartesian3.fromDegrees(lon, lat, 800_000),
        orientation: { heading: 0, pitch: Cesium.Math.toRadians(-45), roll: 0 },
      })
    }

    // Green highlight: use actual boundary polygon only when Nominatim returns Polygon/MultiPolygon.
    // Otherwise Nominatim often returns a Point (single dot); in that case we show the bbox so the
    // region extent is always clear.
    const greenFill = Cesium.Color.fromCssColorString('#16a34a').withAlpha(0.35)
    const greenEdge = Cesium.Color.fromCssColorString('#22c55e')
    const hasAreaGeometry =
      geocoded.boundary_geojson &&
      ['Polygon', 'MultiPolygon'].includes(geocoded.boundary_geojson.type)

    if (hasAreaGeometry) {
      const boundaryDs = new Cesium.GeoJsonDataSource('region-boundary')
      boundaryDs
        .load(geocoded.boundary_geojson, {
          stroke:       greenEdge,
          strokeWidth:  2.5,
          fill:         greenFill,
          clampToGround: true,
        })
        .then(() => {
          boundaryDs.entities.values.forEach(entity => {
            if (entity.polyline) {
              entity.polyline.material = new Cesium.PolylineGlowMaterialProperty({
                glowPower:  0.5,
                taperPower: 1.0,
                color:      greenEdge,
              })
              entity.polyline.width              = 5
              entity.polyline.clampToGround      = true
              entity.polyline.classificationType = Cesium.ClassificationType.TERRAIN
            }
            if (entity.polygon) {
              entity.polygon.material            = greenFill
              entity.polygon.outline             = true
              entity.polygon.outlineColor        = greenEdge
              entity.polygon.outlineWidth        = 2.5
              entity.polygon.clampToGround       = true
              entity.polygon.classificationType  = Cesium.ClassificationType.TERRAIN
            }
          })
          if (!viewer.isDestroyed()) viewer.dataSources.add(boundaryDs)
        })
    }

    // Always show bbox when we don't have a proper polygon (Point/LineString/missing), so the
    // selected region is never just a dot. When we have a polygon, bbox is skipped.
    if (!hasAreaGeometry && geocoded.bbox?.length === 4) {
      const [w, s, e, n] = geocoded.bbox
      viewer.entities.add({
        id: 'region-boundary-bbox',
        rectangle: {
          coordinates: Cesium.Rectangle.fromDegrees(w, s, e, n),
          material:    greenFill,
          outline:     true,
          outlineColor: greenEdge,
          outlineWidth: 2.5,
        },
      })
    }

    // Small center pin for reference (secondary to the polygon/bbox)
    viewer.entities.add({
      id:       'region-center-pin',
      position: Cesium.Cartesian3.fromDegrees(lon, lat, 0),
      point: {
        pixelSize:    6,
        color:        greenEdge,
        outlineColor: Cesium.Color.fromCssColorString('#16a34a').withAlpha(0.5),
        outlineWidth: 4,
        heightReference:          Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    })
  }, [geocoded])

  // ──────────────────────────────────────────────────────────
  // 3. RENDER FORECAST RESULT
  // ──────────────────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || viewer.isDestroyed()) return

    // Clean up old result entities
    ;['flood-zones', 'zone-markers'].forEach(name => {
      viewer.dataSources.getByName(name).forEach(ds => viewer.dataSources.remove(ds))
    })
    viewer.entities.removeById('forecast-pin')

    if (!result) return

    // Map alert level to color
    const ALERT_CESIUM = {
      LOW:      Cesium.Color.fromCssColorString('#22c55e'),
      MEDIUM:   Cesium.Color.fromCssColorString('#c9a96e'),
      HIGH:     Cesium.Color.fromCssColorString('#dc7828'),
      CRITICAL: Cesium.Color.fromCssColorString('#c0392b'),
    }

    const alertColor = ALERT_CESIUM[result.alert_level] ?? ALERT_CESIUM.MEDIUM

    // Render forecast pin at geocoded location
    if (result.lat !== undefined && result.lon !== undefined) {
      viewer.entities.add({
        id: 'forecast-pin',
        position: Cesium.Cartesian3.fromDegrees(result.lon, result.lat, 0),
        point: {
          pixelSize: 18,
          color: alertColor,
          outlineColor: alertColor.withAlpha(0.5),
          outlineWidth: 20,
          heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text: `${(result.flood_probability * 100).toFixed(0)}% Risk`,
          font: '12px "JetBrains Mono", monospace',
          fillColor: Cesium.Color.WHITE.withAlpha(0.9),
          outlineColor: Cesium.Color.BLACK.withAlpha(0.7),
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, -20),
          heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      })
    }
  }, [result])

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 w-full h-full"
      style={{ background: '#07090e' }}
    />
  )
}
