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

  const geocoded     = useGlobeStore(s => s.geocoded)
  const result       = useGlobeStore(s => s.result)
  const selectedZone = useGlobeStore(s => s.selectedZone)

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

    // Green highlight for selected region (precise boundary from Nominatim)
    if (geocoded.boundary_geojson) {
      const boundaryDs = new Cesium.GeoJsonDataSource('region-boundary')
      const greenFill  = Cesium.Color.fromCssColorString('#16a34a').withAlpha(0.32)
      const greenEdge  = Cesium.Color.fromCssColorString('#22c55e')
      boundaryDs
        .load(geocoded.boundary_geojson, {
          stroke:       greenEdge,
          strokeWidth:  2,
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
              entity.polygon.material   = greenFill
              entity.polygon.outline    = true
              entity.polygon.outlineColor = greenEdge
              entity.polygon.outlineWidth = 2.5
              entity.polygon.clampToGround = true
              entity.polygon.classificationType = Cesium.ClassificationType.TERRAIN
            }
          })
          if (!viewer.isDestroyed()) viewer.dataSources.add(boundaryDs)
        })
    } else if (geocoded.bbox?.length === 4) {
      // No precise polygon: draw bbox as rectangle for visibility
      const [w, s, e, n] = geocoded.bbox
      viewer.entities.add({
        id: 'region-boundary-bbox',
        rectangle: {
          coordinates: Cesium.Rectangle.fromDegrees(w, s, e, n),
          material:    Cesium.Color.fromCssColorString('#16a34a').withAlpha(0.25),
          outline:     true,
          outlineColor: Cesium.Color.fromCssColorString('#22c55e'),
          outlineWidth: 2,
        },
      })
    }

    viewer.entities.add({
      id:       'region-center-pin',
      position: Cesium.Cartesian3.fromDegrees(lon, lat, 0),
      point: {
        pixelSize:    10,
        color:        Cesium.Color.fromCssColorString('#22c55e'),
        outlineColor: Cesium.Color.fromCssColorString('#16a34a').withAlpha(0.6),
        outlineWidth: 8,
        heightReference:          Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    })
  }, [geocoded])

  // ──────────────────────────────────────────────────────────
  // 3. RENDER FLOOD RESULTS
  // ──────────────────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || viewer.isDestroyed()) return

    ;['flood-zones', 'zone-markers'].forEach(name => {
      viewer.dataSources.getByName(name).forEach(ds => viewer.dataSources.remove(ds))
    })
    viewer.entities.values
      .filter(e => e.id?.startsWith('zone-pin-'))
      .forEach(e => viewer.entities.remove(e))

    if (!result) return

    const features = result.flood_zones?.features ?? []

    const geoDs = new Cesium.GeoJsonDataSource('flood-zones')
    geoDs.load(result.flood_zones, { clampToGround: false }).then(() => {
      geoDs.entities.values.forEach(entity => {
        if (!entity.polygon) return
        const sev     = entity.properties?.severity?.getValue() ?? 'medium'
        const depth   = entity.properties?.avg_depth_m?.getValue() ?? 1
        const cfg     = SEVERITY_CFG[sev] ?? SEVERITY_CFG.medium
        const extrude = Math.round(depth * cfg.extrude)

        entity.polygon.material            = cfg.fill
        entity.polygon.outline             = true
        entity.polygon.outlineColor        = cfg.outline
        entity.polygon.outlineWidth        = 2
        entity.polygon.extrudedHeight      = extrude
        entity.polygon.height              = 0
        entity.polygon.shadows             = Cesium.ShadowMode.DISABLED
        entity.polygon.classificationType  = Cesium.ClassificationType.BOTH
      })
      if (!viewer.isDestroyed()) viewer.dataSources.add(geoDs)
    })

    features.forEach((feature, i) => {
      const p   = feature.properties
      const sev = p.severity ?? 'medium'
      const cfg = SEVERITY_CFG[sev] ?? SEVERITY_CFG.medium
      const centroid = p.centroid ?? {
        lat: (p.bbox[1] + p.bbox[3]) / 2,
        lon: (p.bbox[0] + p.bbox[2]) / 2,
      }

      viewer.entities.add({
        id:       `zone-pin-${i}`,
        position: Cesium.Cartesian3.fromDegrees(centroid.lon, centroid.lat, 0),
        point: {
          pixelSize:    8,
          color:        cfg.outline,
          outlineColor: cfg.outline.withAlpha(0.3),
          outlineWidth: 10,
          heightReference:          Cesium.HeightReference.CLAMP_TO_GROUND,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text:           p.admin_name ?? `Zone ${i + 1}`,
          font:           '11px "JetBrains Mono", monospace',
          fillColor:      Cesium.Color.WHITE.withAlpha(0.9),
          outlineColor:   Cesium.Color.BLACK.withAlpha(0.7),
          outlineWidth:   2,
          style:          Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset:    new Cesium.Cartesian2(0, -18),
          heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
          translucencyByDistance: new Cesium.NearFarScalar(100_000, 1, 3_000_000, 0),
        },
      })
    })
  }, [result])

  // ──────────────────────────────────────────────────────────
  // 4. SELECTED ZONE — fly + highlight
  // ──────────────────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current
    if (selectedZone === null || !result || !viewer || viewer.isDestroyed()) return

    const feature = result.flood_zones?.features?.[selectedZone]
    if (!feature) return

    const bbox = feature.properties?.bbox
    if (bbox?.length === 4) {
      viewer.camera.flyTo({
        destination: Cesium.Rectangle.fromDegrees(
          bbox[0] - 0.05, bbox[1] - 0.05,
          bbox[2] + 0.05, bbox[3] + 0.05,
        ),
        duration: 1.5,
        easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
      })
    }
  }, [selectedZone, result])

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 w-full h-full"
      style={{ background: '#07090e' }}
    />
  )
}
