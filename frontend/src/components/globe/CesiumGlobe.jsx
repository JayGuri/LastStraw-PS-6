import React, { useEffect, useRef } from 'react'
import * as Cesium from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'
import { useGlobeStore } from '../../stores/globeStore.js'
import { FloodBarDataSource } from './FloodBarDataSource.js'

// Severity → fill/outline/extrusion config
const SEVERITY_CFG = {
  critical: {
    fill:      new Cesium.Color(0.85, 0.25, 0.25, 0.55),
    outline:   new Cesium.Color(0.85, 0.25, 0.25, 1.0),
    glow:      new Cesium.Color(0.85, 0.25, 0.25, 0.15),
    extrude:   3500,   // metres × avg_depth_m
  },
  high: {
    fill:      new Cesium.Color(0.82, 0.41, 0.16, 0.50),
    outline:   new Cesium.Color(0.82, 0.41, 0.16, 1.0),
    glow:      new Cesium.Color(0.82, 0.41, 0.16, 0.12),
    extrude:   2500,
  },
  medium: {
    fill:      new Cesium.Color(0.78, 0.63, 0.09, 0.42),
    outline:   new Cesium.Color(0.78, 0.63, 0.09, 1.0),
    glow:      new Cesium.Color(0.78, 0.63, 0.09, 0.10),
    extrude:   1500,
  },
  low: {
    fill:      new Cesium.Color(0.22, 0.63, 0.35, 0.35),
    outline:   new Cesium.Color(0.22, 0.63, 0.35, 1.0),
    glow:      new Cesium.Color(0.22, 0.63, 0.35, 0.08),
    extrude:   600,
  },
}

export default function CesiumGlobe() {
  const containerRef  = useRef(null)
  const viewerRef     = useRef(null)
  const isRotatingRef = useRef(true)   // controls auto-spin

  const geocoded    = useGlobeStore(s => s.geocoded)
  const result      = useGlobeStore(s => s.result)
  const overlayMode = useGlobeStore(s => s.overlayMode)
  const barMetric   = useGlobeStore(s => s.barMetric)
  const selectedZone = useGlobeStore(s => s.selectedZone)

  // ─────────────────────────────────────────────────────────
  // 1. INIT VIEWER (once)
  // ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return

    Cesium.Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_TOKEN || ''

    const viewer = new Cesium.Viewer(containerRef.current, {
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
      creditContainer:      document.createElement('div'),
      scene3DOnly:          true,
      orderIndependentTranslucency: true,
    })

    const scene = viewer.scene
    const globe = scene.globe

    // ── Dark theme ──
    scene.backgroundColor = Cesium.Color.fromCssColorString('#07090e')
    globe.baseColor        = Cesium.Color.fromCssColorString('#0e141f')

    // ── Real-time sun lighting ──
    globe.enableLighting = true
    scene.skyAtmosphere  = new Cesium.SkyAtmosphere()
    scene.skyBox         = new Cesium.SkyBox({
      sources: {
        positiveX: '/Cesium/Assets/Textures/SkyBox/tycho2t3_80_px.jpg',
        negativeX: '/Cesium/Assets/Textures/SkyBox/tycho2t3_80_mx.jpg',
        positiveY: '/Cesium/Assets/Textures/SkyBox/tycho2t3_80_py.jpg',
        negativeY: '/Cesium/Assets/Textures/SkyBox/tycho2t3_80_my.jpg',
        positiveZ: '/Cesium/Assets/Textures/SkyBox/tycho2t3_80_pz.jpg',
        negativeZ: '/Cesium/Assets/Textures/SkyBox/tycho2t3_80_mz.jpg',
      },
    })

    // Sync clock to real current time so sun angle is accurate
    viewer.clock.currentTime = Cesium.JulianDate.fromDate(new Date())
    viewer.clock.shouldAnimate = false   // don't advance time — just freeze at now

    // ── Auto-rotation: slow westward spin ──
    scene.preRender.addEventListener(() => {
      if (isRotatingRef.current) {
        scene.camera.rotate(Cesium.Cartesian3.UNIT_Z, -0.00012)
      }
    })

    // ── Initial camera: pull back to show whole Earth ──
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(72, 20, 22000000),
      orientation: {
        heading: Cesium.Math.toRadians(0),
        pitch:   Cesium.Math.toRadians(-90),
        roll:    0,
      },
    })

    viewerRef.current = viewer

    return () => {
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        viewerRef.current.destroy()
      }
    }
  }, [])

  // ─────────────────────────────────────────────────────────
  // 2. FLY TO GEOCODED REGION — stop rotation, draw boundary + pin
  // ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!viewerRef.current) return
    const viewer = viewerRef.current

    // Remove old boundary / pin
    ;['region-boundary', 'region-pin'].forEach(name => {
      viewer.dataSources.getByName(name).forEach(ds => viewer.dataSources.remove(ds))
    })
    viewer.entities.removeById('region-center-pin')

    if (!geocoded) {
      isRotatingRef.current = true
      return
    }

    // Stop auto-rotation
    isRotatingRef.current = false

    const { lat, lon, bbox } = geocoded

    // ── Fly to region ──
    if (bbox && bbox.length === 4) {
      viewer.camera.flyTo({
        destination: Cesium.Rectangle.fromDegrees(bbox[0], bbox[1], bbox[2], bbox[3]),
        duration: 2.2,
        easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
      })
    } else {
      viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(lon, lat, 800000),
        orientation: {
          heading: 0,
          pitch: Cesium.Math.toRadians(-45),
          roll: 0,
        },
        duration: 2.2,
        easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
      })
    }

    // ── Boundary outline (gold glow polyline) ──
    if (geocoded.boundary_geojson) {
      const boundaryDs = new Cesium.GeoJsonDataSource('region-boundary')
      boundaryDs.load(geocoded.boundary_geojson, {
        stroke:      Cesium.Color.fromCssColorString('#d4900a').withAlpha(0.85),
        strokeWidth: 3,
        fill:        Cesium.Color.TRANSPARENT,
        clampToGround: true,
      }).then(() => {
        // Upgrade each polyline to a glow material
        boundaryDs.entities.values.forEach(entity => {
          if (entity.polyline) {
            entity.polyline.material = new Cesium.PolylineGlowMaterialProperty({
              glowPower:  0.25,
              taperPower: 1.0,
              color:      Cesium.Color.fromCssColorString('#d4900a'),
            })
            entity.polyline.width = 4
          }
          if (entity.polygon) {
            entity.polygon.material = Cesium.Color.fromCssColorString('#d4900a').withAlpha(0.04)
            entity.polygon.outline  = false
          }
        })
        viewer.dataSources.add(boundaryDs)
      })
    }

    // ── Center pin — glowing gold dot with pulsing ring ──
    viewer.entities.add({
      id: 'region-center-pin',
      position: Cesium.Cartesian3.fromDegrees(lon, lat, 0),
      point: {
        pixelSize:    10,
        color:        Cesium.Color.fromCssColorString('#e8ab30'),
        outlineColor: Cesium.Color.fromCssColorString('#d4900a').withAlpha(0.6),
        outlineWidth: 8,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    })
  }, [geocoded])

  // ─────────────────────────────────────────────────────────
  // 3. RENDER FLOOD RESULTS
  // ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!viewerRef.current) return
    const viewer = viewerRef.current

    // Clear stale flood layers
    ;['flood-zones', 'flood-bars', 'zone-markers'].forEach(name => {
      viewer.dataSources.getByName(name).forEach(ds => viewer.dataSources.remove(ds))
    })
    // Remove old zone-pin entities
    viewer.entities.values
      .filter(e => e.id?.startsWith('zone-pin-'))
      .forEach(e => viewer.entities.remove(e))

    if (!result) return

    const features = result.flood_zones?.features ?? []

    // ── A. Extruded flood zone polygons ──────────────────
    if (overlayMode === 'polygons' || overlayMode === 'both') {
      const geoDs = new Cesium.GeoJsonDataSource('flood-zones')
      geoDs.load(result.flood_zones, { clampToGround: false }).then(() => {
        geoDs.entities.values.forEach(entity => {
          if (!entity.polygon) return
          const sev     = entity.properties?.severity?.getValue() ?? 'medium'
          const depth   = entity.properties?.avg_depth_m?.getValue() ?? 1
          const cfg     = SEVERITY_CFG[sev] ?? SEVERITY_CFG.medium
          const extrude = Math.round(depth * cfg.extrude)

          entity.polygon.material       = cfg.fill
          entity.polygon.outline        = true
          entity.polygon.outlineColor   = cfg.outline
          entity.polygon.outlineWidth   = 2
          entity.polygon.extrudedHeight = extrude
          entity.polygon.height         = 0
          entity.polygon.shadows        = Cesium.ShadowMode.DISABLED
        })
        viewer.dataSources.add(geoDs)
      })
    }

    // ── B. Zone center markers (pulsing pins) ────────────
    features.forEach((feature, i) => {
      const p  = feature.properties
      const sev = p.severity ?? 'medium'
      const cfg = SEVERITY_CFG[sev] ?? SEVERITY_CFG.medium

      const centroid = p.centroid ?? {
        lat: (p.bbox[1] + p.bbox[3]) / 2,
        lon: (p.bbox[0] + p.bbox[2]) / 2,
      }

      // Inner dot
      viewer.entities.add({
        id:       `zone-pin-${i}`,
        position: Cesium.Cartesian3.fromDegrees(centroid.lon, centroid.lat, 0),
        point: {
          pixelSize:    8,
          color:        cfg.outline,
          outlineColor: cfg.outline.withAlpha(0.3),
          outlineWidth: 10,
          heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text:            p.admin_name ?? `Zone ${i + 1}`,
          font:            '11px "JetBrains Mono", monospace',
          fillColor:       Cesium.Color.WHITE.withAlpha(0.85),
          outlineColor:    Cesium.Color.BLACK.withAlpha(0.6),
          outlineWidth:    2,
          style:           Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin:  Cesium.VerticalOrigin.BOTTOM,
          pixelOffset:     new Cesium.Cartesian2(0, -18),
          heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
          translucencyByDistance: new Cesium.NearFarScalar(100000, 1, 3000000, 0),
        },
      })
    })

    // ── C. Vertical bars ─────────────────────────────────
    if (overlayMode === 'bars' || overlayMode === 'both') {
      const barDs = new FloodBarDataSource('flood-bars')
      barDs.loadFromFloodResponse(result.grid_points, barMetric)
      viewer.dataSources.add(barDs)
    }
  }, [result, overlayMode, barMetric])

  // ─────────────────────────────────────────────────────────
  // 4. SELECTED ZONE — fly + highlight
  // ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (selectedZone === null || !result || !viewerRef.current) return
    const feature = result.flood_zones.features[selectedZone]
    if (!feature) return

    const bbox = feature.properties.bbox
    if (bbox?.length === 4) {
      viewerRef.current.camera.flyTo({
        destination: Cesium.Rectangle.fromDegrees(
          bbox[0] - 0.05, bbox[1] - 0.05,
          bbox[2] + 0.05, bbox[3] + 0.05
        ),
        duration: 1.5,
        easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
      })
    }
  }, [selectedZone, result])

  return (
    <div
      ref={containerRef}
      className="absolute inset-0"
      style={{ background: '#07090e' }}
    />
  )
}
