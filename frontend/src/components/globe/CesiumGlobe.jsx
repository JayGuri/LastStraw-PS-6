import React, { useEffect, useRef, useCallback } from 'react'
import * as Cesium from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'
import { useGlobeStore } from '../../stores/globeStore.js'
import { FloodBarDataSource } from './FloodBarDataSource.js'

const SEVERITY_COLORS = {
  critical: 'rgba(216, 64, 64, 0.5)',
  high:     'rgba(208, 104, 40, 0.5)',
  medium:   'rgba(200, 160, 24, 0.4)',
  low:      'rgba(56, 160, 88, 0.3)',
}

const SEVERITY_OUTLINE = {
  critical: '#d84040',
  high:     '#d06828',
  medium:   '#c8a018',
  low:      '#38a058',
}

export default function CesiumGlobe() {
  const containerRef = useRef(null)
  const viewerRef = useRef(null)

  const geocoded = useGlobeStore(s => s.geocoded)
  const result = useGlobeStore(s => s.result)
  const overlayMode = useGlobeStore(s => s.overlayMode)
  const barMetric = useGlobeStore(s => s.barMetric)
  const selectedZone = useGlobeStore(s => s.selectedZone)

  // ── Initialize Viewer ──
  useEffect(() => {
    if (!containerRef.current) return

    Cesium.Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_TOKEN || ''

    const viewer = new Cesium.Viewer(containerRef.current, {
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
      creditContainer: document.createElement('div'),
      scene3DOnly: true,
    })

    viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#07090e')
    viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#0e141f')
    viewer.scene.globe.enableLighting = true
    viewer.scene.skyAtmosphere = new Cesium.SkyAtmosphere()

    viewerRef.current = viewer

    return () => {
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        viewerRef.current.destroy()
      }
    }
  }, [])

  // ── Fly to geocoded region ──
  useEffect(() => {
    if (!geocoded || !viewerRef.current) return
    const viewer = viewerRef.current
    const { lat, lon, bbox } = geocoded

    if (bbox && bbox.length === 4) {
      viewer.camera.flyTo({
        destination: Cesium.Rectangle.fromDegrees(bbox[0], bbox[1], bbox[2], bbox[3]),
        duration: 2.0,
      })
    } else {
      viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(lon, lat, 500000),
        duration: 2.0,
      })
    }

    // Draw boundary outline
    const existingBoundary = viewer.dataSources.getByName('region-boundary')
    existingBoundary.forEach(ds => viewer.dataSources.remove(ds))

    if (geocoded.boundary_geojson) {
      const boundaryDs = new Cesium.GeoJsonDataSource('region-boundary')
      boundaryDs.load(geocoded.boundary_geojson, {
        stroke: Cesium.Color.fromCssColorString('#d4900a').withAlpha(0.7),
        strokeWidth: 2,
        fill: Cesium.Color.fromCssColorString('#d4900a').withAlpha(0.05),
        clampToGround: true,
      }).then(() => {
        viewer.dataSources.add(boundaryDs)
      })
    }
  }, [geocoded])

  // ── Load flood results ──
  useEffect(() => {
    if (!viewerRef.current) return
    const viewer = viewerRef.current

    // Clear previous flood data
    const existingZones = viewer.dataSources.getByName('flood-zones')
    existingZones.forEach(ds => viewer.dataSources.remove(ds))
    const existingBars = viewer.dataSources.getByName('flood-bars')
    existingBars.forEach(ds => viewer.dataSources.remove(ds))

    if (!result) return

    // 1. Polygon overlays
    if (overlayMode === 'polygons' || overlayMode === 'both') {
      const geoDs = new Cesium.GeoJsonDataSource('flood-zones')
      geoDs.load(result.flood_zones, {
        clampToGround: true,
      }).then(() => {
        geoDs.entities.values.forEach(entity => {
          if (!entity.polygon) return
          const severity = entity.properties?.severity?.getValue() ?? 'medium'
          entity.polygon.material = Cesium.Color.fromCssColorString(
            SEVERITY_COLORS[severity] ?? SEVERITY_COLORS.medium
          )
          entity.polygon.outline = true
          entity.polygon.outlineColor = Cesium.Color.fromCssColorString(
            SEVERITY_OUTLINE[severity] ?? SEVERITY_OUTLINE.medium
          )
          entity.polygon.outlineWidth = 1
        })
        viewer.dataSources.add(geoDs)
      })
    }

    // 2. Vertical bars
    if (overlayMode === 'bars' || overlayMode === 'both') {
      const barDs = new FloodBarDataSource('flood-bars')
      barDs.loadFromFloodResponse(result.grid_points, barMetric)
      viewer.dataSources.add(barDs)
    }
  }, [result, overlayMode, barMetric])

  // ── Fly to selected zone ──
  useEffect(() => {
    if (selectedZone === null || !result || !viewerRef.current) return
    const feature = result.flood_zones.features[selectedZone]
    if (!feature) return

    const bbox = feature.properties.bbox
    if (bbox && bbox.length === 4) {
      viewerRef.current.camera.flyTo({
        destination: Cesium.Rectangle.fromDegrees(bbox[0], bbox[1], bbox[2], bbox[3]),
        duration: 1.5,
      })
    }
  }, [selectedZone, result])

  return (
    <div ref={containerRef} className="absolute inset-0" style={{ background: '#07090e' }} />
  )
}
