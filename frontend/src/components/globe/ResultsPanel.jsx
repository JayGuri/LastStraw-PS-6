import React from "react";
import { motion } from "framer-motion";
import { useGlobeStore } from "../../stores/globeStore.js";

const SEVERITY_STYLES = {
  critical: {
    bg: "rgba(192,57,43,0.1)",
    border: "1px solid #c0392b",
    text: "#c0392b",
    label: "CRITICAL",
  },
  high: {
    bg: "rgba(201,169,110,0.1)",
    border: "1px solid #c9a96e",
    text: "#c9a96e",
    label: "HIGH",
  },
  medium: {
    bg: "rgba(236,232,223,0.1)",
    border: "1px solid rgba(236,232,223,0.5)",
    text: "#ece8df",
    label: "MODERATE",
  },
  low: {
    bg: "transparent",
    border: "1px solid rgba(236,232,223,0.2)",
    text: "rgba(236,232,223,0.5)",
    label: "LOW",
  },
};

function SeverityBadge({ severity }) {
  const s = SEVERITY_STYLES[severity] ?? SEVERITY_STYLES.medium;
  return (
    <span
      className="px-1.5 py-0.5 text-[8px] font-mono tracking-widest uppercase"
      style={{ background: s.bg, border: s.border, color: s.text }}
    >
      {s.label}
    </span>
  );
}

export default function ResultsPanel() {
  const result = useGlobeStore((s) => s.result);
  const selectedZone = useGlobeStore((s) => s.selectedZone);
  const setSelectedZone = useGlobeStore((s) => s.setSelectedZone);

  if (!result) return null;
  const { summary, flood_zones } = result;
  const zones = flood_zones?.features ?? [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col flex-1"
      style={{
        background: "#0a0907",
        border: "1px solid rgba(201,169,110,0.15)",
      }}
    >
      {/* Summary header */}
      <div
        className="px-4 py-3 border-b"
        style={{ borderColor: "rgba(201,169,110,0.15)" }}
      >
        <div
          className="text-[9px] font-mono uppercase tracking-[0.3em] mb-4"
          style={{ color: "rgba(201,169,110,0.6)" }}
        >
          Scan Analytics
        </div>
        <div className="grid grid-cols-2 gap-3">
          {[
            {
              label: "Flood Area",
              val: `${summary.total_flood_area_km2.toLocaleString()} km\u00B2`,
              color: "#c0392b",
            },
            {
              label: "Avg Depth",
              val: `${summary.avg_depth_m.toFixed(1)} m`,
              color: "#c9a96e",
            },
            {
              label: "Pop Exposed",
              val: `${(summary.population_exposed / 1e6).toFixed(1)}M`,
              color: "#ece8df",
            },
            {
              label: "Confidence",
              val: `${(summary.confidence_avg * 100).toFixed(0)}%`,
              color: "rgba(236,232,223,0.5)",
            },
          ].map((s) => (
            <div
              key={s.label}
              className="p-2 border"
              style={{
                borderColor: "rgba(236,232,223,0.1)",
                background: "rgba(236,232,223,0.03)",
              }}
            >
              <div
                className="font-mono text-sm tracking-widest"
                style={{ color: s.color }}
              >
                {s.val}
              </div>
              <div
                className="text-[8px] font-mono tracking-widest uppercase mt-1"
                style={{ color: "rgba(236,232,223,0.4)" }}
              >
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Zone table */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <div
          className="px-4 py-2 border-b sticky top-0 bg-[#0a0907]/90 backdrop-blur-sm z-10"
          style={{ borderColor: "rgba(201,169,110,0.15)" }}
        >
          <span
            className="text-[9px] font-mono tracking-widest uppercase"
            style={{ color: "rgba(236,232,223,0.4)" }}
          >
            {zones.length} Flood Zone{zones.length !== 1 ? "s" : ""} Detected
          </span>
        </div>
        {zones.map((feature, i) => {
          const p = feature.properties;
          const isSelected = selectedZone === i;
          return (
            <button
              key={p.zone_id || i}
              onClick={() => setSelectedZone(isSelected ? null : i)}
              className="w-full text-left px-4 py-3 border-b transition-colors"
              style={{
                borderColor: "rgba(201,169,110,0.08)",
                background:
                  isSelected ? "rgba(201,169,110,0.05)" : "transparent",
                borderLeft:
                  isSelected ? "2px solid #c9a96e" : "2px solid transparent",
              }}
              onMouseEnter={(e) => {
                if (!isSelected)
                  e.currentTarget.style.background = "rgba(236,232,223,0.03)";
              }}
              onMouseLeave={(e) => {
                if (!isSelected)
                  e.currentTarget.style.background = "transparent";
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className="text-[11px] font-mono tracking-[0.15em] uppercase"
                  style={{ color: "#ece8df" }}
                >
                  {p.admin_name ??
                    `ZONE-${String(p.zone_id ?? i + 1).padStart(3, "0")}`}
                </span>
                <SeverityBadge severity={p.severity} />
              </div>
              <div
                className="flex items-center gap-4 text-[9px] font-mono tracking-widest uppercase"
                style={{ color: "rgba(236,232,223,0.5)" }}
              >
                <span>
                  {p.area_km2?.toFixed(0)}{" "}
                  <span style={{ color: "rgba(201,169,110,0.5)" }}>KM2</span>
                </span>
                <span>
                  {p.avg_depth_m?.toFixed(1)}
                  <span style={{ color: "rgba(201,169,110,0.5)" }}>m DPTH</span>
                </span>
                <span>
                  {(p.population_exposed / 1000).toFixed(0)}K{" "}
                  <span style={{ color: "rgba(201,169,110,0.5)" }}>POP</span>
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Metadata footer */}
      <div
        className="px-4 py-2 border-t flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2"
        style={{ borderColor: "rgba(201,169,110,0.15)", background: "#060504" }}
      >
        <span
          className="text-[8px] font-mono tracking-[0.2em] uppercase"
          style={{ color: "rgba(236,232,223,0.4)" }}
        >
          SYS &middot; {summary.sensor} // {summary.detector}
        </span>
        <span
          className="text-[8px] font-mono tracking-widest uppercase"
          style={{ color: "rgba(201,169,110,0.4)" }}
        >
          ID: {summary.scene_id?.slice(0, 20) ?? "N/A"}
        </span>
      </div>
    </motion.div>
  );
}
