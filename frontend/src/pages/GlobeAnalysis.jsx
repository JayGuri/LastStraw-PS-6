import React from "react";
import { motion } from "framer-motion";
import CesiumGlobe from "../components/globe/CesiumGlobe.jsx";
import RegionForm from "../components/globe/RegionForm.jsx";
import ResultsPanel from "../components/globe/ResultsPanel.jsx";
import ProgressOverlay from "../components/globe/ProgressOverlay.jsx";
import { useGlobeStore } from "../stores/globeStore.js";

const STATUS_CONFIG = {
  idle: { label: "IDLE", style: "text-text-3 border-white/10" },
  queued: { label: "QUEUED", style: "text-[#c9a96e] border-[#c9a96e]/30" },
  preprocessing: {
    label: "PREPROCESSING",
    style: "text-[#c9a96e] border-[#c9a96e]/30",
  },
  detecting: {
    label: "DETECTING",
    style: "text-[#c9a96e] border-[#c9a96e]/30",
  },
  scoring: { label: "SCORING", style: "text-[#c9a96e] border-[#c9a96e]/30" },
  completed: {
    label: "COMPLETED",
    style: "text-[#ece8df] border-[#ece8df]/40",
  },
  failed: { label: "FAILED", style: "text-[#c0392b] border-[#c0392b]/40" },
};

export default function GlobeAnalysis() {
  const status = useGlobeStore((s) => s.status);
  const result = useGlobeStore((s) => s.result);
  const geocoded = useGlobeStore((s) => s.geocoded);

  const isRunning = [
    "queued",
    "preprocessing",
    "detecting",
    "scoring",
  ].includes(status);
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.idle;

  return (
    <div
      className="min-h-screen pt-14 flex flex-col"
      style={{ background: "#060504" }}
    >
      <div
        className="flex-1 flex flex-col max-w-[1700px] mx-auto w-full
                       px-4 sm:px-8 py-6 sm:py-8 gap-6"
      >
        {/* Header */}
        <div
          className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-4"
          style={{ borderBottom: "1px solid rgba(201,169,110,0.15)" }}
        >
          <div>
            <span
              className="tracking-[0.35em] uppercase mb-2 block"
              style={{
                fontSize: "0.6rem",
                color: "#c9a96e",
                fontFamily: "monospace",
              }}
            >
              Theater Operations
            </span>
            <h1
              className="font-display font-light uppercase tracking-[0.2em]"
              style={{ fontSize: "1.4rem", color: "#ece8df" }}
            >
              Global Flood Detection
            </h1>
          </div>

          <div className="flex flex-col items-end gap-2">
            {/* Status chip */}
            <div
              className={`px-2 py-0.5 text-[10px] uppercase font-mono tracking-widest border bg-transparent ${cfg.style}`}
            >
              {isRunning && (
                <span className="w-1.5 h-1.5 bg-[#c9a96e] inline-block mb-[1px] mr-2" />
              )}
              {cfg.label}
            </div>

            {/* Region display */}
            {geocoded?.display_name && (
              <div
                className="text-[10px] font-mono uppercase tracking-widest"
                style={{ color: "rgba(236,232,223,0.6)" }}
              >
                {geocoded.display_name}
              </div>
            )}
          </div>
        </div>

        {/* Main grid */}
        <div
          className="flex-1 flex flex-col lg:grid lg:grid-cols-12 gap-6"
          style={{ minHeight: "min(600px, 75vh)" }}
        >
          {/* Globe panel */}
          <div
            className="col-span-12 lg:col-span-8 relative min-h-[420px]"
            style={{ border: "1px solid rgba(201,169,110,0.4)" }}
          >
            <CesiumGlobe />
            {isRunning && <ProgressOverlay />}
            {status === "failed" && <ProgressOverlay />}

            {/* Corner brackets */}
            <div className="absolute top-0 left-0 w-2 h-2 border-t-2 border-l-2 border-[#c9a96e] z-10" />
            <div className="absolute top-0 right-0 w-2 h-2 border-t-2 border-r-2 border-[#c9a96e] z-10" />
            <div className="absolute bottom-0 left-0 w-2 h-2 border-b-2 border-l-2 border-[#c9a96e] z-10" />
            <div className="absolute bottom-0 right-0 w-2 h-2 border-b-2 border-r-2 border-[#c9a96e] z-10" />
          </div>

          {/* Sidebar */}
          <div
            className="col-span-12 lg:col-span-4 flex flex-col gap-6
                          overflow-y-auto lg:max-h-[calc(100vh-10rem)] pr-2"
          >
            <RegionForm />
            {result && <ResultsPanel />}
          </div>
        </div>
      </div>
    </div>
  );
}
