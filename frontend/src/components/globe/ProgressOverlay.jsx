import React from "react";
import { motion } from "framer-motion";
import { useGlobeStore } from "../../stores/globeStore.js";

const STAGES = [
  { id: "queued", label: "Queued", icon: "Q" },
  { id: "preprocessing", label: "Preprocessing", icon: "P" },
  { id: "detecting", label: "Flood Detection", icon: "D" },
  { id: "scoring", label: "Risk Scoring", icon: "R" },
];

export default function ProgressOverlay() {
  const status = useGlobeStore((s) => s.status);
  const progress = useGlobeStore((s) => s.progress);
  const error = useGlobeStore((s) => s.error);

  const currentIdx = STAGES.findIndex((s) => s.id === status);

  if (status === "failed") {
    return (
      <div
        className="absolute inset-x-4 bottom-4 p-4 z-10"
        style={{ background: "#0a0907", border: "1px solid #c0392b" }}
      >
        <div
          className="text-[10px] font-mono tracking-widest uppercase mb-1"
          style={{ color: "#c0392b" }}
        >
          Detection Failed
        </div>
        <div
          className="text-[9px] font-mono tracking-widest uppercase"
          style={{ color: "rgba(236,232,223,0.5)" }}
        >
          {error ?? "Unknown error"}
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="absolute inset-x-4 bottom-4 p-4 z-10"
      style={{
        background: "#0a0907",
        border: "1px solid rgba(201,169,110,0.52)",
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <span
          className="text-[9px] font-mono tracking-[0.2em] uppercase"
          style={{ color: "#c9a96e" }}
        >
          Scanning Area...
        </span>
        <span
          className="text-[10px] font-mono tracking-widest"
          style={{ color: "#ece8df" }}
        >
          {progress}%
        </span>
      </div>

      {/* Progress bar */}
      <div
        className="h-0.5 overflow-hidden mb-4"
        style={{ background: "rgba(236,232,223,0.1)" }}
      >
        <motion.div
          className="h-full"
          style={{ background: "#c9a96e" }}
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>

      {/* Stage indicators */}
      <div className="flex items-center gap-4">
        {STAGES.map((stage, i) => {
          const isDone = i < currentIdx;
          const isCurrent = i === currentIdx;
          return (
            <div
              key={stage.id}
              className="flex items-center gap-2 text-[9px] font-mono uppercase tracking-widest"
              style={{
                color:
                  isDone ? "rgba(236,232,223,0.5)"
                  : isCurrent ? "#c9a96e"
                  : "rgba(236,232,223,0.2)",
              }}
            >
              {isDone ?
                <span
                  className="w-3 h-3 flex items-center justify-center text-[8px]"
                  style={{
                    background: "rgba(236,232,223,0.1)",
                    color: "#ece8df",
                  }}
                >
                  X
                </span>
              : isCurrent ?
                <span
                  className="w-3 h-3 flex items-center justify-center"
                  style={{
                    border: "1px solid #c9a96e",
                    background: "rgba(201,169,110,0.1)",
                  }}
                >
                  <span
                    className="w-1.5 h-1.5 animate-pulse"
                    style={{ background: "#c9a96e" }}
                  />
                </span>
              : <span
                  className="w-3 h-3 flex items-center justify-center text-[7px]"
                  style={{ border: "1px solid rgba(236,232,223,0.2)" }}
                >
                  {stage.icon}
                </span>
              }
              <span className="hidden sm:inline">{stage.label}</span>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}
