import React, { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { usePipelineStore } from "../stores/pipelineStore.js";
import LogFeed from "../components/common/LogFeed.jsx";

const STAGE_STATUS_STYLE = {
  idle: {
    ring: "border-[rgba(236,232,223,0.1)]",
    bg: "bg-transparent",
    icon: "[ ]",
    iconColor: "text-[#ece8df]/40 font-mono text-[10px]",
  },
  running: {
    ring: "border-[#c9a96e]",
    bg: "bg-[#c9a96e]/10",
    icon: "[>_]",
    iconColor: "text-[#c9a96e] font-mono text-[10px] animate-pulse",
  },
  done: {
    ring: "border-[rgba(236,232,223,0.3)]",
    bg: "bg-transparent",
    icon: "[X]",
    iconColor: "text-[#ece8df] font-mono text-[10px]",
  },
};

function StageCard({ stage, index, isActive }) {
  const style = STAGE_STATUS_STYLE[stage.status] ?? STAGE_STATUS_STYLE.idle;

  return (
    <motion.div
      layout
      animate={{
        borderColor:
          stage.status === "running" ?
            [
              "rgba(212,144,10,0.2)",
              "rgba(212,144,10,0.6)",
              "rgba(212,144,10,0.2)",
            ]
          : undefined,
      }}
      transition={{
        duration: 1.4,
        repeat: stage.status === "running" ? Infinity : 0,
      }}
      className={`
        p-4 sm:p-5 relative overflow-hidden transition-all duration-500
        ${stage.status === "idle" ? "opacity-50" : ""}
      `}
      style={{
        background:
          stage.status === "running" ? "rgba(201,169,110,0.03)" : "#0a0907",
        border:
          stage.status === "running" ? "1px solid #c9a96e"
          : stage.status === "done" ? "1px solid rgba(236,232,223,0.3)"
          : "1px solid rgba(201,169,110,0.15)",
        borderLeft:
          stage.status === "running" ?
            "4px solid #c9a96e"
          : "1px solid rgba(201,169,110,0.15)",
      }}
    >
      {/* Top stripe (running / done glow) */}
      <div
        className={`absolute top-0 left-0 right-0 h-[1px] transition-all duration-500
        ${
          stage.status === "running" ?
            "bg-[#c9a96e] opacity-30 shadow-[0_0_8px_#c9a96e]"
          : stage.status === "done" ? "bg-[rgba(236,232,223,0.3)]"
          : "bg-transparent"
        }`}
      />

      {/* Shimmer overlay when running */}
      {stage.status === "running" && (
        <div className="absolute inset-0 shimmer pointer-events-none" />
      )}

      <div className="relative flex items-start gap-3 sm:gap-4">
        {/* Status icon */}
        <div
          className={`w-10 h-10 flex items-center justify-center flex-shrink-0
                         border ${style.ring} ${style.bg} transition-all duration-300`}
        >
          <span className={`font-bold ${style.iconColor}`}>{style.icon}</span>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span
              className="text-[9px] font-mono tracking-widest uppercase"
              style={{ color: "rgba(201,169,110,0.6)" }}
            >
              MODULE {String(stage.num).padStart(2, "0")}
            </span>
            {stage.status === "running" && (
              <span
                className="text-[9px] sm:text-[10px] font-mono tracking-[0.2em] uppercase"
                style={{ color: "#c9a96e" }}
              >
                [ PROCESSING ]
              </span>
            )}
            {stage.status === "done" && (
              <span className="text-[9px] sm:text-[10px] font-mono tracking-[0.2em] uppercase text-[#ece8df]">
                [ FINISHED ]
              </span>
            )}
          </div>
          <h3
            className={`font-display font-bold text-sm sm:text-base mb-1 sm:mb-2 transition-colors uppercase tracking-widest
            ${
              stage.status === "running" ? "text-[#c9a96e]"
              : stage.status === "done" ? "text-[#ece8df]"
              : "text-[#ece8df]/60"
            }`}
          >
            {stage.name}
          </h3>
          <p
            className="text-[10px] uppercase font-mono tracking-wide leading-relaxed mb-2 sm:mb-3"
            style={{ color: "rgba(236,232,223,0.5)" }}
          >
            {stage.description}
          </p>

          {/* I/O row */}
          <div className="grid grid-cols-2 gap-2 sm:gap-3 text-[9px] sm:text-[10px] font-mono tracking-wider uppercase">
            <div>
              <div className="mb-1" style={{ color: "rgba(236,232,223,0.4)" }}>
                INPUTS ↓
              </div>
              {stage.inputs.map((inp) => (
                <div
                  key={inp}
                  className="flex items-start gap-1"
                  style={{ color: "rgba(236,232,223,0.6)" }}
                >
                  <span style={{ color: "rgba(201,169,110,0.5)" }}>›</span>
                  <span className="break-all">{inp}</span>
                </div>
              ))}
            </div>
            <div>
              <div className="mb-1" style={{ color: "rgba(201,169,110,0.5)" }}>
                OUTPUTS ↑
              </div>
              {stage.outputs.map((out) => (
                <div
                  key={out}
                  className="flex items-start gap-1"
                  style={{ color: "rgba(201,169,110,0.8)" }}
                >
                  <span style={{ color: "rgba(201,169,110,0.5)" }}>›</span>
                  <span className="break-all">{out}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Libraries */}
          <div className="flex flex-wrap gap-1 sm:gap-1.5 mt-2 sm:mt-3">
            {stage.libs.map((lib) => (
              <span
                key={lib}
                className="font-mono text-[9px] px-1.5 sm:px-2 py-0.5 border"
                style={{
                  background: "transparent",
                  borderColor: "rgba(236,232,223,0.1)",
                  color: "rgba(236,232,223,0.5)",
                }}
              >
                {lib}
              </span>
            ))}
          </div>

          {/* Progress bar */}
          {stage.status === "running" && (
            <motion.div
              className="mt-4 h-0.5 overflow-hidden"
              style={{ background: "rgba(236,232,223,0.1)" }}
            >
              <motion.div
                className="h-full"
                style={{ background: "#c9a96e" }}
                initial={{ width: "0%" }}
                animate={{ width: "95%" }}
                transition={{ duration: stage.duration / 1000, ease: "linear" }}
              />
            </motion.div>
          )}
          {stage.status === "done" && (
            <div
              className="mt-4 h-0.5"
              style={{ background: "rgba(236,232,223,0.5)" }}
            >
              <div
                className="h-full w-full"
                style={{ background: "rgba(236,232,223,0.5)" }}
              />
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

function PipelineConnector({ active, done }) {
  return (
    <div className="flex items-center justify-center h-4 sm:h-6">
      <div
        className="w-[1px] h-full"
        style={{
          background:
            done ? "rgba(236,232,223,0.3)"
            : active ? "#c9a96e"
            : "rgba(201,169,110,0.15)",
        }}
      />
    </div>
  );
}

export default function MissionControl() {
  const {
    stages,
    running,
    done,
    logs,
    elapsed,
    totalDuration,
    startPipeline,
    resetPipeline,
    currentStageIdx,
  } = usePipelineStore();

  const progress =
    totalDuration ? Math.min((elapsed / (totalDuration / 1000)) * 100, 100) : 0;

  // Format elapsed seconds → HH:MM:SS
  const elapsedHH = String(Math.floor(elapsed / 3600)).padStart(2, "0");
  const elapsedMM = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
  const elapsedSS = String(elapsed % 60).padStart(2, "0");
  const elapsedFormatted = `${elapsedHH}:${elapsedMM}:${elapsedSS}`;

  return (
    <div
      className="min-h-screen pt-14 flex flex-col"
      style={{ background: "#060504" }}
    >
      <div className="flex-1 flex flex-col max-w-[1700px] mx-auto w-full px-4 sm:px-8 py-6 sm:py-8 gap-6">
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
              Execution Control
            </span>
            <h1
              className="font-display font-light uppercase tracking-[0.2em]"
              style={{
                fontSize: "1.4rem",
                color: "#ece8df",
                marginBottom: "8px",
              }}
            >
              Mission Pipeline
            </h1>
            <p
              className="font-mono text-[10px] uppercase tracking-wider max-w-lg leading-relaxed"
              style={{ color: "rgba(236,232,223,0.4)" }}
            >
              Simulate the full SAR flood-detection pipeline — from satellite
              ingestion to per-district risk scores. Watch each stage execute in
              real time.
            </p>
          </div>

          <div className="flex flex-col items-end gap-3">
            {/* Elapsed HH:MM:SS timer */}
            {(running || done) && (
              <div
                className="font-mono text-xl tabular-nums tracking-[0.3em] font-light"
                style={{ color: "#c9a96e" }}
              >
                {elapsedFormatted}
              </div>
            )}

            <div className="flex items-center justify-end gap-2 w-full mt-2">
              {!running && !done && (
                <button
                  onClick={startPipeline}
                  className="flex items-center gap-2 px-5 py-2.5 bg-transparent border uppercase font-mono text-[10px] tracking-[0.2em] transition-all hover:bg-[rgba(201,169,110,0.1)]"
                  style={{ borderColor: "#c9a96e", color: "#c9a96e" }}
                >
                  <span style={{ fontSize: "14px", lineHeight: "14px" }}>
                    ▶
                  </span>{" "}
                  INITIALIZE
                </button>
              )}
              {(running || done) && (
                <button
                  onClick={resetPipeline}
                  className="px-5 py-2.5 bg-transparent border uppercase font-mono text-[10px] tracking-[0.2em] transition-all hover:bg-[rgba(236,232,223,0.1)]"
                  style={{
                    borderColor: "rgba(236,232,223,0.3)",
                    color: "#ece8df",
                  }}
                >
                  [ RESET ]
                </button>
              )}

              {/* Status chip */}
              <div
                className="px-3 py-2.5 font-mono text-[10px] tracking-[0.2em] uppercase border"
                style={{
                  background:
                    running ? "rgba(201,169,110,0.05)"
                    : done ? "rgba(236,232,223,0.05)"
                    : "transparent",
                  borderColor:
                    running ? "#c9a96e"
                    : done ? "rgba(236,232,223,0.6)"
                    : "rgba(201,169,110,0.15)",
                  color:
                    running ? "#c9a96e"
                    : done ? "#ece8df"
                    : "rgba(236,232,223,0.5)",
                }}
              >
                {running ?
                  `[RUN] ${elapsed}s`
                : done ?
                  `[END] ${elapsed}s`
                : "[IDLE]"}
              </div>
            </div>
          </div>
        </div>

        {/* Full-width progress bar — directly below header */}
        <div
          className="h-px w-full overflow-hidden"
          style={{
            background: "rgba(201,169,110,0.15)",
            marginTop: "-24px",
            marginBottom: "16px",
          }}
        >
          <div
            className="h-full bg-[#c9a96e] transition-all duration-1000"
            style={{
              width: `${progress}%`,
              boxShadow: "0 0 8px rgba(201,169,110,0.6)",
            }}
          />
        </div>

        <div className="grid lg:grid-cols-3 gap-4 sm:gap-6">
          {/* Left — stage pipeline */}
          <div className="lg:col-span-2">
            <div className="space-y-0">
              {stages.map((stage, i) => (
                <div key={stage.id}>
                  <StageCard
                    stage={stage}
                    index={i}
                    isActive={currentStageIdx === i}
                  />
                  {i < stages.length - 1 && (
                    <PipelineConnector
                      active={currentStageIdx === i && running}
                      done={stage.status === "done"}
                    />
                  )}
                </div>
              ))}
            </div>

            {/* Completion card */}
            <AnimatePresence>
              {done && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="mt-6 p-6 border text-left"
                  style={{
                    background: "#0a0907",
                    borderColor: "rgba(236,232,223,0.3)",
                  }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span
                      className="text-[10px] font-mono tracking-widest uppercase"
                      style={{ color: "rgba(236,232,223,0.6)" }}
                    >
                      Operation Success
                    </span>
                    <span
                      className="uppercase tracking-[0.2em] font-mono text-xs"
                      style={{ color: "#ece8df" }}
                    >
                      [ END ]
                    </span>
                  </div>
                  <div
                    className="font-display font-light text-xl tracking-widest uppercase mb-2"
                    style={{ color: "#ece8df" }}
                  >
                    Pipeline Terminus Reached
                  </div>
                  <div
                    className="text-[10px] font-mono tracking-widest uppercase mb-6"
                    style={{ color: "rgba(236,232,223,0.4)" }}
                  >
                    RUN-20241105-BD-001 // {elapsed}S ELAPSED
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 sm:gap-4">
                    {[
                      { l: "Districts Analyzed", v: "64", color: "#ece8df" },
                      { l: "Flooded km²", v: "18,400", color: "#c9a96e" },
                      {
                        l: "Confidence",
                        v: "87.4%",
                        color: "rgba(236,232,223,0.5)",
                      },
                    ].map((s) => (
                      <div
                        key={s.l}
                        className="p-3 border"
                        style={{
                          background: "transparent",
                          borderColor: "rgba(201,169,110,0.15)",
                        }}
                      >
                        <div
                          className="font-mono text-lg tracking-widest mb-1"
                          style={{ color: s.color }}
                        >
                          {s.v}
                        </div>
                        <div
                          className="text-[8px] font-mono tracking-widest uppercase"
                          style={{ color: "rgba(201,169,110,0.5)" }}
                        >
                          {s.l}
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Right — log feed + metadata */}
          <div className="space-y-4 sm:space-y-5">
            {/* Live log terminal */}
            <div
              className="border border-[rgba(201,169,110,0.15)] overflow-hidden"
              style={{ background: "#0a0907" }}
            >
              <div
                className="flex items-center gap-1.5 px-4 py-2 border-b"
                style={{
                  borderColor: "rgba(201,169,110,0.15)",
                  background: "rgba(201,169,110,0.03)",
                }}
              >
                <span
                  className="font-mono text-[9px] uppercase tracking-[0.3em] font-bold"
                  style={{ color: "#c9a96e" }}
                >
                  Execution Matrix
                </span>
                <span
                  className="font-mono text-[9px] tracking-widest uppercase ml-auto"
                  style={{ color: "rgba(236,232,223,0.3)" }}
                >
                  {logs.length} ENTRIES
                </span>
              </div>
              <LogFeed logs={logs} maxHeight={320} />
              {/* Blinking cursor when running */}
              {running && (
                <div
                  className="font-mono text-xs px-3 py-1"
                  style={{
                    animation: "blink 1s step-end infinite",
                    color: "#c9a96e",
                  }}
                >
                  ▋
                </div>
              )}
            </div>

            {/* Run metadata */}
            <div
              className="border p-4 sm:p-5"
              style={{
                background: "#0a0907",
                borderColor: "rgba(201,169,110,0.15)",
              }}
            >
              <div
                className="text-[9px] font-mono mb-3 uppercase tracking-[0.3em]"
                style={{ color: "#c9a96e" }}
              >
                Run Metadata
              </div>
              {[
                { k: "run_id", v: "RUN-20241105-BD-001" },
                { k: "scene_id", v: "S1_GRD_20241104T001023" },
                { k: "aoi_id", v: "BD-DELTA-01" },
                { k: "sensor", v: "Sentinel-1 GRD" },
                { k: "detector", v: "UNet (Sen1Floods11)" },
                { k: "threshold", v: "-16.0 dB" },
                { k: "crs", v: "EPSG:32645 → 4326" },
                { k: "tile_size", v: "512 × 512 px" },
              ].map(({ k, v }) => (
                <div
                  key={k}
                  className="flex justify-between border-b py-2 pl-2"
                  style={{ borderColor: "rgba(236,232,223,0.05)" }}
                >
                  <span
                    className="text-[9px] font-mono uppercase tracking-widest"
                    style={{ color: "rgba(201,169,110,0.5)" }}
                  >
                    {k}
                  </span>
                  <span
                    className="font-mono text-[10px] tracking-widest uppercase text-right break-all ml-4"
                    style={{ color: "rgba(236,232,223,0.8)" }}
                  >
                    {v}
                  </span>
                </div>
              ))}
            </div>

            {/* Stage timing */}
            <div
              className="border p-4 sm:p-5 mt-4"
              style={{
                background: "#0a0907",
                borderColor: "rgba(201,169,110,0.15)",
              }}
            >
              <div
                className="text-[9px] font-mono mb-3 uppercase tracking-[0.3em]"
                style={{ color: "rgba(236,232,223,0.5)" }}
              >
                Stage Timing Ledger
              </div>
              {stages.map((s) => (
                <div
                  key={s.id}
                  className="flex items-center gap-2 sm:gap-3 py-1.5 border-b pl-2"
                  style={{ borderColor: "rgba(236,232,223,0.05)" }}
                >
                  <span
                    className="text-[10px] font-mono tracking-widest uppercase flex-1 truncate"
                    style={{ color: "rgba(236,232,223,0.7)" }}
                  >
                    {s.name}
                  </span>
                  <span
                    className={`text-[10px] font-mono tracking-widest flex-shrink-0 uppercase ${
                      s.status === "done" ? "text-[#ece8df]"
                      : s.status === "running" ? "text-[#c9a96e] animate-pulse"
                      : "text-[#ece8df]/30"
                    }`}
                  >
                    {s.status === "done" ?
                      `${(s.duration / 1000).toFixed(0)}S`
                    : s.status === "running" ?
                      "CALCULATING"
                    : "--"}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
