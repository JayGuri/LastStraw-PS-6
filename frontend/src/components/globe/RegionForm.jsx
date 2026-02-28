import React, { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useGlobeStore } from "../../stores/globeStore.js";
import { useAppStore } from "../../stores/appStore.js";
import { geocodeApi, parseNominatimResult } from "../../api/geocodeApi.js";
import { floodDetectApi } from "../../api/floodDetectApi.js";
import { mockFloodResponse } from "../../data/mockFloodResponse.js";
import CalendarPicker from "../ui/CalendarPicker.jsx";

// ── Inline autocomplete input ──────────────────────────────────────────────
function GeoSearchInput({
  label,
  placeholder,
  value,
  onChange,
  results,
  onSelect,
  isSearching,
  onClear,
}) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (!wrapRef.current?.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Open when results arrive
  useEffect(() => {
    if (results.length > 0) setOpen(true);
  }, [results]);

  return (
    <div ref={wrapRef} className="relative">
      <label
        className="text-[10px] uppercase font-mono tracking-[0.2em] mb-2 block"
        style={{ color: "rgba(236,232,223,0.5)" }}
      >
        {label}
      </label>
      <div className="relative flex items-center">
        <input
          type="text"
          value={value}
          onChange={(e) => {
            onChange(e.target.value);
            setOpen(true);
          }}
          onFocus={() => {
            if (results.length > 0) setOpen(true);
          }}
          placeholder={placeholder}
          className="w-full text-xs font-mono tracking-wide px-3 py-2.5 transition-all outline-none"
          style={{
            background: "rgba(236,232,223,0.03)",
            border: "1px solid rgba(201,169,110,0.15)",
            color: "#ece8df",
          }}
          onFocusCapture={(e) => {
            e.target.style.borderColor = "#c9a96e";
          }}
          onBlurCapture={(e) => {
            e.target.style.borderColor = "rgba(201,169,110,0.15)";
          }}
          autoComplete="off"
        />
        {isSearching && (
          <span
            className="absolute right-3 text-[10px] font-mono tracking-widest animate-pulse"
            style={{ color: "rgba(201,169,110,0.5)" }}
          >
            ...
          </span>
        )}
        {!isSearching && value && (
          <button
            onClick={() => {
              onClear();
              setOpen(false);
            }}
            className="absolute right-3 text-xs"
            style={{ color: "rgba(236,232,223,0.4)", fontFamily: "monospace" }}
          >
            [X]
          </button>
        )}
      </div>

      <AnimatePresence>
        {open && results.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.12 }}
            className="absolute z-50 w-full mt-1 overflow-hidden"
            style={{
              background: "#0a0907",
              border: "1px solid rgba(201,169,110,0.3)",
              borderTop: "none",
              maxHeight: "200px",
              overflowY: "auto",
            }}
          >
            {results.map((item) => {
              const parts = item.display_name.split(",").map((s) => s.trim());
              const primary = parts[0];
              const secondary = parts.slice(1, 3).join(", ");
              return (
                <button
                  key={item.place_id}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    onSelect(item);
                    setOpen(false);
                  }}
                  className="w-full text-left px-3 py-2.5 flex flex-col hover:bg-[rgba(201,169,110,0.05)] transition-colors"
                  style={{ borderBottom: "1px solid rgba(201,169,110,0.08)" }}
                >
                  <span
                    className="text-[11px] font-mono uppercase tracking-widest truncate"
                    style={{ color: "#ece8df" }}
                  >
                    {primary}
                  </span>
                  {secondary && (
                    <span
                      className="text-[9px] font-mono uppercase tracking-wider truncate"
                      style={{ color: "rgba(236,232,223,0.4)" }}
                    >
                      {secondary}
                    </span>
                  )}
                </button>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
export default function RegionForm() {
  const store = useGlobeStore();
  const isMockMode = useAppStore((s) => s.isMockMode);
  const showNotification = useAppStore((s) => s.showNotification);

  // Three independent field states
  const [cityQuery, setCityQuery] = useState("");
  const [stateQuery, setStateQuery] = useState("");
  const [countryQuery, setCountryQuery] = useState("");

  const [cityResults, setCityResults] = useState([]);
  const [stateResults, setStateResults] = useState([]);
  const [countryResults, setCountryResults] = useState([]);

  const [searchingCity, setSearchingCity] = useState(false);
  const [searchingState, setSearchingState] = useState(false);
  const [searchingCountry, setSearchingCountry] = useState(false);

  const cityDebounce = useRef(null);
  const stateDebounce = useRef(null);
  const countryDebounce = useRef(null);

  // ── Debounced searches ──
  const searchCity = useCallback((q) => {
    clearTimeout(cityDebounce.current);
    if (q.length < 2) {
      setCityResults([]);
      return;
    }
    cityDebounce.current = setTimeout(async () => {
      setSearchingCity(true);
      const { data } = await geocodeApi.search(q, { limit: 6 });
      setCityResults(data ?? []);
      setSearchingCity(false);
    }, 280);
  }, []);

  const searchState = useCallback((q) => {
    clearTimeout(stateDebounce.current);
    if (q.length < 2) {
      setStateResults([]);
      return;
    }
    stateDebounce.current = setTimeout(async () => {
      setSearchingState(true);
      const { data } = await geocodeApi.searchState(q, { limit: 6 });
      setStateResults(data ?? []);
      setSearchingState(false);
    }, 280);
  }, []);

  const searchCountry = useCallback((q) => {
    clearTimeout(countryDebounce.current);
    if (q.length < 2) {
      setCountryResults([]);
      return;
    }
    countryDebounce.current = setTimeout(async () => {
      setSearchingCountry(true);
      const { data } = await geocodeApi.searchCountry(q, { limit: 6 });
      setCountryResults(data ?? []);
      setSearchingCountry(false);
    }, 280);
  }, []);

  // ── Selection handlers ──────────────────────────────────────────────────
  const handleSelectCity = (item) => {
    const parsed = parseNominatimResult(item);
    const cityName = parsed.city || item.display_name.split(",")[0].trim();

    setCityQuery(cityName);
    setCityResults([]);

    // Auto-fill state + country from address components
    if (parsed.state) {
      setStateQuery(parsed.state);
      setStateResults([]);
    }
    if (parsed.country) {
      setCountryQuery(parsed.country);
      setCountryResults([]);
    }

    store.setCity({ name: cityName, lat: parsed.lat, lon: parsed.lon });
    store.setGeocoded({
      ...parsed,
      display_name: [cityName, parsed.state, parsed.country]
        .filter(Boolean)
        .join(", "),
    });
  };

  const handleSelectState = (item) => {
    const parsed = parseNominatimResult(item);
    const stateName = parsed.state || item.display_name.split(",")[0].trim();

    setStateQuery(stateName);
    setStateResults([]);

    // Auto-fill country
    if (parsed.country) {
      setCountryQuery(parsed.country);
      setCountryResults([]);
    }

    store.setState({ name: stateName, lat: parsed.lat, lon: parsed.lon });
    store.setGeocoded({
      ...parsed,
      display_name: [stateName, parsed.country].filter(Boolean).join(", "),
    });
  };

  const handleSelectCountry = (item) => {
    const parsed = parseNominatimResult(item);
    const countryName =
      parsed.country || item.display_name.split(",")[0].trim();

    setCountryQuery(countryName);
    setCountryResults([]);

    store.setCountry({
      name: countryName,
      code: parsed.country_code,
      lat: parsed.lat,
      lon: parsed.lon,
    });
    store.setGeocoded({
      ...parsed,
      display_name: countryName,
    });
  };

  // ── Submit analysis ──────────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!store.geocoded) {
      showNotification("Select a region first", "warning");
      return;
    }

    store.resetRun();
    store.setRunState({ status: "queued", progress: 0 });

    if (isMockMode) {
      simulateMockPolling();
      return;
    }

    const payload = {
      region: {
        center: { lat: store.geocoded.lat, lon: store.geocoded.lon },
        bbox: store.geocoded.bbox,
        boundary_geojson: store.geocoded.boundary_geojson,
        display_name: store.geocoded.display_name,
      },
      date: store.analysisDate || new Date().toISOString().slice(0, 10),
      options: { sensor: "S1_GRD", detector: "unet" },
    };

    const { data, error } = await floodDetectApi.submitDetection(payload);
    if (error) {
      store.setRunState({ status: "failed", error });
      showNotification(error, "error");
      return;
    }

    store.setRunState({ runId: data.run_id, status: data.status });
    startPolling(data.run_id);
  };

  const startPolling = (runId) => {
    const poll = setInterval(async () => {
      const { data, error } = await floodDetectApi.getDetectionStatus(runId);
      if (error) {
        clearInterval(poll);
        store.setRunState({ status: "failed", error });
        showNotification(error, "error");
        return;
      }
      store.setRunState({ status: data.status, progress: data.progress ?? 0 });
      if (data.status === "completed") {
        clearInterval(poll);
        store.setResult(data.result);
        showNotification("Flood detection complete", "success");
      }
      if (data.status === "failed") {
        clearInterval(poll);
        store.setRunState({ error: data.error });
        showNotification(data.error ?? "Detection failed", "error");
      }
    }, 3000);
  };

  const simulateMockPolling = () => {
    const stages = [
      "queued",
      "preprocessing",
      "detecting",
      "scoring",
      "completed",
    ];
    let idx = 0;
    const timer = setInterval(() => {
      idx++;
      if (idx < stages.length) {
        store.setRunState({
          status: stages[idx],
          progress: Math.round((idx / (stages.length - 1)) * 100),
        });
      }
      if (stages[idx] === "completed") {
        clearInterval(timer);
        store.setResult(mockFloodResponse);
        showNotification("Flood detection complete (mock)", "success");
      }
    }, 1500);
  };

  const isRunning = store.isRunning();
  const geo = store.geocoded;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-5 flex flex-col gap-5"
      style={{
        background: "#0a0907",
        border: "1px solid rgba(201,169,110,0.15)",
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between border-b pb-3"
        style={{ borderColor: "rgba(201,169,110,0.15)" }}
      >
        <span
          className="text-[9px] font-mono tracking-[0.3em] uppercase"
          style={{ color: "rgba(201,169,110,0.6)" }}
        >
          Target Coordinates
        </span>
        {isMockMode && (
          <span
            className="text-[8px] font-mono tracking-widest uppercase px-1.5 py-0.5 border"
            style={{ color: "#c0392b", borderColor: "#c0392b" }}
          >
            SIMULATED
          </span>
        )}
      </div>

      {/* Three independent search fields */}
      <GeoSearchInput
        label="City / District"
        placeholder="e.g. Mumbai, New Orleans..."
        value={cityQuery}
        onChange={(q) => {
          setCityQuery(q);
          searchCity(q);
        }}
        results={cityResults}
        onSelect={handleSelectCity}
        isSearching={searchingCity}
        onClear={() => {
          setCityQuery("");
          setCityResults([]);
        }}
      />

      <GeoSearchInput
        label="State / Province"
        placeholder="e.g. Kerala, Louisiana..."
        value={stateQuery}
        onChange={(q) => {
          setStateQuery(q);
          searchState(q);
        }}
        results={stateResults}
        onSelect={handleSelectState}
        isSearching={searchingState}
        onClear={() => {
          setStateQuery("");
          setStateResults([]);
        }}
      />

      <GeoSearchInput
        label="Country"
        placeholder="e.g. India, United States..."
        value={countryQuery}
        onChange={(q) => {
          setCountryQuery(q);
          searchCountry(q);
        }}
        results={countryResults}
        onSelect={handleSelectCountry}
        isSearching={searchingCountry}
        onClear={() => {
          setCountryQuery("");
          setCountryResults([]);
        }}
      />

      {/* Geocoded coordinates readout */}
      <AnimatePresence>
        {geo && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div
              className="px-3 py-3"
              style={{
                border: "1px solid rgba(236,232,223,0.1)",
                background: "rgba(236,232,223,0.03)",
              }}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span
                  className="text-[8px] font-mono uppercase tracking-[0.2em]"
                  style={{ color: "rgba(236,232,223,0.4)" }}
                >
                  Active Target
                </span>
                <span
                  className="text-[8px] font-mono tracking-[0.2em]"
                  style={{ color: "rgba(201,169,110,0.6)" }}
                >
                  {geo.country_code?.toUpperCase() || ""}
                </span>
              </div>
              <div
                className="text-[10px] font-mono uppercase tracking-widest leading-tight truncate mb-2"
                style={{ color: "#ece8df" }}
              >
                {geo.display_name}
              </div>
              <div
                className="flex items-center gap-4 text-[9px] font-mono tracking-widest uppercase"
                style={{ color: "rgba(236,232,223,0.3)" }}
              >
                <span>
                  <span style={{ color: "rgba(201,169,110,0.6)" }}>LAT </span>
                  {geo.lat?.toFixed(4)}
                </span>
                <span>
                  <span style={{ color: "rgba(201,169,110,0.6)" }}>LON </span>
                  {geo.lon?.toFixed(4)}
                </span>
                {geo.bbox && (
                  <span style={{ color: "rgba(201,169,110,0.4)" }}>[BBOX]</span>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Analysis date — Google Calendar–style picker */}
      <CalendarPicker
        label="Analysis Date"
        value={store.analysisDate || ""}
        onChange={(v) => store.setAnalysisDate(v)}
      />

      {/* Analyze button */}
      <button
        onClick={handleAnalyze}
        disabled={!geo || isRunning}
        className="relative group w-full mt-2"
        style={{
          padding: "0.8rem 1rem",
          cursor: !geo || isRunning ? "not-allowed" : "pointer",
          opacity: !geo || isRunning ? 0.5 : 1,
        }}
      >
        <span
          className="absolute inset-0 transition-colors"
          style={{ border: "1px solid rgba(201,169,110,0.4)" }}
        />
        <span
          className="absolute inset-0 translate-x-full group-hover:translate-x-0 transition-transform duration-300"
          style={{ background: "#c0392b" }}
        />

        <span
          className="relative z-10 flex items-center justify-center gap-2 font-mono tracking-[0.2em] uppercase transition-colors"
          style={{ fontSize: "0.65rem", color: "#c9a96e" }}
        >
          {isRunning ?
            <>
              <span className="w-1.5 h-1.5 bg-[#c9a96e] animate-pulse" />
              Initializing Scan...
            </>
          : "Execute Detection"}
        </span>
      </button>
    </motion.div>
  );
}
