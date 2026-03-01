import React, { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useGlobeStore } from "../../stores/globeStore.js";
import { useAppStore } from "../../stores/appStore.js";
import {
  geocodeApi,
  parseNominatimResult,
  hasAreaGeometry,
  sortResultsByBoundary,
} from "../../api/geocodeApi.js";
import { floodDetectApi } from "../../api/floodDetectApi.js";
import { mockFloodResponse } from "../../data/mockFloodResponse.js";
import GeoSearchInput from "../ui/GeoSearchInput.jsx";

// ─────────────────────────────────────────────────────────────────────────────
export default function RegionForm() {
  const store = useGlobeStore();
  const showNotification = useAppStore((s) => s.showNotification);
  const geocoded = store.geocoded;

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

  // When store is violently cleared (e.g. from tab swap or reset), clear inputs
  React.useEffect(() => {
    if (!geocoded) {
      setCityQuery("");
      setStateQuery("");
      setCountryQuery("");
      setCityResults([]);
      setStateResults([]);
      setCountryResults([]);
    }
  }, [geocoded]);

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
      setCityResults(sortResultsByBoundary(data ?? []));
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
      setStateResults(sortResultsByBoundary(data ?? []));
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
      setCountryResults(sortResultsByBoundary(data ?? []));
      setSearchingCountry(false);
    }, 280);
  }, []);

  // Always fetch the actual admin boundary (Polygon/MultiPolygon) for the selected place
  // so the globe shows the real border (e.g. Bharuch-style outline), not just bbox or point.
  const enrichGeocodedWithBoundary = useCallback(
    (item, currentGeocoded) => {
      const osmType = item.osm_type ?? currentGeocoded?.osm_type;
      const osmId = item.osm_id ?? currentGeocoded?.osm_id;
      if (!osmType || osmId == null) return;
      geocodeApi.lookup(osmType, osmId).then(({ data }) => {
        if (!data?.[0]) return;
        const looked = parseNominatimResult(data[0]);
        if (hasAreaGeometry(looked))
          store.setGeocodedBoundary(looked.boundary_geojson, looked.bbox);
      });
    },
    [store],
  );

  // ── Selection handlers ──────────────────────────────────────────────────
  const handleSelectCity = (item) => {
    const parsed = parseNominatimResult(item);
    const cityName = parsed.city || item.display_name.split(",")[0].trim();

    setCityQuery(cityName);
    setCityResults([]);

    if (parsed.state) {
      setStateQuery(parsed.state);
      setStateResults([]);
    }
    if (parsed.country) {
      setCountryQuery(parsed.country);
      setCountryResults([]);
    }

    store.setCity({ name: cityName, lat: parsed.lat, lon: parsed.lon });
    const geocoded = {
      ...parsed,
      display_name: [cityName, parsed.state, parsed.country]
        .filter(Boolean)
        .join(", "),
    };
    store.setGeocoded(geocoded);
    enrichGeocodedWithBoundary(item, geocoded);
  };

  const handleSelectState = (item) => {
    const parsed = parseNominatimResult(item);
    const stateName = parsed.state || item.display_name.split(",")[0].trim();

    setStateQuery(stateName);
    setStateResults([]);

    if (parsed.country) {
      setCountryQuery(parsed.country);
      setCountryResults([]);
    }

    store.setState({ name: stateName, lat: parsed.lat, lon: parsed.lon });
    const geocoded = {
      ...parsed,
      display_name: [stateName, parsed.country].filter(Boolean).join(", "),
    };
    store.setGeocoded(geocoded);
    enrichGeocodedWithBoundary(item, geocoded);
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
    const geocoded = { ...parsed, display_name: countryName };
    store.setGeocoded(geocoded);
    enrichGeocodedWithBoundary(item, geocoded);
  };

  // Compute tomorrow's date for forecast
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  const forecastDate = tomorrow.toISOString().slice(0, 10);

  const handleConfirmArea = () => store.setRegionConfirmed(true);
  const handleChooseDifferent = () => {
    store.clearRegionSelection();
    setCityQuery("");
    setStateQuery("");
    setCountryQuery("");
    setCityResults([]);
    setStateResults([]);
    setCountryResults([]);
  };

  // ── Submit analysis ──────────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!store.geocoded || !store.regionConfirmed) {
      showNotification(
        store.geocoded ?
          "Confirm the highlighted area first"
        : "Select a region first",
        "warning",
      );
      return;
    }

    store.resetRun();
    store.setRunState({ status: "queued", progress: 0 });

    const payload = {
      lat: store.geocoded.lat,
      lon: store.geocoded.lon,
    };

    store.setRunState({ status: "detecting", progress: 50 });

    const { data, error } = await floodDetectApi.submitDetection(payload);

    if (error) {
      showNotification(
        "Live detection failed. Falling back to simulated run.",
        "warning",
      );

      store.resetRun();
      store.setRunState({ status: "queued", progress: 0 });

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
          showNotification("Simulated flood detection complete", "success");
        }
      }, 1500);

      return;
    }

    store.setRunState({ status: "completed", progress: 100 });
    store.setResult(data);
    showNotification("Flood forecast complete", "success");
  };

  const isRunning = store.isRunning();
  const geo = store.geocoded;
  const regionConfirmed = store.regionConfirmed;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-5 flex flex-col gap-5"
      style={{
        background: "#0a0907",
        border: "1px solid rgba(242,209,109,0.15)",
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between border-b pb-3"
        style={{ borderColor: "rgba(242,209,109,0.15)" }}
      >
        <span
          className="text-[9px] font-mono tracking-[0.3em] uppercase"
          style={{ color: "rgba(242,209,109,0.6)" }}
        >
          Target Coordinates
        </span>
      </div>

      {geo && !regionConfirmed && (
        <p
          className="text-[10px] font-mono tracking-wide"
          style={{ color: "rgba(34,197,94,0.9)" }}
        >
          The green area on the globe is your selection. Confirm or choose
          another.
        </p>
      )}

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

      {/* Geocoded + confirm step */}
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
                  {regionConfirmed ? "Confirmed area" : "Area preview"}
                </span>
                <span
                  className="text-[8px] font-mono tracking-[0.2em]"
                  style={{ color: "rgba(242,209,109,0.6)" }}
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
                  <span style={{ color: "rgba(242,209,109,0.6)" }}>LAT </span>
                  {geo.lat?.toFixed(4)}
                </span>
                <span>
                  <span style={{ color: "rgba(242,209,109,0.6)" }}>LON </span>
                  {geo.lon?.toFixed(4)}
                </span>
                {geo.bbox && (
                  <span style={{ color: "rgba(242,209,109,0.4)" }}>[BBOX]</span>
                )}
              </div>

              {!regionConfirmed ?
                <div className="flex gap-2 mt-3">
                  <button
                    type="button"
                    onClick={handleConfirmArea}
                    className="flex-1 py-2 text-[10px] font-mono uppercase tracking-wider border transition-colors"
                    style={{
                      borderColor: "#22c55e",
                      color: "#22c55e",
                      background: "rgba(34,197,94,0.12)",
                    }}
                  >
                    Confirm area
                  </button>
                  <button
                    type="button"
                    onClick={handleChooseDifferent}
                    className="flex-1 py-2 text-[10px] font-mono uppercase tracking-wider border transition-colors"
                    style={{
                      borderColor: "rgba(236,232,223,0.25)",
                      color: "rgba(236,232,223,0.7)",
                    }}
                  >
                    Choose different
                  </button>
                </div>
              : <button
                  type="button"
                  onClick={() => store.setRegionConfirmed(false)}
                  className="mt-2 text-[9px] font-mono uppercase tracking-wider"
                  style={{ color: "rgba(242,209,109,0.7)" }}
                >
                  Change area
                </button>
              }
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Tomorrow's Forecast Window */}
      <div
        className="px-3 py-2 border"
        style={{
          borderColor: "rgba(242,209,109,0.15)",
          background: "rgba(242,209,109,0.04)",
        }}
      >
        <div
          className="text-[8px] font-mono uppercase tracking-[0.25em] mb-1"
          style={{ color: "rgba(242,209,109,0.5)" }}
        >
          Forecast Window
        </div>
        <div
          className="text-[11px] font-mono tracking-widest"
          style={{ color: "#f2d16d" }}
        >
          {forecastDate} · 24H
        </div>
        <div
          className="text-[9px] font-mono mt-0.5"
          style={{ color: "rgba(236,232,223,0.3)" }}
        >
          Tomorrow's flood risk forecast
        </div>
      </div>

      {/* Analyze button — only enabled after area confirmed */}
      <button
        onClick={handleAnalyze}
        disabled={!geo || !regionConfirmed || isRunning}
        className="relative group w-full mt-2 overflow-hidden"
        style={{
          padding: "0.8rem 1rem",
          cursor:
            !geo || !regionConfirmed || isRunning ? "not-allowed" : "pointer",
          opacity: !geo || !regionConfirmed || isRunning ? 0.5 : 1,
        }}
      >
        <span
          className="absolute inset-0 transition-colors"
          style={{ border: "1px solid rgba(242,209,109,0.4)" }}
        />
        <span
          className="absolute inset-0 translate-x-full group-hover:translate-x-0 transition-transform duration-300"
          style={{ background: "#f2d16d" }}
        />

        <span
          className="relative z-10 flex items-center justify-center gap-2 font-mono tracking-[0.2em] uppercase transition-colors text-[#f2d16d] group-hover:text-[#0a0907]"
          style={{ fontSize: "0.65rem" }}
        >
          {isRunning ?
            <>
              <span className="w-1.5 h-1.5 bg-[#f2d16d] group-hover:bg-[#0a0907] animate-pulse" />
              Generating Forecast...
            </>
          : !regionConfirmed && geo ?
            "Confirm area above first"
          : "Run Forecast"}
        </span>
      </button>
    </motion.div>
  );
}
