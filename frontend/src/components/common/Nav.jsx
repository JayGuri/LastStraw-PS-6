import React, { useState, useRef, useEffect } from "react";
import { useAppStore } from "../../stores/appStore.js";
import { useAuth } from "../../hooks/useAuth.js";

const TABS = [
  { id: "landing", label: "Briefing", icon: "◉", emoji: "⬛" },
  { id: "globe", label: "Theater", icon: "⊕", emoji: "🌍" },
  { id: "mission", label: "Operations", icon: "⬡", emoji: "⚡" },
  { id: "insights", label: "Insights", icon: "◈", emoji: "📊" },
];

const SUBSCRIPTION_TIERS = {
  free: {
    label: "Observer",
    color: "#ece8df",
    border: "rgba(236,232,223,0.2)",
  },
  plus: {
    label: "Analyst",
    color: "#c9a96e",
    border: "rgba(201,169,110,0.4)",
  },
  pro: {
    label: "Sovereign",
    color: "#c0392b",
    border: "rgba(192,57,43,0.4)",
  },
  enterprise: {
    label: "Sovereign",
    color: "#c0392b",
    border: "rgba(192,57,43,0.4)",
  },
};
const defaultTier = SUBSCRIPTION_TIERS.free;

function TierBadge({ level }) {
  const tier = SUBSCRIPTION_TIERS[level] ?? defaultTier;
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 tracking-widest uppercase"
      style={{
        fontSize: "0.55rem",
        fontFamily: "monospace",
        color: tier.color,
        border: `1px solid ${tier.border}`,
      }}
    >
      {tier.label}
    </span>
  );
}

export default function Nav() {
  const { activeTab, setActiveTab, isMockMode, toggleMockMode } = useAppStore();
  const { user, isAuthenticated, logout } = useAuth();
  const [profileOpen, setProfileOpen] = useState(false);
  const profileRef = useRef(null);

  useEffect(() => {
    const handler = (e) => {
      if (profileRef.current && !profileRef.current.contains(e.target))
        setProfileOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const tier = user?.subscription_level ?? "free";

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 h-14 flex items-center px-4 sm:px-6"
      style={{
        background: "#060504",
        borderBottom: "1px solid rgba(201,169,110,0.15)",
      }}
    >
      {/* Brand */}
      <div
        className="flex items-center gap-4 mr-6 sm:mr-10 cursor-pointer"
        onClick={() => setActiveTab("landing")}
      >
        <div>
          <div
            className="font-display font-light tracking-[0.25em] leading-none"
            style={{ color: "#ece8df", fontSize: "1.05rem" }}
          >
            AMBROSIA
          </div>
          <div
            className="mt-1 tracking-[0.3em] uppercase"
            style={{
              fontSize: "0.45rem",
              color: "#c9a96e",
              fontFamily: "monospace",
            }}
          >
            Intelligence Platform
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-1 sm:gap-2 flex-1">
        {TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className="relative px-3 sm:px-4 py-2 flex items-center gap-2 transition-colors duration-300"
              style={{
                color: isActive ? "#c9a96e" : "rgba(236,232,223,0.4)",
              }}
            >
              <span className="md:hidden text-xs leading-none">
                {tab.emoji}
              </span>
              <span
                className="hidden md:inline font-light tracking-[0.2em] uppercase hover:text-[#ece8df] transition-colors"
                style={{
                  fontSize: "0.6rem",
                  fontFamily: "monospace",
                  color: "inherit",
                }}
              >
                {tab.label}
              </span>
              {isActive && (
                <span
                  className="absolute bottom-0 left-0 w-full h-[1px]"
                  style={{ background: "#c0392b" }}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* Right */}
      <div className="flex items-center gap-3 sm:gap-5">
        <button
          onClick={toggleMockMode}
          className="tracking-[0.25em] uppercase transition-colors"
          style={{
            fontSize: "0.55rem",
            fontFamily: "monospace",
            padding: "4px 8px",
            color: isMockMode ? "#c0392b" : "rgba(201,169,110,0.7)",
            border: `1px solid ${
              isMockMode ? "rgba(192,57,43,0.4)" : "rgba(201,169,110,0.2)"
            }`,
          }}
        >
          {isMockMode ? "SIMULATED" : "LIVE"}
        </button>

        {isAuthenticated ?
          <div ref={profileRef} className="relative">
            <button
              onClick={() => setProfileOpen((o) => !o)}
              className="flex items-center gap-3 pl-3 pr-2 py-1 transition-colors"
              style={{
                border: "1px solid rgba(236,232,223,0.15)",
                background:
                  profileOpen ? "rgba(236,232,223,0.05)" : "transparent",
              }}
            >
              <TierBadge level={tier} />
              <div
                className="w-6 h-6 flex items-center justify-center font-display"
                style={{
                  background: "rgba(201,169,110,0.1)",
                  border: "1px solid rgba(201,169,110,0.3)",
                  color: "#c9a96e",
                  fontSize: "0.8rem",
                }}
              >
                {(user?.email ?? "?").charAt(0).toUpperCase()}
              </div>
            </button>

            {profileOpen && (
              <div
                className="absolute right-0 top-full mt-2 w-56 py-2 shadow-2xl"
                style={{
                  background: "#0a0907",
                  border: "1px solid rgba(201,169,110,0.2)",
                }}
              >
                <div
                  className="px-4 py-3 border-b"
                  style={{ borderColor: "rgba(236,232,223,0.1)" }}
                >
                  <p
                    className="font-light truncate"
                    style={{ fontSize: "0.75rem", color: "#ece8df" }}
                  >
                    {user?.email}
                  </p>
                  <p
                    className="mt-1 tracking-widest uppercase"
                    style={{
                      fontSize: "0.5rem",
                      color: "rgba(236,232,223,0.5)",
                      fontFamily: "monospace",
                    }}
                  >
                    ID: {user?.id?.slice(0, 8) || "UNKNOWN"}
                  </p>
                </div>
                <button
                  onClick={() => {
                    logout();
                    setProfileOpen(false);
                  }}
                  className="w-full text-left px-4 py-3 tracking-widest uppercase transition-colors hover:bg-white/5"
                  style={{
                    fontSize: "0.6rem",
                    color: "#c0392b",
                    fontFamily: "monospace",
                  }}
                >
                  DISAVOW (Sign Out)
                </button>
              </div>
            )}
          </div>
        : <button
            onClick={() => setActiveTab("login")}
            className="relative group overflow-hidden"
            style={{ padding: "0.4rem 1.25rem" }}
          >
            <span
              className="absolute inset-0"
              style={{ border: "1px solid rgba(201,169,110,0.4)" }}
            />
            <span
              className="absolute inset-0 translate-x-full group-hover:translate-x-0 transition-transform duration-300"
              style={{ background: "#c9a96e" }}
            />
            <span
              className="relative z-10 tracking-[0.2em] uppercase transition-colors group-hover:text-[#060504]"
              style={{
                fontSize: "0.6rem",
                color: "#c9a96e",
                fontFamily: "monospace",
              }}
            >
              Authenticate
            </span>
          </button>
        }
      </div>
    </nav>
  );
}
