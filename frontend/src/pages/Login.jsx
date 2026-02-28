import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import Spline from "@splinetool/react-spline";
import { useAppStore } from "../stores/appStore.js";

export default function Login() {
  const { setActiveTab, showNotification } = useAppStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Aggressive Spline watermark removal hook
  useEffect(() => {
    const removeWatermark = () => {
      document.querySelectorAll("spline-viewer").forEach((viewer) => {
        const logo = viewer.shadowRoot?.querySelector("#logo");
        if (logo) logo.remove();
      });
      document
        .querySelectorAll('a[href*="spline.design"], a[href*="splinetool"]')
        .forEach((a) => {
          a.remove();
        });
    };
    removeWatermark();
    const int1 = setInterval(removeWatermark, 100);
    const int2 = setTimeout(() => clearInterval(int1), 5000);
    return () => clearInterval(int1);
  }, []);

  const handleSignIn = async (e) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) {
      showNotification("Please enter credentials to continue", "warning");
      return;
    }

    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      showNotification(`Welcome back, ${email.split("@")[0]}`, "success");
      setTimeout(() => setActiveTab("dashboard"), 300);
    }, 1200);
  };

  return (
    <div className="flex w-full h-screen bg-[#020617] overflow-hidden font-sans relative">
      {/* ── Top-Left Brand Overview ── */}
      <div className="absolute top-20 left-20 sm:left-12 xl:left-20 z-50 pointer-events-none">
        <div className="font-display font-bold text-3xl tracking-[0.2em] text-white leading-none uppercase">
          Ambrosia
        </div>
      </div>

      {/* ── Left Side: Spline Art & Branding (60%) ── */}
      <div className="relative flex-1 hidden lg:block overflow-hidden pointer-events-auto">
        {/* Large Spline Canvas (Cropped to hide watermark) */}
        <div className="absolute top-0 left-0 w-full h-[calc(100vh+80px)]">
          <Spline scene="https://prod.spline.design/OnP8WcwVxeq8dv0g/scene.splinecode" />
        </div>

        {/* Subtle vignette so the art blends smoothly */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_0%,#020617_120%)] opacity-80 pointer-events-none" />

        {/* Left-Aligned Value Proposition overlaid on the art */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="absolute bottom-16 left-12 xl:left-20 pointer-events-none"
        >
          <h2 className="text-white font-bold text-3xl xl:text-4xl tracking-wide mb-2 max-w-lg leading-tight">
            Real-Time Climate Risk Intelligence.
          </h2>
          <p className="text-white/60 text-lg max-w-md">
            Powered by next-generation satellite data and neural analytics.
          </p>
        </motion.div>
      </div>

      {/* ── Right Side: Glass Login Column (40%) ── */}
      <div className="relative w-full lg:w-[460px] xl:w-[500px] h-full bg-[#020617]/40 sm:bg-black/60 backdrop-blur-3xl border-l border-white/10 flex flex-col justify-center px-10 sm:px-14 z-10 shrink-0">
        {/* Subtle ambient lighting inside the login column */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="relative w-full max-w-[360px] mx-auto"
        >
          <div className="mb-10">
            <h1 className="text-3xl font-bold text-white mb-2 tracking-wide">
              Sign In
            </h1>
            <p className="text-white/40 text-[13px] font-medium">
              Enter your email and password to access the platform.
            </p>
          </div>

          <form onSubmit={handleSignIn} className="w-full flex flex-col gap-6">
            <div className="flex flex-col gap-2 focus-within:text-blue-400 text-white/50 transition-colors">
              <label className="text-[11px] uppercase tracking-widest font-semibold px-1">
                Email
              </label>
              <div className="relative">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@company.com"
                  className="w-full bg-black/40 border border-white/10 text-white rounded-xl px-4 py-3.5 text-sm outline-none transition-all placeholder:text-white/20 focus:bg-black/60 focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 shadow-inner"
                />
              </div>
            </div>

            <div className="flex flex-col gap-2 focus-within:text-blue-400 text-white/50 transition-colors">
              <label className="text-[11px] uppercase tracking-widest font-semibold px-1">
                Password
              </label>
              <div className="relative">
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••"
                  className="w-full bg-black/40 border border-white/10 text-white rounded-xl px-4 py-3.5 text-sm outline-none transition-all placeholder:text-white/20 focus:bg-black/60 focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 shadow-inner"
                />
              </div>
            </div>

            <button
              id="login-form-submit"
              disabled={isLoading}
              type="submit"
              className="group relative w-full mt-4 bg-white/10 text-white font-medium rounded-xl py-3.5 text-sm hover:bg-white/20 transition-all flex justify-center items-center disabled:opacity-50 disabled:cursor-not-allowed border border-white/10 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-blue-500/20 to-teal-500/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

              <span className="relative z-10 flex items-center gap-2">
                {isLoading ?
                  <svg
                    className="animate-spin h-4 w-4 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                : "Sign In"}
              </span>
            </button>
          </form>
        </motion.div>
      </div>

      {/* Mobile Spline Fallback Background */}
      <div className="absolute inset-0 lg:hidden pointer-events-none -z-10 overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-[calc(100vh+80px)] opacity-50">
          <Spline scene="https://prod.spline.design/OnP8WcwVxeq8dv0g/scene.splinecode" />
        </div>
        <div className="absolute inset-0 bg-[#020617]/80" />
      </div>
    </div>
  );
}
