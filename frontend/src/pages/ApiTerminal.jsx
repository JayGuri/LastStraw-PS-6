import React, { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useApiStore } from '../stores/apiStore.js'
import { ENDPOINTS, ERROR_CODES } from '../data/endpoints.js'

const METHOD_COLORS = {
  POST: { bg: 'bg-gold/10',      border: 'border-gold/25',  text: 'text-gold' },
  GET:  { bg: 'bg-ice/[0.08]',   border: 'border-ice/20',   text: 'text-ice'  },
}

const RUN_STEPS = ['queued','preprocessing','detecting','scoring','completed']

function JsonSyntax({ text }) {
  if (!text) return null
  const colored = text
    .replace(/("(?:[^"\\]|\\.)*")\s*:/g, '<span style="color:#e8ab30">$1</span>:')
    .replace(/:\s*("(?:[^"\\]|\\.)*")/g, ': <span style="color:#4ab0d8">$1</span>')
    .replace(/:\s*(\d+\.?\d*)/g, ': <span style="color:#d06828">$1</span>')
    .replace(/:\s*(true|false|null)/g, ': <span style="color:#d84040">$1</span>')

  return (
    <pre
      className="text-[10px] sm:text-xs font-mono text-text leading-relaxed whitespace-pre-wrap break-all"
      dangerouslySetInnerHTML={{ __html: colored }}
    />
  )
}

function CopyButton({ getText }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard?.writeText(getText())
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      onClick={handleCopy}
      className={`absolute top-2 sm:top-3 right-2 sm:right-3 text-[9px] sm:text-[10px] font-mono
                 px-1.5 sm:px-2 py-0.5 sm:py-1 rounded border transition-all
                 ${copied
                   ? 'bg-low/10 border-low/30 text-low'
                   : 'text-text-3 hover:text-text bg-white/5 border-white/5 hover:border-white/10'}`}
    >
      {copied ? 'COPIED ✓' : 'copy'}
    </button>
  )
}

function EndpointCard({ ep, isActive, onClick }) {
  const mc = METHOD_COLORS[ep.method] ?? METHOD_COLORS.GET
  return (
    <button onClick={onClick}
      className={`w-full text-left p-3 sm:p-4 rounded-xl border transition-all duration-200
        ${isActive
          ? 'card-glass border-gold/20 shadow-gold-sm'
          : 'bg-bg-card border-white/5 hover:border-white/10 hover:bg-bg-3'}`}
    >
      <div className="flex items-start gap-2 sm:gap-3">
        <span className={`text-[10px] sm:text-xs font-mono font-bold px-1.5 sm:px-2 py-0.5 rounded border flex-shrink-0 mt-0.5
                          ${mc.bg} ${mc.border} ${mc.text}`}>
          {ep.method}
        </span>
        <div className="min-w-0">
          <div className="text-[10px] sm:text-xs font-mono text-text/70 mb-0.5 truncate">{ep.path}</div>
          <div className="font-medium text-xs sm:text-sm text-text">{ep.summary}</div>
          <div className="text-[10px] sm:text-xs text-text-3 mt-0.5 line-clamp-2">{ep.description}</div>
        </div>
      </div>
    </button>
  )
}

export default function ApiTerminal() {
  const {
    activeEndpoint, setActiveEndpoint,
    activeTab, setActiveTab,
    testerRunning, testerOutput, testerStep,
    runTester, clearTester,
  } = useApiStore()

  const ep = ENDPOINTS.find(e => e.id === activeEndpoint) ?? ENDPOINTS[0]
  const outputRef = useRef(null)

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [testerOutput])

  return (
    <div className="min-h-screen bg-bg noise pt-14 sm:pt-16">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 py-4 sm:py-8">

        {/* Header */}
        <div className="mb-5 sm:mb-8">
          <div className="section-tag section-tag-ice mb-2 sm:mb-3 text-[10px] sm:text-xs">API Reference</div>
          <h1 className="font-display font-extrabold text-2xl sm:text-4xl text-text">
            API Terminal
          </h1>
          <p className="text-text-2 text-xs sm:text-sm mt-1.5 sm:mt-2 flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-2">
            <span className="font-mono text-ice bg-ice/5 border border-ice/15 px-1.5 sm:px-2 py-0.5 rounded
                             text-[10px] sm:text-xs inline-block truncate max-w-full">
              https://api.cosmeon.io/api/v1
            </span>
            <span className="text-[10px] sm:text-xs">REST API for flood detection, district risk, and pipeline management.</span>
          </p>
        </div>

        <div className="grid lg:grid-cols-12 gap-4 sm:gap-6">

          {/* Left — endpoint list */}
          <div className="lg:col-span-3 space-y-1.5 sm:space-y-2">
            <div className="data-tag text-text-3 px-1 mb-2 sm:mb-3">
              Endpoints
            </div>
            {/* On mobile, show as horizontal scroll list; on lg+ show vertical */}
            <div className="flex lg:flex-col gap-1.5 sm:gap-2 overflow-x-auto lg:overflow-x-visible pb-2 lg:pb-0
                            scrollbar-none -mx-1 px-1">
              {ENDPOINTS.map(e => (
                <div key={e.id} className="min-w-[200px] sm:min-w-[240px] lg:min-w-0 flex-shrink-0 lg:flex-shrink">
                  <EndpointCard
                    ep={e}
                    isActive={e.id === activeEndpoint}
                    onClick={() => { setActiveEndpoint(e.id); clearTester() }}
                  />
                </div>
              ))}
            </div>

            {/* Error codes mini-table */}
            <div className="mt-4 sm:mt-6 bg-bg-card rounded-xl p-3 sm:p-4 border border-white/5
                            hidden lg:block">
              <div className="data-tag text-text-3 mb-2 sm:mb-3">
                Error Codes
              </div>
              {ERROR_CODES.map((e, i) => (
                <div key={e.code}
                     className={`py-1.5 sm:py-2 border-b border-white/[0.03] last:border-0 px-1 rounded transition-colors
                                 hover:bg-white/[0.02]
                                 ${i % 2 === 0 ? '' : 'bg-white/[0.01]'}`}>
                  <div className="flex items-center gap-1.5 sm:gap-2">
                    <span className="text-[9px] sm:text-[10px] font-mono text-critical/80 bg-critical/5
                                     border border-critical/15 px-1 sm:px-1.5 py-0.5 rounded">
                      {e.status}
                    </span>
                    <span className="text-[9px] sm:text-[10px] font-mono text-gold/80 truncate">{e.code}</span>
                  </div>
                  <div className="text-[9px] sm:text-[10px] text-text-3 mt-0.5">{e.desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Middle — endpoint detail */}
          <div className="lg:col-span-5 space-y-3 sm:space-y-5">
            <AnimatePresence mode="wait">
              <motion.div
                key={ep.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                {/* Title + badge */}
                <div className="flex items-start gap-2 sm:gap-3 mb-3 sm:mb-4">
                  <span className={`text-[10px] sm:text-sm font-mono font-bold px-2 sm:px-3 py-0.5 sm:py-1 rounded-lg border flex-shrink-0
                                    ${METHOD_COLORS[ep.method]?.bg}
                                    ${METHOD_COLORS[ep.method]?.border}
                                    ${METHOD_COLORS[ep.method]?.text}`}>
                    {ep.method}
                  </span>
                  <div className="min-w-0">
                    <div className="font-mono text-sm sm:text-lg font-bold text-text break-all">{ep.path}</div>
                    <div className="text-[10px] sm:text-sm text-text-2 mt-1">{ep.description}</div>
                  </div>
                </div>

                {/* Response status */}
                <div className="flex items-center gap-2 sm:gap-3 mb-3 sm:mb-4 flex-wrap">
                  <span className="text-[10px] sm:text-xs font-mono bg-low/10 border border-low/20 text-low px-1.5 sm:px-2 py-0.5 sm:py-1 rounded">
                    {ep.statusCode} {ep.statusText}
                  </span>
                  <span className="text-[9px] sm:text-xs text-text-3 font-mono">
                    rate limit: 10 req/min (Phase 2)
                  </span>
                </div>

                {/* Parameters table */}
                {ep.params.length > 0 && (
                  <div className="bg-bg-card rounded-xl border border-white/5 overflow-hidden mb-3 sm:mb-4">
                    <div className="px-3 sm:px-4 py-1.5 sm:py-2 border-b border-white/5 text-[10px] sm:text-xs font-mono text-text-2">
                      Parameters
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-[10px] sm:text-xs" style={{ minWidth: 400 }}>
                        <thead>
                          <tr className="border-b border-white/5" style={{ background: 'rgba(255,255,255,0.03)' }}>
                            <th className="text-left px-3 sm:px-4 py-1.5 sm:py-2 text-text-3 font-mono font-normal whitespace-nowrap">Name</th>
                            <th className="text-left px-3 sm:px-4 py-1.5 sm:py-2 text-text-3 font-mono font-normal whitespace-nowrap">Type</th>
                            <th className="text-left px-3 sm:px-4 py-1.5 sm:py-2 text-text-3 font-mono font-normal whitespace-nowrap">Req</th>
                            <th className="text-left px-3 sm:px-4 py-1.5 sm:py-2 text-text-3 font-mono font-normal">Description</th>
                          </tr>
                        </thead>
                        <tbody>
                          {ep.params.map(p => (
                            <tr key={p.name} className="border-b border-white/[0.03] last:border-0 hover:bg-white/[0.01]">
                              <td className="px-3 sm:px-4 py-1.5 sm:py-2 font-mono text-gold/80 whitespace-nowrap">{p.name}</td>
                              <td className="px-3 sm:px-4 py-1.5 sm:py-2 font-mono text-ice/60 whitespace-nowrap">{p.type}</td>
                              <td className="px-3 sm:px-4 py-1.5 sm:py-2">
                                {p.required
                                  ? <span className="font-mono text-[9px] sm:text-[10px] px-1.5 py-0.5 rounded border
                                                     bg-critical/10 text-critical border-critical/20">yes</span>
                                  : <span className="text-text-3 font-mono">no</span>}
                              </td>
                              <td className="px-3 sm:px-4 py-1.5 sm:py-2 text-text-2">{p.desc}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Example tabs */}
                <div className="rounded-xl border border-white/5 overflow-hidden"
                     style={{ background: '#050810' }}>
                  <div className="flex border-b border-white/5 bg-bg-card">
                    {['curl','python','response'].map(t => (
                      <button key={t}
                        onClick={() => setActiveTab(t)}
                        className={`px-3 sm:px-4 py-2 sm:py-2.5 text-[10px] sm:text-xs font-mono transition-all
                          ${activeTab === t
                            ? 'card-glass shadow-gold-sm text-gold border-b-2 border-gold'
                            : 'text-text-3 hover:text-text-2'}`}
                      >
                        {t === 'curl' ? 'cURL' : t === 'python' ? 'Python' : 'Response'}
                      </button>
                    ))}
                  </div>
                  <div className="relative">
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={`${ep.id}-${activeTab}`}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.15 }}
                      >
                        {activeTab !== 'response' ? (
                          <pre className="p-3 sm:p-4 text-[10px] sm:text-xs font-mono text-text-2 overflow-x-auto leading-relaxed
                                          whitespace-pre-wrap max-h-48 sm:max-h-64">
                            {activeTab === 'curl' ? ep.curlExample : ep.pythonExample}
                          </pre>
                        ) : (
                          <div className="p-3 sm:p-4 max-h-48 sm:max-h-64 overflow-auto">
                            <JsonSyntax text={JSON.stringify(ep.response, null, 2)} />
                          </div>
                        )}
                      </motion.div>
                    </AnimatePresence>
                    <CopyButton getText={() =>
                      activeTab === 'curl' ? ep.curlExample
                        : activeTab === 'python' ? ep.pythonExample
                        : JSON.stringify(ep.response, null, 2)
                    } />
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Right — live tester */}
          <div className="lg:col-span-4 space-y-3 sm:space-y-4">
            <div className="bg-bg-card rounded-xl border border-white/5 overflow-hidden flex flex-col">
              {/* Terminal title bar — macOS chrome */}
              <div className="flex items-center gap-2 px-3 sm:px-4 py-2 sm:py-3 bg-bg-2 border-b border-white/5">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full" style={{ background: '#ff5f57' }} />
                  <div className="w-3 h-3 rounded-full" style={{ background: '#febc2e' }} />
                  <div className="w-3 h-3 rounded-full" style={{ background: '#28c840' }} />
                </div>
                <span className="text-[10px] sm:text-xs font-mono text-text-3 ml-1 sm:ml-2 truncate">
                  cosmeon — api-terminal
                </span>
                <div className="ml-auto flex items-center gap-2 flex-shrink-0">
                  {testerRunning && (
                    <span className="text-[9px] sm:text-[10px] font-mono text-gold animate-pulse">● RUNNING</span>
                  )}
                  {ep.id === 'submit' && testerStep >= 0 && !testerRunning && (
                    <span className="text-[9px] sm:text-[10px] font-mono text-low">✓ DONE</span>
                  )}
                </div>
              </div>

              {/* Pipeline step tracker (only for submit endpoint while active) */}
              {ep.id === 'submit' && testerStep >= 0 && (
                <div className="px-3 sm:px-4 py-1.5 sm:py-2 bg-bg border-b border-white/5
                                flex items-center gap-1.5 sm:gap-2 overflow-x-auto scrollbar-none">
                  {RUN_STEPS.map((step, i) => (
                    <React.Fragment key={step}>
                      <span className={`whitespace-nowrap transition-all
                        ${i < testerStep
                          ? 'text-[8px] sm:text-[9px] font-mono text-low/70 line-through decoration-low/40'
                          : i === testerStep
                          ? 'text-[8px] sm:text-[9px] font-mono bg-gold/15 text-gold border border-gold/30 rounded px-1.5 py-0.5'
                          : 'text-[8px] sm:text-[9px] font-mono text-text-3'}`}>
                        {i < testerStep ? '✓' : i === testerStep ? '●' : '○'} {step}
                      </span>
                      {i < RUN_STEPS.length - 1 && (
                        <span className="text-text-3 text-[8px] sm:text-[9px] flex-shrink-0">›</span>
                      )}
                    </React.Fragment>
                  ))}
                </div>
              )}

              {/* Output area */}
              <div ref={outputRef}
                   className="flex-1 p-3 sm:p-4 overflow-y-auto bg-bg font-mono text-[10px] sm:text-xs
                              leading-relaxed min-h-[200px] sm:min-h-[256px] max-h-[320px] sm:max-h-[480px]"
              >
                {!testerOutput && !testerRunning && (
                  <div className="text-text-3 text-[10px] sm:text-xs">
                    <span className="text-gold">$</span> Click <span className="text-gold">Send Request</span> to
                    execute <span className="text-ice break-all">{ep.method} {ep.path}</span>
                  </div>
                )}
                {testerOutput && (
                  <div className="text-text/80 whitespace-pre-wrap break-all">
                    {testerOutput}
                    {testerRunning && <span className="cursor" />}
                  </div>
                )}
              </div>

              {/* Status bar */}
              <div className="px-3 sm:px-4 py-1 sm:py-1.5 bg-bg border-t border-white/[0.04]
                              flex items-center gap-2 text-[9px] sm:text-[10px] font-mono text-text-3">
                <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${testerRunning ? 'bg-gold animate-pulse' : testerOutput ? 'bg-low' : 'bg-white/10'}`} />
                <span>{testerRunning ? 'Executing request…' : testerOutput ? `${ep.method} ${ep.path}` : 'Ready'}</span>
                {testerOutput && !testerRunning && (
                  <span className="ml-auto text-low">200 OK</span>
                )}
              </div>

              {/* Action bar */}
              <div className="px-3 sm:px-4 py-2 sm:py-3 bg-bg-2 border-t border-white/5
                              flex items-center gap-2 sm:gap-3">
                <button
                  disabled={testerRunning}
                  onClick={() => runTester(ep.id)}
                  className={`flex-1 py-2 sm:py-2.5 rounded-lg text-xs sm:text-sm font-display font-bold transition-all
                    ${testerRunning
                      ? 'bg-white/5 text-text-3 cursor-not-allowed'
                      : 'bg-gold text-bg hover:bg-gold-lt shadow-gold-sm hover:shadow-gold-md'}`}
                >
                  {testerRunning ? '⟳ Executing…' : '▶  Send Request'}
                </button>
                <button
                  onClick={clearTester}
                  className="px-2.5 sm:px-3 py-2 sm:py-2.5 rounded-lg border border-white/10 text-text-3
                             hover:text-text hover:border-white/20 transition-colors
                             text-[10px] sm:text-xs font-mono"
                >
                  Clear
                </button>
              </div>
            </div>

            {/* Schema info */}
            <div className="card-glass rounded-xl p-3 sm:p-4">
              <div className="data-tag text-text-3 mb-2 sm:mb-3">
                Response Schema
              </div>
              {[
                { field: 'run_id',         type: 'uuid',   desc: 'Unique pipeline run identifier' },
                { field: 'status',         type: 'enum',   desc: 'queued|preprocessing|detecting|scoring|completed|failed' },
                { field: 'flood_area_km2', type: 'float',  desc: 'Total new inundated area in km²' },
                { field: 'risk_level',     type: 'enum',   desc: 'Low | Medium | High | Critical' },
                { field: 'confidence',     type: 'float',  desc: 'Model output probability 0.0–1.0' },
                { field: 'geotiff_url',    type: 'string', desc: 'Presigned S3 URL (TTL: 900s)' },
              ].map(s => (
                <div key={s.field}
                     className="flex items-start gap-1.5 sm:gap-2 py-1.5 sm:py-2
                                border-b border-white/[0.03] last:border-0">
                  <span className="text-gold font-mono text-[9px] sm:text-[10px]
                                   min-w-[80px] sm:min-w-[100px] flex-shrink-0">{s.field}</span>
                  <span className="text-ice font-mono text-[9px] sm:text-[10px]
                                   min-w-[36px] sm:min-w-[42px] flex-shrink-0">{s.type}</span>
                  <span className="text-text-2 text-[9px] sm:text-[10px] leading-snug">{s.desc}</span>
                </div>
              ))}
            </div>

            {/* Error codes — shown on mobile (hidden on lg where it's in sidebar) */}
            <div className="lg:hidden bg-bg-card rounded-xl p-3 sm:p-4 border border-white/5">
              <div className="data-tag text-text-3 mb-2 sm:mb-3">
                Error Codes
              </div>
              {ERROR_CODES.map((e, i) => (
                <div key={e.code}
                     className={`py-1.5 sm:py-2 border-b border-white/[0.03] last:border-0 px-1 rounded transition-colors
                                 hover:bg-white/[0.02]
                                 ${i % 2 === 0 ? '' : 'bg-white/[0.01]'}`}>
                  <div className="flex items-center gap-1.5 sm:gap-2">
                    <span className="text-[9px] sm:text-[10px] font-mono text-critical/80 bg-critical/5
                                     border border-critical/15 px-1 sm:px-1.5 py-0.5 rounded">
                      {e.status}
                    </span>
                    <span className="text-[9px] sm:text-[10px] font-mono text-gold/80 truncate">{e.code}</span>
                  </div>
                  <div className="text-[9px] sm:text-[10px] text-text-3 mt-0.5">{e.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
