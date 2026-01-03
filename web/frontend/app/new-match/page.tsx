"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  api,
  ModelInfo,
  TimeControlPreset,
  MatchConfig,
  ProcessDetail,
} from "@/lib/api";

export default function NewMatchPage() {
  const router = useRouter();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [timePresets, setTimePresets] = useState<TimeControlPreset[]>([]);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Process monitoring
  const [activePid, setActivePid] = useState<number | null>(null);
  const [processDetail, setProcessDetail] = useState<ProcessDetail | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Form state
  const [modelA, setModelA] = useState("");
  const [modelB, setModelB] = useState("");
  const [timePreset, setTimePreset] = useState("");
  const [customTime, setCustomTime] = useState(false);
  const [initialTime, setInitialTime] = useState(300);
  const [increment, setIncrement] = useState(3);
  const [firstTo, setFirstTo] = useState(1);
  const [useRethinking, setUseRethinking] = useState(true);
  const [maxRethinks, setMaxRethinks] = useState(2);
  const [maxParsingFailures, setMaxParsingFailures] = useState(3);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Per-model reasoning configuration
  const [reasoningBudgetA, setReasoningBudgetA] = useState(8000);
  const [reasoningBudgetB, setReasoningBudgetB] = useState(8000);
  const [showReasoningA, setShowReasoningA] = useState(false);
  const [showReasoningB, setShowReasoningB] = useState(false);

  useEffect(() => {
    async function fetchConfig() {
      try {
        const config = await api.getConfig();
        setModels(config.models);
        setTimePresets(config.time_control_presets);
        
        // Set defaults
        if (config.models.length >= 2) {
          setModelA(config.models[0].id);
          setModelB(config.models[1].id);
        }
        if (config.time_control_presets.length > 0) {
          const defaultPreset = config.time_control_presets.find(p => p.id === "blitz-5") 
            || config.time_control_presets[0];
          setTimePreset(defaultPreset.id);
          setInitialTime(defaultPreset.initial_time);
          setIncrement(defaultPreset.increment);
        }
      } catch (err) {
        console.error("Failed to fetch config:", err);
        setError("Failed to load configuration");
      } finally {
        setLoading(false);
      }
    }
    fetchConfig();
  }, []);

  // Poll for process status when we have an active PID
  useEffect(() => {
    if (!activePid) return;

    let retryCount = 0;
    const maxRetries = 3;

    const pollStatus = async () => {
      try {
        const detail = await api.getProcessDetail(activePid);
        retryCount = 0; // Reset on success
        setProcessDetail(detail);
        
        // Auto-scroll logs
        logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
        
        // Stop polling if process is done
        if (detail.status === "completed" || detail.status === "failed" || detail.status === "stopped") {
          setStarting(false);
        }
      } catch (err: unknown) {
        retryCount++;
        console.error("Failed to fetch process status:", err);
        
        // If we get 404 multiple times, the process is gone (server restarted or process ended)
        if (retryCount >= maxRetries) {
          console.log("Process not found, clearing state");
          setActivePid(null);
          setProcessDetail(null);
          setStarting(false);
          setError("Process no longer tracked (server may have restarted). Check the Matches page for results.");
        }
      }
    };

    // Initial fetch
    pollStatus();
    
    // Poll every 2 seconds
    const interval = setInterval(pollStatus, 2000);
    
    return () => clearInterval(interval);
  }, [activePid]);

  const handleTimePresetChange = (presetId: string) => {
    setTimePreset(presetId);
    const preset = timePresets.find(p => p.id === presetId);
    if (preset) {
      setInitialTime(preset.initial_time);
      setIncrement(preset.increment);
      setCustomTime(false);
    }
  };

  const handleStartMatch = async () => {
    if (!modelA || !modelB) {
      setError("Please select both models");
      return;
    }
    // Note: Same model vs itself is allowed - useful for testing different reasoning budgets

    setStarting(true);
    setError(null);
    setProcessDetail(null);

    try {
      const config: MatchConfig = {
        model_a: modelA,
        model_b: modelB,
        initial_time_seconds: initialTime,
        increment_seconds: increment,
        first_to: firstTo,
        use_rethinking: useRethinking,
        max_rethinks: maxRethinks,
        max_parsing_failures: maxParsingFailures,
        reasoning_budget_a: reasoningBudgetA,
        reasoning_budget_b: reasoningBudgetB,
        show_reasoning_a: showReasoningA,
        show_reasoning_b: showReasoningB,
      };

      const result = await api.startMatch(config);
      
      if (result.status === "error") {
        setError(result.error || result.message);
        setStarting(false);
        return;
      }
      
      // Start monitoring the process
      if (result.process_id) {
        setActivePid(result.process_id);
      }
    } catch (err) {
      console.error("Failed to start match:", err);
      setError("Failed to start match. Check that API keys are configured.");
      setStarting(false);
    }
  };
  
  const handleStopMatch = async () => {
    if (!activePid) return;
    
    try {
      await api.stopProcess(activePid);
      setStarting(false);
    } catch (err) {
      console.error("Failed to stop match:", err);
    }
  };

  const resetForm = () => {
    setActivePid(null);
    setProcessDetail(null);
    setStarting(false);
    setError(null);
  };

  // Group models by provider
  const modelsByProvider = models.reduce((acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = [];
    }
    acc[model.provider].push(model);
    return acc;
  }, {} as Record<string, ModelInfo[]>);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-gray-400">Loading configuration...</div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto space-y-8 animate-fade-in">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-2">New Match</h1>
        <p className="text-gray-400">Configure and launch a chess battle between AI models</p>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          {error}
        </div>
      )}

      {/* Model Selection */}
      <div className="card p-6 space-y-6">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <span className="text-2xl">🤖</span> Select Models
        </h2>
        
        <div className="grid grid-cols-2 gap-6">
          {/* Model A */}
          <div className="space-y-3">
            <label className="block text-sm font-medium text-gray-400">
              Model A (White first)
            </label>
            <select
              value={modelA}
              onChange={(e) => setModelA(e.target.value)}
              className="w-full bg-arena-bg border border-arena-border rounded-lg px-4 py-3 
                         focus:outline-none focus:ring-2 focus:ring-arena-accent focus:border-transparent
                         text-white appearance-none cursor-pointer"
            >
              <option value="">Select model...</option>
              {Object.entries(modelsByProvider).map(([provider, providerModels]) => (
                <optgroup key={provider} label={provider}>
                  {providerModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
            
            {/* Model A Reasoning Options */}
            {modelA && (
              <div className="space-y-2 p-3 bg-arena-bg/50 rounded-lg border border-arena-border/50">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-gray-400">🧠 Reasoning Budget</span>
                  <span className="text-xs text-arena-accent font-mono">{reasoningBudgetA.toLocaleString()}</span>
                </div>
                <input
                  type="range"
                  value={reasoningBudgetA}
                  onChange={(e) => setReasoningBudgetA(parseInt(e.target.value))}
                  min={1000}
                  max={48000}
                  step={1000}
                  className="w-full accent-arena-accent h-1"
                />
                <div className="flex justify-between text-[10px] text-gray-600">
                  <span>1K</span>
                  <span>24K</span>
                  <span>48K</span>
                </div>
                <label className="flex items-center gap-2 mt-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showReasoningA}
                    onChange={(e) => setShowReasoningA(e.target.checked)}
                    className="w-3 h-3 rounded border-arena-border bg-arena-bg text-arena-accent focus:ring-arena-accent"
                  />
                  <span className="text-xs text-gray-500">Show reasoning traces</span>
                </label>
              </div>
            )}
          </div>

          {/* Model B */}
          <div className="space-y-3">
            <label className="block text-sm font-medium text-gray-400">
              Model B (Black first)
            </label>
            <select
              value={modelB}
              onChange={(e) => setModelB(e.target.value)}
              className="w-full bg-arena-bg border border-arena-border rounded-lg px-4 py-3 
                         focus:outline-none focus:ring-2 focus:ring-arena-accent focus:border-transparent
                         text-white appearance-none cursor-pointer"
            >
              <option value="">Select model...</option>
              {Object.entries(modelsByProvider).map(([provider, providerModels]) => (
                <optgroup key={provider} label={provider}>
                  {providerModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
            
            {/* Model B Reasoning Options */}
            {modelB && (
              <div className="space-y-2 p-3 bg-arena-bg/50 rounded-lg border border-arena-border/50">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-gray-400">🧠 Reasoning Budget</span>
                  <span className="text-xs text-purple-400 font-mono">{reasoningBudgetB.toLocaleString()}</span>
                </div>
                <input
                  type="range"
                  value={reasoningBudgetB}
                  onChange={(e) => setReasoningBudgetB(parseInt(e.target.value))}
                  min={1000}
                  max={48000}
                  step={1000}
                  className="w-full accent-purple-500 h-1"
                />
                <div className="flex justify-between text-[10px] text-gray-600">
                  <span>1K</span>
                  <span>24K</span>
                  <span>48K</span>
                </div>
                <label className="flex items-center gap-2 mt-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showReasoningB}
                    onChange={(e) => setShowReasoningB(e.target.checked)}
                    className="w-3 h-3 rounded border-arena-border bg-arena-bg text-purple-500 focus:ring-purple-500"
                  />
                  <span className="text-xs text-gray-500">Show reasoning traces</span>
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Quick comparison preview */}
        {modelA && modelB && (
          <div className="bg-arena-bg rounded-lg p-4 flex items-center justify-center gap-8">
            <div className="text-center">
              <div className="text-lg font-semibold text-arena-accent">
                {models.find(m => m.id === modelA)?.name}
              </div>
              <div className="text-xs text-gray-500">
                {models.find(m => m.id === modelA)?.provider}
              </div>
              <div className="text-xs text-arena-accent/70 mt-1">
                {reasoningBudgetA.toLocaleString()} tokens
              </div>
            </div>
            <div className="text-2xl font-bold text-gray-500">⚔️</div>
            <div className="text-center">
              <div className="text-lg font-semibold text-purple-400">
                {models.find(m => m.id === modelB)?.name}
                {modelA === modelB && <span className="text-gray-500 ml-1">(mirror)</span>}
              </div>
              <div className="text-xs text-gray-500">
                {models.find(m => m.id === modelB)?.provider}
              </div>
              <div className="text-xs text-purple-400/70 mt-1">
                {reasoningBudgetB.toLocaleString()} tokens
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Time Control */}
      <div className="card p-6 space-y-6">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <span className="text-2xl">⏱️</span> Time Control
        </h2>

        <div className="grid grid-cols-3 gap-3">
          {timePresets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handleTimePresetChange(preset.id)}
              className={`p-4 rounded-lg border transition-all ${
                timePreset === preset.id && !customTime
                  ? "border-arena-accent bg-arena-accent/10 text-white"
                  : "border-arena-border hover:border-gray-600 text-gray-400 hover:text-white"
              }`}
            >
              <div className="font-semibold">{preset.name}</div>
              <div className="text-xs opacity-70">
                {Math.floor(preset.initial_time / 60)}:{String(preset.initial_time % 60).padStart(2, "0")} + {preset.increment}s
              </div>
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="customTime"
            checked={customTime}
            onChange={(e) => setCustomTime(e.target.checked)}
            className="w-4 h-4 rounded border-arena-border bg-arena-bg text-arena-accent focus:ring-arena-accent"
          />
          <label htmlFor="customTime" className="text-sm text-gray-400 cursor-pointer">
            Custom time control
          </label>
        </div>

        {customTime && (
          <div className="grid grid-cols-2 gap-4 pt-2">
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-400">
                Initial Time (seconds)
              </label>
              <input
                type="number"
                value={initialTime}
                onChange={(e) => setInitialTime(parseInt(e.target.value) || 0)}
                min={10}
                max={3600}
                className="w-full bg-arena-bg border border-arena-border rounded-lg px-4 py-3 
                           focus:outline-none focus:ring-2 focus:ring-arena-accent text-white"
              />
            </div>
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-400">
                Increment (seconds)
              </label>
              <input
                type="number"
                value={increment}
                onChange={(e) => setIncrement(parseInt(e.target.value) || 0)}
                min={0}
                max={60}
                className="w-full bg-arena-bg border border-arena-border rounded-lg px-4 py-3 
                           focus:outline-none focus:ring-2 focus:ring-arena-accent text-white"
              />
            </div>
          </div>
        )}
      </div>

      {/* Match Format */}
      <div className="card p-6 space-y-6">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <span className="text-2xl">🏆</span> Match Format
        </h2>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-400">
            First to win
          </label>
          <div className="flex gap-3">
            {[1, 2, 3, 4, 5].map((n) => (
              <button
                key={n}
                onClick={() => setFirstTo(n)}
                className={`flex-1 py-3 rounded-lg border transition-all ${
                  firstTo === n
                    ? "border-arena-accent bg-arena-accent/10 text-white font-semibold"
                    : "border-arena-border hover:border-gray-600 text-gray-400"
                }`}
              >
                {n} {n === 1 ? "game" : "games"}
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-500">
            Best of {firstTo * 2 - 1} games (up to {firstTo * 2 - 1} games may be played)
          </p>
        </div>
      </div>

      {/* Advanced Options */}
      <div className="card p-6 space-y-4">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
        >
          <span className="text-xl">{showAdvanced ? "▼" : "▶"}</span>
          <h2 className="text-xl font-semibold">Advanced Options</h2>
        </button>

        {showAdvanced && (
          <div className="space-y-6 pt-4 border-t border-arena-border">
            {/* Rethinking */}
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Enable Rethinking</div>
                <div className="text-sm text-gray-500">Allow models to retry on parsing failures</div>
              </div>
              <button
                onClick={() => setUseRethinking(!useRethinking)}
                className={`w-14 h-8 rounded-full transition-colors relative ${
                  useRethinking ? "bg-arena-accent" : "bg-arena-border"
                }`}
              >
                <div
                  className={`absolute top-1 w-6 h-6 rounded-full bg-white transition-transform ${
                    useRethinking ? "translate-x-7" : "translate-x-1"
                  }`}
                />
              </button>
            </div>

            {useRethinking && (
              <div className="grid grid-cols-2 gap-4 pl-4 border-l-2 border-arena-border">
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-400">
                    Max Rethinks
                  </label>
                  <input
                    type="number"
                    value={maxRethinks}
                    onChange={(e) => setMaxRethinks(parseInt(e.target.value) || 0)}
                    min={1}
                    max={5}
                    className="w-full bg-arena-bg border border-arena-border rounded-lg px-4 py-2 
                               focus:outline-none focus:ring-2 focus:ring-arena-accent text-white"
                  />
                </div>
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-400">
                    Max Parsing Failures
                  </label>
                  <input
                    type="number"
                    value={maxParsingFailures}
                    onChange={(e) => setMaxParsingFailures(parseInt(e.target.value) || 0)}
                    min={1}
                    max={10}
                    className="w-full bg-arena-bg border border-arena-border rounded-lg px-4 py-2 
                               focus:outline-none focus:ring-2 focus:ring-arena-accent text-white"
                  />
                </div>
              </div>
            )}

            {/* Note about reasoning */}
            <div className="text-sm text-gray-500 italic">
              💡 Reasoning budgets are configured per-model above in the model selection section.
            </div>
          </div>
        )}
      </div>

      {/* Process Status Panel */}
      {processDetail && (
        <div className="card p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <span className="text-2xl">📊</span> Match Process
            </h2>
            <div className="flex items-center gap-3">
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                processDetail.status === "running" 
                  ? "bg-blue-500/20 text-blue-400" 
                  : processDetail.status === "completed"
                  ? "bg-green-500/20 text-green-400"
                  : processDetail.status === "failed"
                  ? "bg-red-500/20 text-red-400"
                  : "bg-gray-500/20 text-gray-400"
              }`}>
                {processDetail.status === "running" && (
                  <span className="inline-block w-2 h-2 bg-blue-400 rounded-full animate-pulse mr-2" />
                )}
                {processDetail.status.charAt(0).toUpperCase() + processDetail.status.slice(1)}
              </span>
              <span className="text-sm text-gray-500">
                PID: {processDetail.pid}
              </span>
            </div>
          </div>

          {/* Match Info */}
          <div className="flex items-center justify-center gap-6 py-3 bg-arena-bg rounded-lg">
            <span className="font-medium">{processDetail.model_a}</span>
            <span className="text-gray-500">vs</span>
            <span className="font-medium">{processDetail.model_b}</span>
            <span className="text-sm text-gray-500">
              ({Math.floor(processDetail.running_seconds)}s)
            </span>
          </div>

          {/* Error Display */}
          {processDetail.error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
              {processDetail.error}
            </div>
          )}

          {/* Logs */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-400">Output Logs</span>
              <span className="text-xs text-gray-500">{processDetail.logs.length} lines</span>
            </div>
            <div className="bg-black rounded-lg p-4 max-h-96 overflow-y-auto font-mono text-xs">
              {processDetail.logs.length === 0 ? (
                <div className="text-gray-500 italic">Waiting for output...</div>
              ) : (
                processDetail.logs.map((log, i) => {
                  // Strip ANSI color codes
                  const cleanLine = log.line.replace(/\u001b\[[0-9;]*m/g, "");
                  
                  // Determine line color based on content
                  const lineColor = 
                    cleanLine.includes("Error") || cleanLine.includes("❌") || cleanLine.includes("failed")
                      ? "text-red-400"
                      : cleanLine.includes("✅") || cleanLine.includes("🎉") || cleanLine.includes("WINNER")
                      ? "text-green-400"
                      : cleanLine.includes("⏰") || cleanLine.includes("Thinking time")
                      ? "text-blue-400"
                      : cleanLine.includes("===") || cleanLine.includes("BLITZ")
                      ? "text-cyan-400 font-bold"
                      : cleanLine.includes("Move ") && cleanLine.includes("turn")
                      ? "text-yellow-400"
                      : cleanLine.includes("Final move:")
                      ? "text-green-300"
                      : cleanLine.startsWith("I1225") || cleanLine.includes("HTTP Request")
                      ? "text-gray-600"  // Dim the HTTP logs
                      : "text-gray-300";
                  
                  return (
                    <div key={i} className="py-0.5 leading-relaxed">
                      <span className="text-gray-700 mr-2 select-none">
                        {new Date(log.time).toLocaleTimeString()}
                      </span>
                      <span className={lineColor}>
                        {cleanLine}
                      </span>
                    </div>
                  );
                })
              )}
              <div ref={logsEndRef} />
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            {processDetail.status === "running" ? (
              <button
                onClick={handleStopMatch}
                className="flex-1 py-2 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
              >
                Stop Match
              </button>
            ) : (
              <>
                <button
                  onClick={resetForm}
                  className="flex-1 py-2 rounded-lg bg-arena-border text-gray-300 hover:bg-gray-700 transition-colors"
                >
                  Start New Match
                </button>
                {processDetail.status === "completed" && (
                  <button
                    onClick={() => router.push("/matches")}
                    className="flex-1 py-2 rounded-lg bg-arena-accent text-white hover:opacity-90 transition-colors"
                  >
                    View Results →
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* Start Button */}
      {!processDetail && (
        <button
          onClick={handleStartMatch}
          disabled={starting || !modelA || !modelB}
          className={`w-full py-4 rounded-xl text-lg font-bold transition-all ${
            starting || !modelA || !modelB
              ? "bg-gray-700 text-gray-500 cursor-not-allowed"
              : "bg-gradient-to-r from-arena-accent to-purple-500 text-white hover:opacity-90 hover:scale-[1.02] active:scale-[0.98]"
          }`}
        >
          {starting ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Starting Match...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <span>⚔️</span> Start Match
            </span>
          )}
        </button>
      )}
    </div>
  );
}

