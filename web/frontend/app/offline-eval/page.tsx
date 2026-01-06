"use client";

import { useEffect, useState } from "react";
import { api, OfflineEvalSummary, OfflineTimeoutAnalysis, OfflineAblationComparison, OfflineResponseTimeData, OfflineMoveQualityAnalysis } from "@/lib/api";
import { TimeoutBar, ResponseTimeBar, AblationCompareCard, MoveQualityBar, BlunderRateBar, QualityByStyleCard } from "@/components/charts/OfflineEvalCharts";

export default function OfflineEvalPage() {
  const [summary, setSummary] = useState<OfflineEvalSummary | null>(null);
  const [timeouts, setTimeouts] = useState<OfflineTimeoutAnalysis | null>(null);
  const [ablation, setAblation] = useState<OfflineAblationComparison | null>(null);
  const [responseTimes, setResponseTimes] = useState<OfflineResponseTimeData[]>([]);
  const [moveQuality, setMoveQuality] = useState<OfflineMoveQualityAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [summaryData, timeoutData, ablationData, responseData, qualityData] = await Promise.all([
          api.getOfflineEvalSummary(),
          api.getOfflineEvalTimeouts(),
          api.getOfflineEvalAblation(),
          api.getOfflineEvalResponseTimes(),
          api.getOfflineEvalMoveQuality(),
        ]);
        setSummary(summaryData);
        setTimeouts(timeoutData);
        setAblation(ablationData);
        setResponseTimes(responseData.data);
        setMoveQuality(qualityData);
        
        // Auto-select first model
        if (summaryData.models.length > 0) {
          setSelectedModel(summaryData.models[0]);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load data");
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-950 text-zinc-100 flex items-center justify-center">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-xl">Loading experiments...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-zinc-950 text-zinc-100 flex items-center justify-center">
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-6 max-w-md">
          <div className="text-red-400 font-medium mb-2">Failed to load data</div>
          <div className="text-zinc-400 text-sm">{error}</div>
        </div>
      </div>
    );
  }

  if (!summary || summary.total_evaluations === 0) {
    return (
      <div className="min-h-screen bg-zinc-950 text-zinc-100 p-8">
        <h1 className="text-3xl font-bold mb-4">Offline Experiments</h1>
        <div className="bg-zinc-900 rounded-xl p-12 text-center max-w-2xl mx-auto">
          <div className="text-6xl mb-4">🧪</div>
          <p className="text-zinc-400 mb-6">No offline evaluation results found.</p>
          <div className="bg-zinc-800 rounded-lg p-4 text-left">
            <p className="text-zinc-500 text-sm mb-2">Run an evaluation:</p>
            <code className="text-cyan-400 text-sm">
              python scripts/run_offline_eval.py --model gemini-3-flash --ablation
            </code>
          </div>
        </div>
      </div>
    );
  }

  // Group data by model
  const modelTimeouts = selectedModel 
    ? timeouts?.by_model_time.filter(t => t.model_id === selectedModel) ?? []
    : [];
  const modelResponseTimes = selectedModel
    ? responseTimes.filter(t => t.model_id === selectedModel)
    : [];

  // Calculate max response time for scaling
  const maxResponseTime = Math.max(
    ...responseTimes.map(r => r.avg_response_time ?? 0),
    60
  );

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <div className="border-b border-zinc-800 bg-gradient-to-r from-zinc-900 to-zinc-950">
        <div className="container mx-auto px-6 py-8">
          <h1 className="text-3xl font-bold mb-2">Offline Experiments</h1>
          <p className="text-zinc-400">Controlled time pressure analysis on fixed chess positions</p>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <SummaryCard 
            icon="📊" 
            label="Evaluations" 
            value={summary.total_evaluations.toLocaleString()} 
          />
          <SummaryCard 
            icon="🤖" 
            label="Models" 
            value={summary.models.length} 
          />
          <SummaryCard 
            icon="⏱️" 
            label="Time Levels" 
            value={summary.time_levels.length} 
          />
          <SummaryCard 
            icon="⏰" 
            label="Timeout Rate" 
            value={`${(summary.overall_timeout_rate * 100).toFixed(1)}%`}
            color={summary.overall_timeout_rate > 0.3 ? "red" : summary.overall_timeout_rate > 0.1 ? "amber" : "emerald"}
          />
        </div>

        {/* Ablation Comparison - Hero Section */}
        {ablation?.available && ablation.models && Object.keys(ablation.models).length > 0 && (
          <section className="mb-10">
            <div className="flex items-center gap-3 mb-6">
              <span className="text-2xl">🧪</span>
              <h2 className="text-2xl font-bold">Prompt Style Ablation</h2>
            </div>
            <p className="text-zinc-400 mb-6 max-w-2xl">
              Does time pressure guidance actually help? Comparing <span className="text-cyan-400">time info only</span> (just the clock) 
              vs <span className="text-purple-400">standard</span> prompts (with urgency guidance).
            </p>
            
            <div className="grid md:grid-cols-2 gap-6">
              {Object.entries(ablation.models).map(([modelId, styles]) => {
                const styleNames = Object.keys(styles);
                if (styleNames.length < 2) return null;
                
                const [styleAName, styleBName] = styleNames;
                const styleAData = styles[styleAName];
                const styleBData = styles[styleBName];
                
                return (
                  <AblationCompareCard
                    key={modelId}
                    model={modelId}
                    styleA={{
                      name: styleAName,
                      timeoutRate: styleAData.timeout_rate,
                      avgTime: styleAData.avg_response_time,
                      avgTokens: styleAData.avg_thinking_tokens,
                      avgCpLoss: styleAData.avg_centipawn_loss,
                    }}
                    styleB={{
                      name: styleBName,
                      timeoutRate: styleBData.timeout_rate,
                      avgTime: styleBData.avg_response_time,
                      avgTokens: styleBData.avg_thinking_tokens,
                      avgCpLoss: styleBData.avg_centipawn_loss,
                    }}
                  />
                );
              })}
            </div>
          </section>
        )}

        {/* Model Selector */}
        {summary.models.length > 1 && (
          <div className="flex gap-2 mb-6">
            {summary.models.map((model) => (
              <button
                key={model}
                onClick={() => setSelectedModel(model)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  selectedModel === model
                    ? "bg-cyan-600 text-white shadow-lg shadow-cyan-600/20"
                    : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white"
                }`}
              >
                {model}
              </button>
            ))}
          </div>
        )}

        {/* Charts Grid */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Timeout Analysis */}
          {modelTimeouts.length > 0 && (
            <section className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
              <div className="flex items-center gap-2 mb-6">
                <span className="text-xl">⏰</span>
                <h2 className="text-lg font-semibold">Timeout Rate by Time Level</h2>
              </div>
              <p className="text-zinc-500 text-sm mb-4">
                How often does the model run out of time? Hover for details.
              </p>
              
              <div className="space-y-1">
                {modelTimeouts
                  .sort((a, b) => b.time_remaining - a.time_remaining)
                  .map((item) => (
                    <TimeoutBar
                      key={`${item.model_id}-${item.time_remaining}`}
                      model={item.model_id}
                      timeLevel={item.time_remaining}
                      rate={item.rate}
                      timeouts={item.timeouts}
                      total={item.total}
                    />
                  ))}
              </div>
            </section>
          )}

          {/* Response Time Analysis */}
          {modelResponseTimes.length > 0 && (
            <section className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
              <div className="flex items-center gap-2 mb-6">
                <span className="text-xl">📊</span>
                <h2 className="text-lg font-semibold">Response Time vs Time Available</h2>
              </div>
              <p className="text-zinc-500 text-sm mb-4">
                Does the model adjust its thinking based on time pressure? The vertical line shows the time limit.
              </p>
              
              <div className="space-y-1">
                {modelResponseTimes
                  .sort((a, b) => b.time_remaining - a.time_remaining)
                  .map((item) => (
                    <ResponseTimeBar
                      key={`${item.model_id}-${item.time_remaining}`}
                      model={item.model_id}
                      timeLevel={item.time_remaining}
                      avgTime={item.avg_response_time ?? 0}
                      stdDev={item.std_response_time}
                      avgTokens={item.avg_thinking_tokens}
                      maxTime={maxResponseTime * 1.2}
                    />
                  ))}
              </div>
            </section>
          )}
        </div>

        {/* Move Quality Section */}
        {summary.has_move_quality && moveQuality?.available ? (
          <>
            {/* Quality by Prompt Style */}
            {moveQuality.by_prompt_style.length > 0 && (
              <section className="mt-10 mb-8">
                <div className="flex items-center gap-3 mb-6">
                  <span className="text-2xl">♟️</span>
                  <h2 className="text-2xl font-bold">Move Quality by Prompt Style</h2>
                </div>
                <p className="text-zinc-400 mb-6 max-w-2xl">
                  Does the prompt style affect the quality of chess moves? Lower centipawn loss means better moves.
                </p>
                
                <div className="grid md:grid-cols-2 gap-6">
                  {summary.models.map((model) => {
                    const modelStyles = moveQuality.by_prompt_style.filter(s => s.model_id === model);
                    if (modelStyles.length < 2) return null;
                    
                    return (
                      <QualityByStyleCard
                        key={model}
                        model={model}
                        styles={modelStyles.map(s => ({
                          name: s.prompt_style,
                          avgCpLoss: s.avg_centipawn_loss,
                          blunderRate: s.blunder_rate,
                        }))}
                      />
                    );
                  })}
                </div>
              </section>
            )}

            {/* Move Quality Charts Grid */}
            <div className="grid lg:grid-cols-2 gap-8 mt-8">
              {/* Move Quality by Time Level */}
              {selectedModel && moveQuality.by_model_time.filter(m => m.model_id === selectedModel).length > 0 && (
                <section className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
                  <div className="flex items-center gap-2 mb-6">
                    <span className="text-xl">♟️</span>
                    <h2 className="text-lg font-semibold">Move Quality by Time Level</h2>
                  </div>
                  <p className="text-zinc-500 text-sm mb-4">
                    Does move quality degrade under time pressure? Lower centipawn loss = better moves.
                  </p>
                  
                  <div className="space-y-1">
                    {moveQuality.by_model_time
                      .filter(m => m.model_id === selectedModel)
                      .sort((a, b) => b.time_remaining - a.time_remaining)
                      .map((item) => (
                        <MoveQualityBar
                          key={`${item.model_id}-${item.time_remaining}`}
                          model={item.model_id}
                          timeLevel={item.time_remaining}
                          avgCpLoss={item.avg_centipawn_loss}
                          blunderRate={item.blunder_rate}
                          bestMoveRate={item.best_move_rate}
                        />
                      ))}
                  </div>
                </section>
              )}

              {/* Blunder Rate by Time Level */}
              {selectedModel && moveQuality.by_model_time.filter(m => m.model_id === selectedModel && m.blunder_rate !== null).length > 0 && (
                <section className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
                  <div className="flex items-center gap-2 mb-6">
                    <span className="text-xl">💥</span>
                    <h2 className="text-lg font-semibold">Blunder Rate by Time Level</h2>
                  </div>
                  <p className="text-zinc-500 text-sm mb-4">
                    When do models make catastrophic mistakes? A blunder is a move losing 100+ centipawns.
                  </p>
                  
                  <div className="space-y-1">
                    {moveQuality.by_model_time
                      .filter(m => m.model_id === selectedModel && m.blunder_rate !== null)
                      .sort((a, b) => b.time_remaining - a.time_remaining)
                      .map((item) => (
                        <BlunderRateBar
                          key={`blunder-${item.model_id}-${item.time_remaining}`}
                          model={item.model_id}
                          timeLevel={item.time_remaining}
                          blunderRate={item.blunder_rate!}
                        />
                      ))}
                  </div>
                </section>
              )}
            </div>
          </>
        ) : (
          <div className="mt-8 bg-zinc-900 border border-zinc-800 rounded-lg p-4 flex items-center gap-3">
            <span className="text-2xl opacity-50">♟️</span>
            <div>
              <div className="text-zinc-400">Move quality analysis not yet computed</div>
              <div className="text-zinc-500 text-sm">
                Run: <code className="text-cyan-400">python scripts/run_offline_eval.py --stockfish</code>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function SummaryCard({ 
  icon,
  label, 
  value, 
  color = "default" 
}: { 
  icon: string;
  label: string; 
  value: string | number; 
  color?: "default" | "red" | "amber" | "emerald";
}) {
  const colorClasses = {
    default: "text-white",
    red: "text-red-400",
    amber: "text-amber-400", 
    emerald: "text-emerald-400",
  };

  return (
    <div className="bg-zinc-900 rounded-xl p-5 border border-zinc-800 hover:border-zinc-700 transition-colors">
      <div className="flex items-center gap-2 text-zinc-400 text-sm mb-2">
        <span>{icon}</span>
        <span>{label}</span>
      </div>
      <div className={`text-2xl font-bold ${colorClasses[color]}`}>{value}</div>
    </div>
  );
}
