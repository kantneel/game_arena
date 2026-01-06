"use client";

import { useEffect, useState, useMemo } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

// === TYPE DEFINITIONS ===

interface DegradationPoint {
  time_bucket_start: number;
  time_bucket_end: number;
  bucket_label: string;
  move_count: number;
  avg_cp_loss: number;
  median_cp_loss: number;
  blunder_rate: number;
  p90_cp_loss: number;
}

interface ModelDegradationCurve {
  model_name: string;
  total_moves: number;
  curve: DegradationPoint[];
  degradation_ratio: number;
  critical_threshold: number | null;
}

interface EfficiencyMetrics {
  model_name: string;
  total_moves: number;
  avg_time_per_move: number;
  avg_cp_loss: number;
  avg_tokens_per_move: number | null;
  quality_per_second: number;
  quality_per_token: number | null;
  efficiency_by_time_spent: Array<{
    time_range: string;
    move_count: number;
    avg_quality: number;
    avg_time: number;
    efficiency: number;
  }>;
  optimal_time_range: string | null;
}

interface TimeAllocationStats {
  model_name: string;
  opening_avg_time: number;
  middlegame_avg_time: number;
  endgame_avg_time: number;
  opening_avg_quality: number;
  middlegame_avg_quality: number;
  endgame_avg_quality: number;
  complexity_time_correlation: number;
  complexity_quality_correlation: number;
}

interface PositionPerformance {
  model_name: string;
  simple_pos_quality: number;
  complex_pos_quality: number;
  complexity_penalty: number;
  routine_pos_quality: number;
  critical_pos_quality: number;
  criticality_penalty: number;
  critical_time_ratio: number;
}

interface OutcomeCorrelate {
  factor: string;
  correlation_with_win: number;
  sample_size: number;
  description: string;
}

interface MatchupSummary {
  model_a: string;
  model_b: string;
  is_same_model: boolean;
  total_games: number;
  model_a_wins: number;
  model_b_wins: number;
  draws: number;
  model_a_win_rate: number;
  avg_game_length: number;
  notes: string[];
}

interface TournamentSummary {
  total_unique_matchups: number;
  same_model_matchups: number;
  cross_model_matchups: number;
  matchups: MatchupSummary[];
  model_rankings: Array<{
    model: string;
    wins: number;
    losses: number;
    games: number;
    win_rate: number;
  }>;
}

interface InsightsData {
  total_matches: number;
  total_games: number;
  total_moves: number;
  total_analyzed_moves: number;
  outliers_capped: number;
  cp_loss_cap: number;
  same_model_games_excluded: number;
  tournament: TournamentSummary | null;
  degradation_curves: ModelDegradationCurve[];
  efficiency_metrics: EfficiencyMetrics[];
  time_allocation: TimeAllocationStats[];
  position_performance: PositionPerformance[];
  outcome_correlates: OutcomeCorrelate[];
  scatter_data: Array<{
    model: string;
    time_remaining: number;
    time_taken: number;
    cp_loss: number;
    num_legal_moves: number | null;
  }>;
}

// === HELPER FUNCTIONS ===

const MODEL_COLORS: Record<string, string> = {
  "gemini-3-pro": "#22c55e",
  "gemini-3-flash": "#3b82f6",
  "gemini-2.5-pro": "#f97316",
  "gemini-2.5-flash": "#a855f7",
};

function getModelColor(model: string): string {
  return MODEL_COLORS[model] || "#6b7280";
}

function getCpColor(cp: number): string {
  if (cp < 10) return "#22c55e";
  if (cp < 25) return "#84cc16";
  if (cp < 50) return "#eab308";
  if (cp < 100) return "#f97316";
  return "#ef4444";
}

function getCorrelationColor(corr: number): string {
  const abs = Math.abs(corr);
  if (abs < 0.1) return "#6b7280";
  if (abs < 0.3) return "#fbbf24";
  if (abs < 0.5) return "#f97316";
  return "#ef4444";
}

function formatCorrelation(corr: number): string {
  const sign = corr >= 0 ? "+" : "";
  return `${sign}${corr.toFixed(3)}`;
}

// === MAIN COMPONENT ===

export default function InsightsPage() {
  const [data, setData] = useState<InsightsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("all");

  useEffect(() => {
    const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
    const url = selectedModel === "all" 
      ? `${API_BASE}/insights`
      : `${API_BASE}/insights?model_filter=${encodeURIComponent(selectedModel)}`;
    
    setLoading(true);
    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        setData(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [selectedModel]);

  const availableModels = useMemo(() => {
    if (!data) return [];
    return data.degradation_curves.map((d) => d.model_name);
  }, [data]);

  // Prepare degradation chart data
  const degradationChartData = useMemo(() => {
    if (!data) return [];
    
    // Get all unique time buckets
    const buckets = new Set<string>();
    data.degradation_curves.forEach((curve) => {
      curve.curve.forEach((point) => buckets.add(point.bucket_label));
    });
    
    // Create data points for each bucket
    const bucketOrder = [
      "5min+ (Abundant)",
      "3-5min (High)",
      "2-3min (Comfortable)",
      "1-2min (Medium)",
      "30-60s (Low)",
      "0-30s (Critical)",
    ];
    
    return bucketOrder.map((label) => {
      const point: Record<string, any> = { bucket: label };
      data.degradation_curves.forEach((curve) => {
        const match = curve.curve.find((p) => p.bucket_label === label);
        if (match) {
          point[curve.model_name] = match.median_cp_loss;
          point[`${curve.model_name}_blunder`] = match.blunder_rate * 100;
        }
      });
      return point;
    }).filter((p) => Object.keys(p).length > 1);
  }, [data]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-xl text-gray-400">Loading insights...</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-xl text-red-400">Error: {error || "No data"}</div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">⏱️ Time-Constrained Reasoning Analysis</h1>
        <p className="text-gray-400">
          How do LLM agents perform under time pressure? Analysis of {data.total_games} games, {data.total_analyzed_moves.toLocaleString()} moves.
        </p>
        {data.outliers_capped > 0 && (
          <p className="text-gray-500 text-sm mt-1">
            📊 {data.outliers_capped} extreme values capped at {data.cp_loss_cap} CP for cleaner analysis.
          </p>
        )}
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="card p-4 text-center">
          <div className="text-3xl font-bold text-cyan-400">{data.total_matches}</div>
          <div className="text-gray-400 text-sm">Matches</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-3xl font-bold text-purple-400">{data.total_games}</div>
          <div className="text-gray-400 text-sm">Games</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-3xl font-bold text-green-400">{data.total_analyzed_moves.toLocaleString()}</div>
          <div className="text-gray-400 text-sm">Analyzed Moves</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-3xl font-bold text-orange-400">{data.degradation_curves.length}</div>
          <div className="text-gray-400 text-sm">Models</div>
        </div>
      </div>

      {/* 0. TOURNAMENT OVERVIEW */}
      {data.tournament && (
        <div className="card p-6 mb-8 border-2 border-purple-500/30">
          <h2 className="text-xl font-bold mb-2">🏆 Tournament Overview</h2>
          <p className="text-gray-400 text-sm mb-4">
            {data.tournament.cross_model_matchups} cross-model matchups, {data.tournament.same_model_matchups} same-model matchups
            {data.same_model_games_excluded > 0 && (
              <span className="text-yellow-400 ml-2">
                ({data.same_model_games_excluded} same-model games noted but analyzed separately)
              </span>
            )}
          </p>
          
          {/* Model Rankings */}
          {data.tournament.model_rankings.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-medium mb-3">🥇 Model Rankings (Cross-Model Only)</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {data.tournament.model_rankings.map((r, i) => (
                  <div key={r.model} className={`rounded-lg p-4 ${
                    i === 0 ? "bg-yellow-500/10 border border-yellow-500/30" :
                    i === 1 ? "bg-gray-400/10 border border-gray-400/30" :
                    i === 2 ? "bg-orange-600/10 border border-orange-600/30" :
                    "bg-gray-800/50"
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-2xl">
                        {i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : `#${i + 1}`}
                      </span>
                      <span className="font-medium" style={{ color: getModelColor(r.model) }}>
                        {r.model}
                      </span>
                    </div>
                    <div className="text-2xl font-bold mb-1">
                      {(r.win_rate * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-400">
                      {r.wins}W - {r.losses}L ({r.games} games)
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Matchup Matrix */}
          <div>
            <h3 className="text-lg font-medium mb-3">📊 Matchup Results</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-2 px-3">Matchup</th>
                    <th className="text-center py-2 px-3">Games</th>
                    <th className="text-center py-2 px-3">Score</th>
                    <th className="text-center py-2 px-3">Win Rate</th>
                    <th className="text-center py-2 px-3">Avg Length</th>
                    <th className="text-left py-2 px-3">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {data.tournament.matchups.filter(m => !m.is_same_model).map((m, i) => (
                    <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/50">
                      <td className="py-2 px-3">
                        <span style={{ color: getModelColor(m.model_a) }}>{m.model_a}</span>
                        <span className="text-gray-500 mx-2">vs</span>
                        <span style={{ color: getModelColor(m.model_b) }}>{m.model_b}</span>
                      </td>
                      <td className="text-center py-2 px-3">{m.total_games}</td>
                      <td className="text-center py-2 px-3">
                        <span style={{ color: getModelColor(m.model_a) }}>{m.model_a_wins}</span>
                        <span className="text-gray-500 mx-1">-</span>
                        <span style={{ color: getModelColor(m.model_b) }}>{m.model_b_wins}</span>
                        {m.draws > 0 && <span className="text-gray-500 ml-1">({m.draws}D)</span>}
                      </td>
                      <td className="text-center py-2 px-3">
                        <div className="flex items-center justify-center gap-2">
                          <div className="w-20 h-2 bg-gray-700 rounded-full overflow-hidden">
                            <div 
                              className="h-full"
                              style={{ 
                                width: `${m.model_a_win_rate * 100}%`,
                                backgroundColor: getModelColor(m.model_a)
                              }}
                            />
                          </div>
                          <span className="text-xs">{(m.model_a_win_rate * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="text-center py-2 px-3 text-gray-400">{m.avg_game_length.toFixed(0)} moves</td>
                      <td className="py-2 px-3 text-xs text-gray-500 max-w-xs truncate">
                        {m.notes.length > 0 ? m.notes[0] : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {/* Same-model matchups note */}
            {data.tournament.matchups.filter(m => m.is_same_model).length > 0 && (
              <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-sm">
                <span className="text-yellow-400">⚠️ Same-model matchups detected:</span>
                <span className="text-gray-400 ml-2">
                  {data.tournament.matchups.filter(m => m.is_same_model).map(m => 
                    `${m.model_a} vs itself (${m.total_games} games)`
                  ).join(", ")}
                </span>
                <p className="text-xs text-gray-500 mt-1">
                  These are excluded from rankings. Use round-robin with distinct model pairs for meaningful results.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 1. DEGRADATION CURVES */}
      <div className="card p-6 mb-8">
        <h2 className="text-xl font-bold mb-2">📉 Degradation Curves</h2>
        <p className="text-gray-400 text-sm mb-4">
          How does move quality (median CP loss) change as remaining time decreases?
        </p>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Quality Degradation Chart */}
          <div>
            <h3 className="text-sm font-medium text-gray-400 mb-3">Median CP Loss by Time Remaining</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={degradationChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
                <XAxis 
                  dataKey="bucket" 
                  tick={{ fill: "#9ca3af", fontSize: 10 }} 
                  angle={-20}
                  textAnchor="end"
                  height={60}
                />
                <YAxis tick={{ fill: "#9ca3af", fontSize: 12 }} domain={[0, 'auto']} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#12121a", border: "1px solid #2e2e3e" }}
                  formatter={(value: number) => value.toFixed(1)}
                />
                <Legend />
                {data.degradation_curves.map((curve) => (
                  <Line
                    key={curve.model_name}
                    type="monotone"
                    dataKey={curve.model_name}
                    stroke={getModelColor(curve.model_name)}
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    name={curve.model_name}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Blunder Rate Chart */}
          <div>
            <h3 className="text-sm font-medium text-gray-400 mb-3">Blunder Rate (%) by Time Remaining</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={degradationChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
                <XAxis 
                  dataKey="bucket" 
                  tick={{ fill: "#9ca3af", fontSize: 10 }} 
                  angle={-20}
                  textAnchor="end"
                  height={60}
                />
                <YAxis tick={{ fill: "#9ca3af", fontSize: 12 }} tickFormatter={(v) => `${v}%`} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#12121a", border: "1px solid #2e2e3e" }}
                  formatter={(value: number) => `${value.toFixed(1)}%`}
                />
                <Legend />
                <ReferenceLine y={15} stroke="#ef4444" strokeDasharray="3 3" label={{ value: "Critical", fill: "#ef4444", fontSize: 10 }} />
                {data.degradation_curves.map((curve) => (
                  <Line
                    key={curve.model_name}
                    type="monotone"
                    dataKey={`${curve.model_name}_blunder`}
                    stroke={getModelColor(curve.model_name)}
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    name={curve.model_name}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Degradation Summary Table */}
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Degradation Summary</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-3">Model</th>
                  <th className="text-right py-2 px-3">Total Moves</th>
                  <th className="text-right py-2 px-3">Degradation Ratio</th>
                  <th className="text-right py-2 px-3">Critical Threshold</th>
                  <th className="text-left py-2 px-3">Assessment</th>
                </tr>
              </thead>
              <tbody>
                {data.degradation_curves.map((curve, i) => (
                  <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/50">
                    <td className="py-2 px-3">
                      <span className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full" style={{ backgroundColor: getModelColor(curve.model_name) }} />
                        {curve.model_name}
                      </span>
                    </td>
                    <td className="text-right py-2 px-3">{curve.total_moves}</td>
                    <td className="text-right py-2 px-3">
                      <span className={`font-medium ${
                        curve.degradation_ratio < 2 ? "text-green-400" :
                        curve.degradation_ratio < 5 ? "text-yellow-400" :
                        "text-red-400"
                      }`}>
                        {curve.degradation_ratio.toFixed(1)}x
                      </span>
                    </td>
                    <td className="text-right py-2 px-3">
                      {curve.critical_threshold ? `${curve.critical_threshold}s` : "—"}
                    </td>
                    <td className="py-2 px-3">
                      {curve.degradation_ratio < 2 ? (
                        <span className="text-green-400">🟢 Pressure Resilient</span>
                      ) : curve.degradation_ratio < 5 ? (
                        <span className="text-yellow-400">🟡 Moderate Degradation</span>
                      ) : (
                        <span className="text-red-400">🔴 Collapses Under Pressure</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 text-xs text-gray-500">
            💡 <strong>Degradation Ratio</strong> = (CP loss when &lt;60s) ÷ (CP loss when &gt;180s). Lower is better.
          </div>
        </div>
      </div>

      {/* 2. EFFICIENCY METRICS */}
      <div className="card p-6 mb-8">
        <h2 className="text-xl font-bold mb-2">⚡ Efficiency Analysis</h2>
        <p className="text-gray-400 text-sm mb-4">
          Quality per second spent. Are models using their time wisely?
        </p>
        
        <div className="overflow-x-auto mb-6">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-2 px-3">Model</th>
                <th className="text-right py-2 px-3">Avg Time/Move</th>
                <th className="text-right py-2 px-3">Avg CP Loss</th>
                <th className="text-right py-2 px-3">Quality/Second</th>
                <th className="text-right py-2 px-3">Quality/1K Tokens</th>
                <th className="text-left py-2 px-3">Optimal Time</th>
              </tr>
            </thead>
            <tbody>
              {data.efficiency_metrics.map((m, i) => (
                <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/50">
                  <td className="py-2 px-3">
                    <span className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full" style={{ backgroundColor: getModelColor(m.model_name) }} />
                      {m.model_name}
                    </span>
                  </td>
                  <td className="text-right py-2 px-3">{m.avg_time_per_move.toFixed(1)}s</td>
                  <td className="text-right py-2 px-3" style={{ color: getCpColor(m.avg_cp_loss) }}>
                    {m.avg_cp_loss.toFixed(1)}
                  </td>
                  <td className="text-right py-2 px-3 text-cyan-400 font-medium">
                    {m.quality_per_second.toFixed(2)}
                  </td>
                  <td className="text-right py-2 px-3 text-purple-400">
                    {m.quality_per_token ? ((100 - m.avg_cp_loss) / (m.avg_tokens_per_move! / 1000)).toFixed(1) : "—"}
                  </td>
                  <td className="py-2 px-3">
                    {m.optimal_time_range ? (
                      <span className="bg-cyan-500/20 text-cyan-400 px-2 py-0.5 rounded text-xs">
                        {m.optimal_time_range}
                      </span>
                    ) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="text-xs text-gray-500">
          💡 <strong>Quality/Second</strong> = (100 - Avg CP Loss) / Avg Time. Higher means more efficient use of time.
        </div>
      </div>

      {/* 3. TIME ALLOCATION */}
      <div className="card p-6 mb-8">
        <h2 className="text-xl font-bold mb-2">📊 Time Allocation Strategy</h2>
        <p className="text-gray-400 text-sm mb-4">
          How do models distribute their time across game phases? Do they spend more time on complex positions?
        </p>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Time by Phase */}
          <div>
            <h3 className="text-sm font-medium text-gray-400 mb-3">Avg Time Per Move by Phase</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={data.time_allocation.map((ta) => ({
                model: ta.model_name,
                Opening: ta.opening_avg_time,
                Middlegame: ta.middlegame_avg_time,
                Endgame: ta.endgame_avg_time,
              }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
                <XAxis dataKey="model" tick={{ fill: "#9ca3af", fontSize: 10 }} />
                <YAxis tick={{ fill: "#9ca3af", fontSize: 12 }} />
                <Tooltip contentStyle={{ backgroundColor: "#12121a", border: "1px solid #2e2e3e" }} />
                <Legend />
                <Bar dataKey="Opening" fill="#22c55e" />
                <Bar dataKey="Middlegame" fill="#f97316" />
                <Bar dataKey="Endgame" fill="#a855f7" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Complexity Correlation */}
          <div>
            <h3 className="text-sm font-medium text-gray-400 mb-3">Complexity Awareness</h3>
            <div className="space-y-4">
              {data.time_allocation.map((ta, i) => (
                <div key={i} className="bg-gray-800/50 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-3 h-3 rounded-full" style={{ backgroundColor: getModelColor(ta.model_name) }} />
                    <span className="font-medium">{ta.model_name}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Complexity → Time:</span>
                      <span className={`ml-2 font-medium ${
                        ta.complexity_time_correlation > 0.1 ? "text-green-400" :
                        ta.complexity_time_correlation < -0.1 ? "text-red-400" :
                        "text-gray-400"
                      }`}>
                        {formatCorrelation(ta.complexity_time_correlation)}
                      </span>
                      <span className="text-xs text-gray-500 ml-1">
                        {ta.complexity_time_correlation > 0.1 ? "✓ Spends more time on complex" : 
                         ta.complexity_time_correlation < -0.1 ? "✗ Rushes complex positions" : "~ Neutral"}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Complexity → Errors:</span>
                      <span className={`ml-2 font-medium ${
                        ta.complexity_quality_correlation > 0.2 ? "text-red-400" :
                        ta.complexity_quality_correlation < 0.1 ? "text-green-400" :
                        "text-yellow-400"
                      }`}>
                        {formatCorrelation(ta.complexity_quality_correlation)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 4. POSITION-DEPENDENT PERFORMANCE */}
      <div className="card p-6 mb-8">
        <h2 className="text-xl font-bold mb-2">♟️ Position-Dependent Performance</h2>
        <p className="text-gray-400 text-sm mb-4">
          How does performance vary with position complexity and criticality?
        </p>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-2 px-3">Model</th>
                <th className="text-right py-2 px-3">Simple Positions</th>
                <th className="text-right py-2 px-3">Complex Positions</th>
                <th className="text-right py-2 px-3">Complexity Penalty</th>
                <th className="text-right py-2 px-3">Routine Positions</th>
                <th className="text-right py-2 px-3">Critical Positions</th>
                <th className="text-right py-2 px-3">Time on Critical</th>
              </tr>
            </thead>
            <tbody>
              {data.position_performance.map((pp, i) => (
                <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/50">
                  <td className="py-2 px-3">
                    <span className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full" style={{ backgroundColor: getModelColor(pp.model_name) }} />
                      {pp.model_name}
                    </span>
                  </td>
                  <td className="text-right py-2 px-3" style={{ color: getCpColor(pp.simple_pos_quality) }}>
                    {pp.simple_pos_quality.toFixed(1)}
                  </td>
                  <td className="text-right py-2 px-3" style={{ color: getCpColor(pp.complex_pos_quality) }}>
                    {pp.complex_pos_quality.toFixed(1)}
                  </td>
                  <td className="text-right py-2 px-3">
                    <span className={`${pp.complexity_penalty > 20 ? "text-red-400" : pp.complexity_penalty > 10 ? "text-yellow-400" : "text-green-400"}`}>
                      +{pp.complexity_penalty.toFixed(1)}
                    </span>
                  </td>
                  <td className="text-right py-2 px-3" style={{ color: getCpColor(pp.routine_pos_quality) }}>
                    {pp.routine_pos_quality.toFixed(1)}
                  </td>
                  <td className="text-right py-2 px-3" style={{ color: getCpColor(pp.critical_pos_quality) }}>
                    {pp.critical_pos_quality.toFixed(1)}
                  </td>
                  <td className="text-right py-2 px-3">
                    <span className={`${pp.critical_time_ratio > 1.2 ? "text-green-400" : pp.critical_time_ratio < 0.9 ? "text-red-400" : "text-gray-400"}`}>
                      {pp.critical_time_ratio.toFixed(2)}x
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 text-xs text-gray-500">
          💡 <strong>Complexity Penalty</strong> = Extra CP loss in complex positions. <strong>Time on Critical</strong> = Ratio of time spent on critical vs routine positions (should be &gt;1.0).
        </div>
      </div>

      {/* 5. OUTCOME CORRELATES */}
      <div className="card p-6 mb-8">
        <h2 className="text-xl font-bold mb-2">🎯 What Predicts Winning?</h2>
        <p className="text-gray-400 text-sm mb-4">
          Correlation between various factors and game outcomes.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {data.outcome_correlates.map((oc, i) => (
            <div key={i} className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex justify-between items-start mb-2">
                <span className="font-medium">{oc.factor}</span>
                <span 
                  className="text-xl font-bold"
                  style={{ color: getCorrelationColor(oc.correlation_with_win) }}
                >
                  {formatCorrelation(oc.correlation_with_win)}
                </span>
              </div>
              <p className="text-xs text-gray-400">{oc.description}</p>
              <p className="text-xs text-gray-500 mt-1">n = {oc.sample_size}</p>
            </div>
          ))}
        </div>
        
        {data.outcome_correlates.length === 0 && (
          <div className="text-center text-gray-500 py-8">
            Need more completed games to calculate outcome correlates.
          </div>
        )}
      </div>

      {/* Scatter Plot: Time Remaining vs Quality */}
      <div className="card p-6 mb-8">
        <h2 className="text-xl font-bold mb-2">📈 Time vs Quality Scatter</h2>
        <p className="text-gray-400 text-sm mb-4">
          Each point is a move. Does more remaining time mean better moves? (CP loss capped at {data.cp_loss_cap})
        </p>
        
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
            <XAxis 
              dataKey="time_remaining" 
              type="number" 
              name="Time Remaining"
              tick={{ fill: "#9ca3af", fontSize: 12 }}
              label={{ value: "Time Remaining (s)", position: "bottom", fill: "#9ca3af", offset: 0 }}
            />
            <YAxis 
              dataKey="cp_loss" 
              type="number" 
              name="CP Loss"
              tick={{ fill: "#9ca3af", fontSize: 12 }}
              domain={[0, Math.min(data.cp_loss_cap, 150)]}
              label={{ value: "CP Loss", angle: -90, position: "insideLeft", fill: "#9ca3af" }}
            />
            <Tooltip 
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="bg-gray-900 border border-gray-700 rounded p-2 text-xs">
                    <div className="font-bold" style={{ color: getModelColor(d.model) }}>{d.model}</div>
                    <div>Time: {d.time_remaining?.toFixed(0)}s</div>
                    <div>CP Loss: {d.cp_loss?.toFixed(0)}</div>
                  </div>
                );
              }}
            />
            <Legend />
            {availableModels.map((model) => (
              <Scatter
                key={model}
                name={model}
                data={data.scatter_data.filter((d) => d.model === model)}
                fill={getModelColor(model)}
                fillOpacity={0.5}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Key Takeaways */}
      <div className="card p-6">
        <h2 className="text-xl font-bold mb-4">🔑 Key Research Questions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="font-medium text-cyan-400 mb-2">Degradation</div>
            <ul className="text-gray-400 space-y-1 list-disc list-inside">
              <li>Which model is most pressure-resilient?</li>
              <li>Is there a critical threshold where quality collapses?</li>
              <li>Is degradation linear or sudden?</li>
            </ul>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="font-medium text-purple-400 mb-2">Efficiency</div>
            <ul className="text-gray-400 space-y-1 list-disc list-inside">
              <li>What's the optimal time to spend per move?</li>
              <li>Are there diminishing returns on thinking time?</li>
              <li>Which model gets most quality per second?</li>
            </ul>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="font-medium text-green-400 mb-2">Strategy</div>
            <ul className="text-gray-400 space-y-1 list-disc list-inside">
              <li>Do models allocate time wisely across phases?</li>
              <li>Do they spend more time on complex positions?</li>
              <li>Do they recognize critical moments?</li>
            </ul>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="font-medium text-orange-400 mb-2">Outcomes</div>
            <ul className="text-gray-400 space-y-1 list-disc list-inside">
              <li>Does move quality predict winning?</li>
              <li>Are blunders more predictive than average quality?</li>
              <li>Does time management affect outcomes?</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
