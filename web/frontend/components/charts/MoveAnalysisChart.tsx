"use client";

import { useMemo } from "react";
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
import { MoveRecord } from "@/lib/api";

interface MoveAnalysisChartProps {
  moves: MoveRecord[];
  whiteModel: string;
  blackModel: string;
}

// Get color for CP loss
function getCpLossColor(cpLoss: number | null): string {
  if (cpLoss === null) return "#6b7280";
  if (cpLoss < 0) return "#10b981"; // emerald
  if (cpLoss <= 10) return "#22c55e"; // green
  if (cpLoss <= 25) return "#84cc16"; // lime
  if (cpLoss <= 50) return "#eab308"; // yellow
  if (cpLoss <= 100) return "#f97316"; // orange
  return "#ef4444"; // red
}

// Custom tooltip for scatterplot
function ScatterTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const data = payload[0].payload;
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-sm">
      <div className="font-medium text-white mb-1">Move {data.moveNum}</div>
      <div className="text-gray-400">
        <span className={data.color === "white" ? "text-white" : "text-indigo-400"}>
          {data.player}
        </span>
        {" played "}
        <span className="font-mono text-cyan-400">{data.move}</span>
      </div>
      <div className="mt-2 space-y-1">
        <div>Time: <span className="text-white">{data.timeTaken?.toFixed(1)}s</span></div>
        <div>
          CP Loss: <span style={{ color: getCpLossColor(data.cpLoss) }}>{data.cpLoss?.toFixed(0) ?? "—"}</span>
          {data.isCapped && <span className="text-yellow-400 ml-1">⚠️ outlier</span>}
        </div>
        {data.bestMove && data.move !== data.bestMove && (
          <div>Best: <span className="text-cyan-400 font-mono">{data.bestMove}</span></div>
        )}
      </div>
    </div>
  );
}

// Custom tooltip for bar chart
function CpLossTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const data = payload[0].payload;
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-sm">
      <div className="font-medium text-white mb-1">Move {label}</div>
      <div className="space-y-1">
        {data.whiteCpLoss !== null && (
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 bg-white rounded-sm" />
            <span>White: </span>
            <span style={{ color: getCpLossColor(data.whiteCpLoss) }}>
              {data.whiteCpLoss?.toFixed(0)} CP
              {data.whiteCpCapped && " ⚠️"}
            </span>
          </div>
        )}
        {data.blackCpLoss !== null && (
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 bg-indigo-500 rounded-sm" />
            <span>Black: </span>
            <span style={{ color: getCpLossColor(data.blackCpLoss) }}>
              {data.blackCpLoss?.toFixed(0)} CP
              {data.blackCpCapped && " ⚠️"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// Cap value and return if it was capped
function capValue(value: number | null, cap: number): { value: number | null; capped: boolean } {
  if (value === null) return { value: null, capped: false };
  if (value > cap) return { value: cap, capped: true };
  if (value < -cap) return { value: -cap, capped: true };
  return { value, capped: false };
}

const CP_LOSS_CAP = 150; // Cap at 150 for better visualization

export function MoveAnalysisChart({ moves, whiteModel, blackModel }: MoveAnalysisChartProps) {
  // Check if we have analysis data
  const hasAnalysis = useMemo(() => {
    return moves.some((m) => m.centipawn_loss !== null);
  }, [moves]);

  // Prepare data
  const { combinedData, scatterData, modelStats } = useMemo(() => {
    const whiteMoves = moves.filter((m) => m.color === "white");
    const blackMoves = moves.filter((m) => m.color === "black");
    
    const maxMoves = Math.max(whiteMoves.length, blackMoves.length);
    const combined = [];
    
    for (let i = 0; i < maxMoves; i++) {
      const whiteCp = whiteMoves[i]?.centipawn_loss ?? null;
      const blackCp = blackMoves[i]?.centipawn_loss ?? null;
      const whiteCapped = capValue(whiteCp, CP_LOSS_CAP);
      const blackCapped = capValue(blackCp, CP_LOSS_CAP);
      
      combined.push({
        moveNum: i + 1,
        whiteTime: whiteMoves[i]?.time_remaining ?? null,
        blackTime: blackMoves[i]?.time_remaining ?? null,
        whiteTimeTaken: whiteMoves[i]?.time_taken ?? null,
        blackTimeTaken: blackMoves[i]?.time_taken ?? null,
        whiteCpLoss: whiteCp,
        blackCpLoss: blackCp,
        whiteCpLossCapped: whiteCapped.value,
        blackCpLossCapped: blackCapped.value,
        whiteCpCapped: whiteCapped.capped,
        blackCpCapped: blackCapped.capped,
      });
    }
    
    // Scatter data - one point per move
    const scatter = moves.map((move, idx) => {
      const capped = capValue(move.centipawn_loss, CP_LOSS_CAP);
      return {
        moveNum: idx + 1,
        timeTaken: move.time_taken,
        thinkingTokens: move.thinking_tokens,
        cpLoss: move.centipawn_loss,
        cpLossCapped: capped.value,
        isCapped: capped.capped,
        color: move.color,
        player: move.player,
        move: move.move,
        bestMove: move.best_move,
        numLegalMoves: move.num_legal_moves,
        evalSharpness: move.eval_sharpness,
        positionEvalAbs: move.position_eval_abs,
      };
    });
    
    // Per-model stats
    const whiteAnalyzed = whiteMoves.filter(m => m.centipawn_loss !== null);
    const blackAnalyzed = blackMoves.filter(m => m.centipawn_loss !== null);
    
    const calcStats = (moves: MoveRecord[]) => {
      if (moves.length === 0) return null;
      const cpLosses = moves.map(m => m.centipawn_loss ?? 0);
      return {
        avgCpLoss: cpLosses.reduce((a, b) => a + b, 0) / moves.length,
        blunders: moves.filter(m => (m.centipawn_loss ?? 0) >= 100).length,
        mistakes: moves.filter(m => (m.centipawn_loss ?? 0) >= 50 && (m.centipawn_loss ?? 0) < 100).length,
        inaccuracies: moves.filter(m => (m.centipawn_loss ?? 0) >= 25 && (m.centipawn_loss ?? 0) < 50).length,
        good: moves.filter(m => (m.centipawn_loss ?? 0) < 25).length,
        totalMoves: moves.length,
      };
    };
    
    return {
      combinedData: combined,
      scatterData: scatter,
      modelStats: {
        white: calcStats(whiteAnalyzed),
        black: calcStats(blackAnalyzed),
      },
    };
  }, [moves]);

  return (
    <div className="space-y-8">
      {/* Per-Model Quality Summary */}
      {hasAnalysis && modelStats.white && modelStats.black && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* White model stats */}
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="w-4 h-4 bg-white rounded-sm" />
              <h3 className="font-medium">{whiteModel}</h3>
            </div>
            <div className="grid grid-cols-4 gap-2 text-center text-xs mb-3">
              <div className="bg-green-500/20 rounded p-2">
                <div className="text-green-400 font-bold">{modelStats.white.good}</div>
                <div className="text-gray-400">Good</div>
              </div>
              <div className="bg-yellow-500/20 rounded p-2">
                <div className="text-yellow-400 font-bold">{modelStats.white.inaccuracies}</div>
                <div className="text-gray-400">Inaccuracies</div>
              </div>
              <div className="bg-orange-500/20 rounded p-2">
                <div className="text-orange-400 font-bold">{modelStats.white.mistakes}</div>
                <div className="text-gray-400">Mistakes</div>
              </div>
              <div className="bg-red-500/20 rounded p-2">
                <div className="text-red-400 font-bold">{modelStats.white.blunders}</div>
                <div className="text-gray-400">Blunders</div>
              </div>
            </div>
            <div className="text-sm text-gray-400">
              Avg CP Loss: <span className="text-white font-medium">{modelStats.white.avgCpLoss.toFixed(1)}</span>
              <span className="text-gray-500 ml-2">({modelStats.white.totalMoves} moves)</span>
            </div>
          </div>

          {/* Black model stats */}
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="w-4 h-4 bg-indigo-500 rounded-sm" />
              <h3 className="font-medium">{blackModel}</h3>
            </div>
            <div className="grid grid-cols-4 gap-2 text-center text-xs mb-3">
              <div className="bg-green-500/20 rounded p-2">
                <div className="text-green-400 font-bold">{modelStats.black.good}</div>
                <div className="text-gray-400">Good</div>
              </div>
              <div className="bg-yellow-500/20 rounded p-2">
                <div className="text-yellow-400 font-bold">{modelStats.black.inaccuracies}</div>
                <div className="text-gray-400">Inaccuracies</div>
              </div>
              <div className="bg-orange-500/20 rounded p-2">
                <div className="text-orange-400 font-bold">{modelStats.black.mistakes}</div>
                <div className="text-gray-400">Mistakes</div>
              </div>
              <div className="bg-red-500/20 rounded p-2">
                <div className="text-red-400 font-bold">{modelStats.black.blunders}</div>
                <div className="text-gray-400">Blunders</div>
              </div>
            </div>
            <div className="text-sm text-gray-400">
              Avg CP Loss: <span className="text-white font-medium">{modelStats.black.avgCpLoss.toFixed(1)}</span>
              <span className="text-gray-500 ml-2">({modelStats.black.totalMoves} moves)</span>
            </div>
          </div>
        </div>
      )}

      {/* CP Loss per Move */}
      {hasAnalysis && (
        <div className="card p-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">
            Centipawn Loss per Move
            <span className="text-xs text-gray-500 ml-2">(capped at {CP_LOSS_CAP} for visibility)</span>
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={combinedData} barGap={0}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
              <XAxis
                dataKey="moveNum"
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                label={{
                  value: "Move",
                  position: "insideBottom",
                  offset: -5,
                  fill: "#9ca3af",
                }}
              />
              <YAxis
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                domain={[-50, CP_LOSS_CAP]}
                label={{
                  value: "CP Loss",
                  angle: -90,
                  position: "insideLeft",
                  fill: "#9ca3af",
                }}
              />
              <Tooltip content={<CpLossTooltip />} />
              <Legend />
              <ReferenceLine y={25} stroke="#eab308" strokeDasharray="3 3" label={{ value: "Inaccuracy", fill: "#eab308", fontSize: 10 }} />
              <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="3 3" label={{ value: "Blunder", fill: "#ef4444", fontSize: 10 }} />
              <Bar dataKey="whiteCpLossCapped" name={whiteModel} fill="#f8fafc" radius={[2, 2, 0, 0]}>
                {combinedData.map((entry, index) => (
                  <Cell 
                    key={`white-${index}`} 
                    fill={getCpLossColor(entry.whiteCpLoss)} 
                    stroke={entry.whiteCpCapped ? "#fff" : undefined}
                    strokeWidth={entry.whiteCpCapped ? 2 : 0}
                    strokeDasharray={entry.whiteCpCapped ? "3 2" : undefined}
                  />
                ))}
              </Bar>
              <Bar dataKey="blackCpLossCapped" name={blackModel} fill="#6366f1" radius={[2, 2, 0, 0]}>
                {combinedData.map((entry, index) => (
                  <Cell 
                    key={`black-${index}`} 
                    fill={getCpLossColor(entry.blackCpLoss)} 
                    opacity={0.7}
                    stroke={entry.blackCpCapped ? "#6366f1" : undefined}
                    strokeWidth={entry.blackCpCapped ? 2 : 0}
                    strokeDasharray={entry.blackCpCapped ? "3 2" : undefined}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Time vs Quality Scatterplot */}
      {hasAnalysis && (
        <div className="card p-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">
            Time vs Move Quality
            <span className="text-xs text-gray-500 ml-2">(hover for move details, capped at {CP_LOSS_CAP})</span>
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
              <XAxis
                dataKey="timeTaken"
                type="number"
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                label={{
                  value: "Time Taken (s)",
                  position: "insideBottom",
                  offset: -10,
                  fill: "#9ca3af",
                }}
                domain={[0, "auto"]}
              />
              <YAxis
                dataKey="cpLossCapped"
                type="number"
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                label={{
                  value: "CP Loss",
                  angle: -90,
                  position: "insideLeft",
                  fill: "#9ca3af",
                }}
                domain={[-25, CP_LOSS_CAP]}
              />
              <Tooltip content={<ScatterTooltip />} />
              <Legend />
              <ReferenceLine y={25} stroke="#eab308" strokeDasharray="3 3" />
              <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="3 3" />
              <Scatter
                name={`${whiteModel} (White)`}
                data={scatterData.filter(d => d.color === "white" && d.cpLoss !== null)}
                fill="#f8fafc"
              >
                {scatterData.filter(d => d.color === "white" && d.cpLoss !== null).map((entry, index) => (
                  <Cell 
                    key={`white-${index}`} 
                    fill={getCpLossColor(entry.cpLoss)} 
                    stroke={entry.isCapped ? "#fff" : "#fff"} 
                    strokeWidth={entry.isCapped ? 3 : 1}
                    strokeDasharray={entry.isCapped ? "2 1" : undefined}
                  />
                ))}
              </Scatter>
              <Scatter
                name={`${blackModel} (Black)`}
                data={scatterData.filter(d => d.color === "black" && d.cpLoss !== null)}
                fill="#6366f1"
                shape="square"
              >
                {scatterData.filter(d => d.color === "black" && d.cpLoss !== null).map((entry, index) => (
                  <Cell 
                    key={`black-${index}`} 
                    fill={getCpLossColor(entry.cpLoss)} 
                    stroke={entry.isCapped ? "#6366f1" : "#6366f1"} 
                    strokeWidth={entry.isCapped ? 3 : 1}
                    strokeDasharray={entry.isCapped ? "2 1" : undefined}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div className="text-xs text-gray-500 mt-2 text-center">
            🔍 Look for patterns: Do faster moves tend to have higher CP loss (worse quality)?
          </div>
        </div>
      )}

      {/* Time Analysis Charts */}
      <div className="card p-6">
        <h3 className="text-sm font-medium text-gray-400 mb-3">
          Time Remaining
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={combinedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
            <XAxis
              dataKey="moveNum"
              stroke="#6b7280"
              tick={{ fill: "#9ca3af", fontSize: 12 }}
              label={{
                value: "Move",
                position: "insideBottom",
                offset: -5,
                fill: "#9ca3af",
              }}
            />
            <YAxis
              stroke="#6b7280"
              tick={{ fill: "#9ca3af", fontSize: 12 }}
              label={{
                value: "Seconds",
                angle: -90,
                position: "insideLeft",
                fill: "#9ca3af",
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#12121a",
                border: "1px solid #2e2e3e",
                borderRadius: "8px",
              }}
              labelStyle={{ color: "#9ca3af" }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="whiteTime"
              name={whiteModel}
              stroke="#f8fafc"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="blackTime"
              name={blackModel}
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="card p-6">
        <h3 className="text-sm font-medium text-gray-400 mb-3">
          Time per Move
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={combinedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
            <XAxis
              dataKey="moveNum"
              stroke="#6b7280"
              tick={{ fill: "#9ca3af", fontSize: 12 }}
            />
            <YAxis
              stroke="#6b7280"
              tick={{ fill: "#9ca3af", fontSize: 12 }}
              label={{
                value: "Seconds",
                angle: -90,
                position: "insideLeft",
                fill: "#9ca3af",
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#12121a",
                border: "1px solid #2e2e3e",
                borderRadius: "8px",
              }}
              labelStyle={{ color: "#9ca3af" }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="whiteTimeTaken"
              name={`${whiteModel} (time/move)`}
              stroke="#f8fafc"
              strokeWidth={2}
              dot={{ r: 2 }}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="blackTimeTaken"
              name={`${blackModel} (time/move)`}
              stroke="#6366f1"
              strokeWidth={2}
              dot={{ r: 2 }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Time vs Thinking Tokens - sanity check */}
      {scatterData.some(d => d.thinkingTokens !== null) && (
        <div className="card p-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">
            Time Taken vs Thinking Tokens
            <span className="text-xs text-gray-500 ml-2">(sanity check: should correlate)</span>
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
              <XAxis
                dataKey="timeTaken"
                type="number"
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                label={{ value: "Time Taken (s)", position: "insideBottom", offset: -10, fill: "#9ca3af" }}
                domain={[0, "auto"]}
              />
              <YAxis
                dataKey="thinkingTokens"
                type="number"
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                label={{ value: "Thinking Tokens", angle: -90, position: "insideLeft", fill: "#9ca3af" }}
                domain={[0, "auto"]}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0].payload;
                  return (
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-sm">
                      <div className="font-medium text-white mb-1">Move {d.moveNum}: {d.move}</div>
                      <div className="text-gray-400 text-xs mb-2">{d.player}</div>
                      <div>Time: <span className="text-white">{d.timeTaken?.toFixed(1)}s</span></div>
                      <div>Tokens: <span className="text-cyan-400">{d.thinkingTokens?.toLocaleString()}</span></div>
                      <div>CP Loss: <span style={{ color: getCpLossColor(d.cpLoss) }}>{d.cpLoss?.toFixed(0) ?? "—"}</span></div>
                    </div>
                  );
                }}
              />
              <Legend />
              <Scatter
                name={`${whiteModel} (White)`}
                data={scatterData.filter(d => d.color === "white" && d.thinkingTokens !== null)}
                fill="#f8fafc"
              >
                {scatterData.filter(d => d.color === "white" && d.thinkingTokens !== null).map((entry, index) => (
                  <Cell key={`white-${index}`} fill={getCpLossColor(entry.cpLoss)} stroke="#fff" strokeWidth={1} />
                ))}
              </Scatter>
              <Scatter
                name={`${blackModel} (Black)`}
                data={scatterData.filter(d => d.color === "black" && d.thinkingTokens !== null)}
                fill="#6366f1"
                shape="square"
              >
                {scatterData.filter(d => d.color === "black" && d.thinkingTokens !== null).map((entry, index) => (
                  <Cell key={`black-${index}`} fill={getCpLossColor(entry.cpLoss)} stroke="#6366f1" strokeWidth={1} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div className="text-xs text-gray-500 mt-2 text-center">
            🔍 Colors indicate move quality (green=good, red=blunder). Outliers may indicate retries or issues.
          </div>
        </div>
      )}

      {/* Position Complexity vs CP Loss */}
      {hasAnalysis && scatterData.some(d => d.numLegalMoves !== null) && (
        <div className="card p-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">
            Position Complexity vs Move Quality
            <span className="text-xs text-gray-500 ml-2">(are harder positions causing mistakes?)</span>
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
              <XAxis
                dataKey="numLegalMoves"
                type="number"
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                label={{ value: "Legal Moves Available", position: "insideBottom", offset: -10, fill: "#9ca3af" }}
                domain={[0, "auto"]}
              />
              <YAxis
                dataKey="cpLossCapped"
                type="number"
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                label={{ value: "CP Loss", angle: -90, position: "insideLeft", fill: "#9ca3af" }}
                domain={[-25, CP_LOSS_CAP]}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0].payload;
                  return (
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-sm">
                      <div className="font-medium text-white mb-1">Move {d.moveNum}: {d.move}</div>
                      <div className="text-gray-400 text-xs mb-2">{d.player}</div>
                      <div>Legal Moves: <span className="text-cyan-400">{d.numLegalMoves}</span></div>
                      <div>Eval Sharpness: <span className="text-purple-400">{d.evalSharpness}</span></div>
                      <div>CP Loss: <span style={{ color: getCpLossColor(d.cpLoss) }}>{d.cpLoss?.toFixed(0) ?? "—"}</span></div>
                      <div>Time: <span className="text-white">{d.timeTaken?.toFixed(1)}s</span></div>
                    </div>
                  );
                }}
              />
              <Legend />
              <ReferenceLine y={25} stroke="#eab308" strokeDasharray="3 3" />
              <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="3 3" />
              <Scatter
                name={`${whiteModel} (White)`}
                data={scatterData.filter(d => d.color === "white" && d.numLegalMoves !== null && d.cpLoss !== null)}
                fill="#f8fafc"
              >
                {scatterData.filter(d => d.color === "white" && d.numLegalMoves !== null && d.cpLoss !== null).map((entry, index) => (
                  <Cell key={`white-${index}`} fill={getCpLossColor(entry.cpLoss)} stroke="#fff" strokeWidth={1} />
                ))}
              </Scatter>
              <Scatter
                name={`${blackModel} (Black)`}
                data={scatterData.filter(d => d.color === "black" && d.numLegalMoves !== null && d.cpLoss !== null)}
                fill="#6366f1"
                shape="square"
              >
                {scatterData.filter(d => d.color === "black" && d.numLegalMoves !== null && d.cpLoss !== null).map((entry, index) => (
                  <Cell key={`black-${index}`} fill={getCpLossColor(entry.cpLoss)} stroke="#6366f1" strokeWidth={1} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div className="text-xs text-gray-500 mt-2 text-center">
            🔍 More legal moves = more complex position. Look for clusters of mistakes in high-complexity regions.
          </div>
        </div>
      )}

      {/* Eval Sharpness vs CP Loss */}
      {hasAnalysis && scatterData.some(d => d.evalSharpness !== null) && (
        <div className="card p-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">
            Position Criticality vs Move Quality
            <span className="text-xs text-gray-500 ml-2">(eval sharpness = gap between best and 2nd best)</span>
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
              <XAxis
                dataKey="evalSharpness"
                type="number"
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                label={{ value: "Eval Sharpness (CP)", position: "insideBottom", offset: -10, fill: "#9ca3af" }}
                domain={[0, "auto"]}
              />
              <YAxis
                dataKey="cpLossCapped"
                type="number"
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                label={{ value: "CP Loss", angle: -90, position: "insideLeft", fill: "#9ca3af" }}
                domain={[-25, CP_LOSS_CAP]}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0].payload;
                  return (
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-sm">
                      <div className="font-medium text-white mb-1">Move {d.moveNum}: {d.move}</div>
                      <div className="text-gray-400 text-xs mb-2">{d.player}</div>
                      <div>Sharpness: <span className="text-purple-400">{d.evalSharpness} CP</span></div>
                      <div className="text-xs text-gray-500">(gap between best and 2nd best move)</div>
                      <div className="mt-1">CP Loss: <span style={{ color: getCpLossColor(d.cpLoss) }}>{d.cpLoss?.toFixed(0) ?? "—"}</span></div>
                      {d.bestMove && d.move !== d.bestMove && (
                        <div>Best: <span className="text-cyan-400 font-mono">{d.bestMove}</span></div>
                      )}
                    </div>
                  );
                }}
              />
              <Legend />
              <ReferenceLine y={25} stroke="#eab308" strokeDasharray="3 3" />
              <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="3 3" />
              <Scatter
                name={`${whiteModel} (White)`}
                data={scatterData.filter(d => d.color === "white" && d.evalSharpness !== null && d.cpLoss !== null)}
                fill="#f8fafc"
              >
                {scatterData.filter(d => d.color === "white" && d.evalSharpness !== null && d.cpLoss !== null).map((entry, index) => (
                  <Cell key={`white-${index}`} fill={getCpLossColor(entry.cpLoss)} stroke="#fff" strokeWidth={1} />
                ))}
              </Scatter>
              <Scatter
                name={`${blackModel} (Black)`}
                data={scatterData.filter(d => d.color === "black" && d.evalSharpness !== null && d.cpLoss !== null)}
                fill="#6366f1"
                shape="square"
              >
                {scatterData.filter(d => d.color === "black" && d.evalSharpness !== null && d.cpLoss !== null).map((entry, index) => (
                  <Cell key={`black-${index}`} fill={getCpLossColor(entry.cpLoss)} stroke="#6366f1" strokeWidth={1} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div className="text-xs text-gray-500 mt-2 text-center">
            🔍 High sharpness = critical position where only one move is good. Blunders in high-sharpness positions are more costly.
          </div>
        </div>
      )}
    </div>
  );
}

