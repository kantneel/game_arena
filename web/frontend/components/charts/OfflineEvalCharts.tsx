"use client";

import { useState } from "react";

interface TimeoutBarProps {
  model: string;
  timeLevel: number;
  rate: number;
  timeouts: number;
  total: number;
}

export function TimeoutBar({ model, timeLevel, rate, timeouts, total }: TimeoutBarProps) {
  const [hovered, setHovered] = useState(false);
  const percentage = rate * 100;
  
  const getColor = (pct: number) => {
    if (pct > 50) return { bg: "bg-red-500", text: "text-red-400", glow: "shadow-red-500/30" };
    if (pct > 20) return { bg: "bg-amber-500", text: "text-amber-400", glow: "shadow-amber-500/30" };
    return { bg: "bg-emerald-500", text: "text-emerald-400", glow: "shadow-emerald-500/30" };
  };
  
  const colors = getColor(percentage);
  
  return (
    <div 
      className="relative group cursor-pointer"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex items-center gap-4 py-2">
        {/* Label */}
        <div className="w-32 text-sm text-zinc-400 truncate">{timeLevel}s</div>
        
        {/* Bar container */}
        <div className="flex-1 h-8 bg-zinc-800 rounded-lg overflow-hidden relative">
          {/* Bar fill */}
          <div 
            className={`h-full ${colors.bg} transition-all duration-500 ease-out ${hovered ? `shadow-lg ${colors.glow}` : ''}`}
            style={{ width: `${Math.max(percentage, 2)}%` }}
          />
          
          {/* Percentage label inside bar */}
          <div className={`absolute inset-0 flex items-center px-3 ${percentage > 15 ? 'text-white' : colors.text} font-mono text-sm font-medium`}>
            {percentage.toFixed(1)}%
          </div>
        </div>
        
        {/* Count */}
        <div className="w-20 text-right text-sm text-zinc-500 font-mono">
          {timeouts}/{total}
        </div>
      </div>
      
      {/* Tooltip */}
      {hovered && (
        <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 z-50 pointer-events-none">
          <div className="bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3 shadow-xl min-w-[200px]">
            <div className="text-white font-medium mb-2">{model}</div>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-zinc-400">Time Level:</span>
                <span className="font-mono">{timeLevel}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Timeout Rate:</span>
                <span className={`font-mono ${colors.text}`}>{percentage.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Timeouts:</span>
                <span className="font-mono">{timeouts} of {total}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Success:</span>
                <span className="font-mono text-emerald-400">{total - timeouts} moves</span>
              </div>
            </div>
            {/* Arrow */}
            <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-zinc-700" />
          </div>
        </div>
      )}
    </div>
  );
}

interface ResponseTimeBarProps {
  model: string;
  timeLevel: number;
  avgTime: number;
  stdDev: number | null;
  avgTokens: number | null;
  maxTime?: number;
}

export function ResponseTimeBar({ model, timeLevel, avgTime, stdDev, avgTokens, maxTime = 60 }: ResponseTimeBarProps) {
  const [hovered, setHovered] = useState(false);
  const percentage = Math.min((avgTime / maxTime) * 100, 100);
  
  // Color based on whether response time exceeds time level
  const isOverTime = avgTime > timeLevel;
  const colors = isOverTime 
    ? { bg: "bg-gradient-to-r from-red-600 to-red-400", text: "text-red-400" }
    : avgTime > timeLevel * 0.7 
    ? { bg: "bg-gradient-to-r from-amber-600 to-amber-400", text: "text-amber-400" }
    : { bg: "bg-gradient-to-r from-cyan-600 to-cyan-400", text: "text-cyan-400" };
  
  return (
    <div 
      className="relative group cursor-pointer"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex items-center gap-4 py-2">
        {/* Label */}
        <div className="w-32 text-sm text-zinc-400">{timeLevel}s available</div>
        
        {/* Bar container */}
        <div className="flex-1 h-10 bg-zinc-800 rounded-lg overflow-hidden relative">
          {/* Time limit indicator */}
          <div 
            className="absolute top-0 bottom-0 w-0.5 bg-zinc-500 z-10"
            style={{ left: `${Math.min((timeLevel / maxTime) * 100, 100)}%` }}
          />
          
          {/* Bar fill */}
          <div 
            className={`h-full ${colors.bg} transition-all duration-500 ease-out ${hovered ? 'brightness-110' : ''}`}
            style={{ width: `${percentage}%` }}
          />
          
          {/* Time label */}
          <div className="absolute inset-0 flex items-center justify-between px-3">
            <span className={`font-mono text-sm font-medium ${percentage > 20 ? 'text-white' : colors.text}`}>
              {avgTime.toFixed(1)}s
            </span>
            {avgTokens && (
              <span className="text-xs text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded">
                {Math.round(avgTokens)} tokens
              </span>
            )}
          </div>
        </div>
      </div>
      
      {/* Tooltip */}
      {hovered && (
        <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 z-50 pointer-events-none">
          <div className="bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3 shadow-xl min-w-[240px]">
            <div className="text-white font-medium mb-2">{model}</div>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-zinc-400">Time Available:</span>
                <span className="font-mono">{timeLevel}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Avg Response:</span>
                <span className={`font-mono ${colors.text}`}>{avgTime.toFixed(2)}s</span>
              </div>
              {stdDev !== null && (
                <div className="flex justify-between">
                  <span className="text-zinc-400">Std Deviation:</span>
                  <span className="font-mono">±{stdDev.toFixed(2)}s</span>
                </div>
              )}
              {avgTokens !== null && (
                <div className="flex justify-between">
                  <span className="text-zinc-400">Avg Tokens:</span>
                  <span className="font-mono">{Math.round(avgTokens)}</span>
                </div>
              )}
              <div className="pt-2 border-t border-zinc-700 mt-2">
                <div className="flex justify-between">
                  <span className="text-zinc-400">Time Margin:</span>
                  <span className={`font-mono ${isOverTime ? 'text-red-400' : 'text-emerald-400'}`}>
                    {isOverTime ? '' : '+'}{(timeLevel - avgTime).toFixed(1)}s
                  </span>
                </div>
              </div>
            </div>
            <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-zinc-700" />
          </div>
        </div>
      )}
    </div>
  );
}

interface AblationCompareCardProps {
  model: string;
  styleA: {
    name: string;
    timeoutRate: number | null;
    avgTime: number;
    avgTokens: number;
    avgCpLoss?: number;
  };
  styleB: {
    name: string;
    timeoutRate: number | null;
    avgTime: number;
    avgTokens: number;
    avgCpLoss?: number;
  };
}

export function AblationCompareCard({ model, styleA, styleB }: AblationCompareCardProps) {
  const [hoveredSide, setHoveredSide] = useState<'a' | 'b' | null>(null);
  
  const timeoutDiff = (styleA.timeoutRate ?? 0) - (styleB.timeoutRate ?? 0);
  const timeDiff = styleA.avgTime - styleB.avgTime;
  const tokenDiff = styleA.avgTokens - styleB.avgTokens;
  
  const getBetterSide = (diff: number) => {
    if (Math.abs(diff) < 0.01) return 'tie';
    return diff > 0 ? 'b' : 'a';  // Lower is better for timeout, time, tokens
  };
  
  const timeoutWinner = getBetterSide(timeoutDiff);
  const timeWinner = getBetterSide(timeDiff);
  
  return (
    <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800 hover:border-zinc-700 transition-colors">
      <h3 className="text-lg font-semibold mb-4">{model}</h3>
      
      <div className="grid grid-cols-2 gap-4">
        {/* Style A */}
        <div 
          className={`p-4 rounded-lg transition-all cursor-pointer ${
            hoveredSide === 'a' ? 'bg-zinc-800 ring-2 ring-cyan-500/50' : 'bg-zinc-800/50'
          }`}
          onMouseEnter={() => setHoveredSide('a')}
          onMouseLeave={() => setHoveredSide(null)}
        >
          <div className="text-sm text-cyan-400 font-medium mb-3 capitalize">
            {styleA.name.replace('_', ' ')}
          </div>
          
          <div className="space-y-3">
            <MetricRow 
              label="Timeout" 
              value={styleA.timeoutRate !== null ? `${(styleA.timeoutRate * 100).toFixed(1)}%` : 'N/A'}
              isWinner={timeoutWinner === 'a'}
              isLoser={timeoutWinner === 'b'}
            />
            <MetricRow 
              label="Avg Time" 
              value={`${styleA.avgTime.toFixed(1)}s`}
              isWinner={timeWinner === 'a'}
              isLoser={timeWinner === 'b'}
            />
            <MetricRow 
              label="Tokens" 
              value={Math.round(styleA.avgTokens).toLocaleString()}
            />
            {styleA.avgCpLoss !== undefined && (
              <MetricRow 
                label="CP Loss" 
                value={styleA.avgCpLoss.toFixed(1)}
              />
            )}
          </div>
        </div>
        
        {/* Style B */}
        <div 
          className={`p-4 rounded-lg transition-all cursor-pointer ${
            hoveredSide === 'b' ? 'bg-zinc-800 ring-2 ring-purple-500/50' : 'bg-zinc-800/50'
          }`}
          onMouseEnter={() => setHoveredSide('b')}
          onMouseLeave={() => setHoveredSide(null)}
        >
          <div className="text-sm text-purple-400 font-medium mb-3 capitalize">
            {styleB.name.replace('_', ' ')}
          </div>
          
          <div className="space-y-3">
            <MetricRow 
              label="Timeout" 
              value={styleB.timeoutRate !== null ? `${(styleB.timeoutRate * 100).toFixed(1)}%` : 'N/A'}
              isWinner={timeoutWinner === 'b'}
              isLoser={timeoutWinner === 'a'}
            />
            <MetricRow 
              label="Avg Time" 
              value={`${styleB.avgTime.toFixed(1)}s`}
              isWinner={timeWinner === 'b'}
              isLoser={timeWinner === 'a'}
            />
            <MetricRow 
              label="Tokens" 
              value={Math.round(styleB.avgTokens).toLocaleString()}
            />
            {styleB.avgCpLoss !== undefined && (
              <MetricRow 
                label="CP Loss" 
                value={styleB.avgCpLoss.toFixed(1)}
              />
            )}
          </div>
        </div>
      </div>
      
      {/* Summary */}
      <div className="mt-4 pt-4 border-t border-zinc-800">
        <div className="text-sm text-zinc-400">
          {timeoutWinner === 'b' && timeoutDiff > 0.1 ? (
            <span>
              <span className="text-purple-400 font-medium">{styleB.name.replace('_', ' ')}</span>
              {' '}reduces timeout rate by{' '}
              <span className="text-emerald-400 font-mono">{Math.abs(timeoutDiff * 100).toFixed(1)}%</span>
            </span>
          ) : timeoutWinner === 'a' && timeoutDiff < -0.1 ? (
            <span>
              <span className="text-cyan-400 font-medium">{styleA.name.replace('_', ' ')}</span>
              {' '}has lower timeout rate by{' '}
              <span className="text-emerald-400 font-mono">{Math.abs(timeoutDiff * 100).toFixed(1)}%</span>
            </span>
          ) : (
            <span className="text-zinc-500">Similar timeout rates between styles</span>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricRow({ 
  label, 
  value, 
  isWinner = false,
  isLoser = false,
}: { 
  label: string; 
  value: string; 
  isWinner?: boolean;
  isLoser?: boolean;
}) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className={`font-mono text-sm ${
        isWinner ? 'text-emerald-400 font-medium' : 
        isLoser ? 'text-red-400/70' : 
        'text-zinc-300'
      }`}>
        {value}
        {isWinner && <span className="ml-1 text-xs">✓</span>}
      </span>
    </div>
  );
}

// Move Quality Charts

interface MoveQualityBarProps {
  model: string;
  timeLevel: number;
  avgCpLoss: number | null;
  blunderRate: number | null;
  bestMoveRate: number | null;
}

export function MoveQualityBar({ model, timeLevel, avgCpLoss, blunderRate, bestMoveRate }: MoveQualityBarProps) {
  const [hovered, setHovered] = useState(false);
  
  if (avgCpLoss === null) return null;
  
  // Color based on centipawn loss
  const getQualityColor = (cp: number) => {
    if (cp < 15) return { bg: "bg-emerald-500", text: "text-emerald-400", label: "Excellent" };
    if (cp < 30) return { bg: "bg-green-500", text: "text-green-400", label: "Good" };
    if (cp < 50) return { bg: "bg-yellow-500", text: "text-yellow-400", label: "Okay" };
    if (cp < 100) return { bg: "bg-orange-500", text: "text-orange-400", label: "Inaccurate" };
    return { bg: "bg-red-500", text: "text-red-400", label: "Poor" };
  };
  
  const colors = getQualityColor(avgCpLoss);
  const barWidth = Math.min((avgCpLoss / 150) * 100, 100);
  
  return (
    <div 
      className="relative group cursor-pointer"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex items-center gap-4 py-2">
        {/* Label */}
        <div className="w-32 text-sm text-zinc-400">{timeLevel}s</div>
        
        {/* Bar container */}
        <div className="flex-1 h-8 bg-zinc-800 rounded-lg overflow-hidden relative">
          {/* Bar fill - inverted (lower is better, so we show less bar) */}
          <div 
            className={`h-full ${colors.bg} transition-all duration-500 ease-out ${hovered ? 'brightness-110' : ''}`}
            style={{ width: `${barWidth}%` }}
          />
          
          {/* CP loss label */}
          <div className={`absolute inset-0 flex items-center px-3 ${barWidth > 25 ? 'text-white' : colors.text} font-mono text-sm font-medium`}>
            {avgCpLoss.toFixed(1)} cp
          </div>
        </div>
        
        {/* Quality label */}
        <div className={`w-24 text-right text-sm font-medium ${colors.text}`}>
          {colors.label}
        </div>
      </div>
      
      {/* Tooltip */}
      {hovered && (
        <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 z-50 pointer-events-none">
          <div className="bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3 shadow-xl min-w-[220px]">
            <div className="text-white font-medium mb-2">{model} @ {timeLevel}s</div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-zinc-400">Avg CP Loss:</span>
                <span className={`font-mono ${colors.text}`}>{avgCpLoss.toFixed(1)}</span>
              </div>
              {blunderRate !== null && (
                <div className="flex justify-between">
                  <span className="text-zinc-400">Blunder Rate:</span>
                  <span className={`font-mono ${blunderRate > 0.1 ? 'text-red-400' : 'text-emerald-400'}`}>
                    {(blunderRate * 100).toFixed(1)}%
                  </span>
                </div>
              )}
              {bestMoveRate !== null && (
                <div className="flex justify-between">
                  <span className="text-zinc-400">Best Move:</span>
                  <span className={`font-mono ${bestMoveRate > 0.3 ? 'text-emerald-400' : 'text-zinc-300'}`}>
                    {(bestMoveRate * 100).toFixed(1)}%
                  </span>
                </div>
              )}
              <div className="pt-2 border-t border-zinc-700 text-xs text-zinc-500">
                Lower CP loss = Better moves
              </div>
            </div>
            <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-zinc-700" />
          </div>
        </div>
      )}
    </div>
  );
}

interface BlunderRateBarProps {
  model: string;
  timeLevel: number;
  blunderRate: number;
  total?: number;
}

export function BlunderRateBar({ model, timeLevel, blunderRate }: BlunderRateBarProps) {
  const [hovered, setHovered] = useState(false);
  const percentage = blunderRate * 100;
  
  const getColor = (pct: number) => {
    if (pct < 5) return { bg: "bg-emerald-500", text: "text-emerald-400" };
    if (pct < 10) return { bg: "bg-green-500", text: "text-green-400" };
    if (pct < 20) return { bg: "bg-yellow-500", text: "text-yellow-400" };
    if (pct < 35) return { bg: "bg-orange-500", text: "text-orange-400" };
    return { bg: "bg-red-500", text: "text-red-400" };
  };
  
  const colors = getColor(percentage);
  
  return (
    <div 
      className="relative group cursor-pointer"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex items-center gap-4 py-2">
        <div className="w-32 text-sm text-zinc-400">{timeLevel}s</div>
        
        <div className="flex-1 h-8 bg-zinc-800 rounded-lg overflow-hidden relative">
          <div 
            className={`h-full ${colors.bg} transition-all duration-500 ease-out ${hovered ? 'brightness-110' : ''}`}
            style={{ width: `${Math.max(percentage, 2)}%` }}
          />
          
          <div className={`absolute inset-0 flex items-center px-3 ${percentage > 15 ? 'text-white' : colors.text} font-mono text-sm font-medium`}>
            {percentage.toFixed(1)}% blunders
          </div>
        </div>
      </div>
      
      {hovered && (
        <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 z-50 pointer-events-none">
          <div className="bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3 shadow-xl">
            <div className="text-white font-medium mb-2">{model}</div>
            <div className="text-sm">
              <span className="text-zinc-400">At {timeLevel}s remaining:</span>
              <span className={`ml-2 font-mono ${colors.text}`}>{percentage.toFixed(1)}%</span>
              <span className="text-zinc-500"> blunder rate</span>
            </div>
            <div className="text-xs text-zinc-500 mt-2">
              Blunder = 100+ centipawn loss
            </div>
            <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-zinc-700" />
          </div>
        </div>
      )}
    </div>
  );
}

interface QualityByStyleCardProps {
  model: string;
  styles: {
    name: string;
    avgCpLoss: number | null;
    blunderRate: number | null;
  }[];
}

export function QualityByStyleCard({ model, styles }: QualityByStyleCardProps) {
  const validStyles = styles.filter(s => s.avgCpLoss !== null);
  if (validStyles.length < 2) return null;
  
  // Find winner (lower CP loss is better)
  const sortedStyles = [...validStyles].sort((a, b) => (a.avgCpLoss ?? 999) - (b.avgCpLoss ?? 999));
  const winner = sortedStyles[0];
  const loser = sortedStyles[sortedStyles.length - 1];
  const improvement = (loser.avgCpLoss ?? 0) - (winner.avgCpLoss ?? 0);
  
  return (
    <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
      <h3 className="text-lg font-semibold mb-4">{model}</h3>
      
      <div className="space-y-4">
        {validStyles.map((style) => {
          const isWinner = style.name === winner.name;
          const cpLoss = style.avgCpLoss ?? 0;
          const maxCp = Math.max(...validStyles.map(s => s.avgCpLoss ?? 0));
          const barWidth = maxCp > 0 ? (cpLoss / maxCp) * 100 : 0;
          
          return (
            <div key={style.name} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className={`capitalize ${isWinner ? 'text-emerald-400 font-medium' : 'text-zinc-400'}`}>
                  {style.name.replace('_', ' ')}
                  {isWinner && ' ✓'}
                </span>
                <span className="font-mono text-zinc-300">
                  {cpLoss.toFixed(1)} cp
                </span>
              </div>
              <div className="h-3 bg-zinc-800 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full transition-all ${isWinner ? 'bg-emerald-500' : 'bg-zinc-600'}`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>
              {style.blunderRate !== null && (
                <div className="text-xs text-zinc-500">
                  Blunder rate: {(style.blunderRate * 100).toFixed(1)}%
                </div>
              )}
            </div>
          );
        })}
      </div>
      
      {improvement > 5 && (
        <div className="mt-4 pt-4 border-t border-zinc-800 text-sm text-zinc-400">
          <span className="text-emerald-400 font-medium capitalize">{winner.name.replace('_', ' ')}</span>
          {' '}produces{' '}
          <span className="text-emerald-400 font-mono">{improvement.toFixed(1)} cp</span>
          {' '}better moves
        </div>
      )}
    </div>
  );
}

