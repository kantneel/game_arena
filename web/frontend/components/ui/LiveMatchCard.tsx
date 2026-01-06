"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Match, GameDetail, api } from "@/lib/api";

interface LiveMatchCardProps {
  match: Match;
}

interface TimeState {
  modelATime: number;
  modelBTime: number;
  initialTime: number;
  toMove: "model_a" | "model_b";
  moveCount: number;
}

export function LiveMatchCard({ match }: LiveMatchCardProps) {
  const [timeState, setTimeState] = useState<TimeState | null>(null);

  useEffect(() => {
    const fetchGameTime = async () => {
      try {
        // Calculate which game is currently in progress
        const currentGameNum = match.total_games + 1;
        
        // Try to fetch the live game data
        const gameData = await api.getLiveGame(match.match_id, currentGameNum);
        
        if (gameData && gameData.moves) {
          const moves = gameData.moves;
          
          // Parse initial time from time control (e.g., "300+3" -> 300)
          const timeMatch = match.time_control.match(/^(\d+)/);
          const initialTime = timeMatch ? parseInt(timeMatch[1], 10) : 300;
          
          // For odd games, model_a is white; for even, model_b is white
          const modelAIsWhite = currentGameNum % 2 === 1;
          
          // Calculate remaining times
          const getTimeForColor = (color: string): number => {
            const colorMoves = moves.filter(m => m.color === color);
            if (colorMoves.length === 0) return initialTime;
            return colorMoves[colorMoves.length - 1].time_remaining;
          };
          
          const modelATime = modelAIsWhite 
            ? getTimeForColor("white") 
            : getTimeForColor("black");
          const modelBTime = modelAIsWhite 
            ? getTimeForColor("black") 
            : getTimeForColor("white");
          
          // Who's to move
          const toMove = moves.length % 2 === 0
            ? (modelAIsWhite ? "model_a" : "model_b")
            : (modelAIsWhite ? "model_b" : "model_a");
          
          setTimeState({
            modelATime,
            modelBTime,
            initialTime,
            toMove,
            moveCount: moves.length,
          });
        }
      } catch (error) {
        // Game data not available yet
        console.log("Live game data not available yet");
      }
    };

    fetchGameTime();
    
    // Refresh every 3 seconds
    const interval = setInterval(fetchGameTime, 1000);
    return () => clearInterval(interval);
  }, [match.match_id, match.total_games, match.time_control]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(Math.max(0, seconds) / 60);
    const secs = Math.floor(Math.max(0, seconds) % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const getTimeBarPercent = (time: number, initial: number): number => {
    return Math.max(0, Math.min(100, (time / initial) * 100));
  };

  const getTimeBarColor = (time: number, initial: number): string => {
    const percent = (time / initial) * 100;
    if (percent > 50) return "bg-green-500";
    if (percent > 25) return "bg-yellow-500";
    if (percent > 10) return "bg-orange-500";
    return "bg-red-500";
  };

  return (
    <Link href={`/live/${match.match_id}`}>
      <div className="card-hover p-6 cursor-pointer border-red-500/30 bg-gradient-to-br from-arena-card to-red-950/20">
        {/* Live Badge */}
        <div className="flex items-center gap-2 mb-4">
          <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          <span className="text-xs font-medium text-red-400 uppercase tracking-wider">
            Live
          </span>
          {match.notes && (
            <span 
              className="text-xs text-gray-500 hover:text-gray-300 cursor-help relative group"
              title={match.notes}
            >
              📝
              <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 text-xs text-white bg-gray-800 rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
                {match.notes}
              </span>
            </span>
          )}
          {timeState && (
            <span className="text-xs text-gray-500 ml-auto">
              Move {timeState.moveCount}
            </span>
          )}
        </div>

        {/* Models with Time Bars */}
        <div className="flex items-center justify-between gap-4">
          {/* Model A */}
          <div className="flex-1 min-w-0">
            <div className={`text-lg font-bold truncate ${
              timeState?.toMove === "model_a" ? "text-arena-accent" : "text-white"
            }`}>
              {match.model_a}
              {timeState?.toMove === "model_a" && (
                <span className="ml-2 text-xs text-arena-accent animate-pulse">●</span>
              )}
            </div>
            {timeState && (
              <div className="mt-2 space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className={`font-mono font-bold ${
                    timeState.modelATime < 30 ? "text-red-400" : "text-gray-300"
                  }`}>
                    {formatTime(timeState.modelATime)}
                  </span>
                </div>
                <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-500 ${getTimeBarColor(timeState.modelATime, timeState.initialTime)}`}
                    style={{ width: `${getTimeBarPercent(timeState.modelATime, timeState.initialTime)}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Score */}
          <div className="px-4 text-center flex-shrink-0">
            <div className="text-2xl font-bold font-mono">
              <span className="text-white">{match.model_a_score}</span>
              <span className="text-gray-500 mx-2">-</span>
              <span className="text-white">{match.model_b_score}</span>
            </div>
            <div className="text-xs text-gray-400 mt-1">
              Game {match.total_games + 1}
            </div>
          </div>

          {/* Model B */}
          <div className="flex-1 min-w-0 text-right">
            <div className={`text-lg font-bold truncate ${
              timeState?.toMove === "model_b" ? "text-arena-accent" : "text-white"
            }`}>
              {timeState?.toMove === "model_b" && (
                <span className="mr-2 text-xs text-arena-accent animate-pulse">●</span>
              )}
              {match.model_b}
            </div>
            {timeState && (
              <div className="mt-2 space-y-1">
                <div className="flex items-center justify-end text-xs">
                  <span className={`font-mono font-bold ${
                    timeState.modelBTime < 30 ? "text-red-400" : "text-gray-300"
                  }`}>
                    {formatTime(timeState.modelBTime)}
                  </span>
                </div>
                <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-500 ml-auto ${getTimeBarColor(timeState.modelBTime, timeState.initialTime)}`}
                    style={{ width: `${getTimeBarPercent(timeState.modelBTime, timeState.initialTime)}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Time Control */}
        <div className="mt-4 text-center text-sm text-gray-400">
          {match.time_control} • Click to watch
        </div>
      </div>
    </Link>
  );
}
