"use client";

import Link from "next/link";
import { Match } from "@/lib/api";

interface LiveMatchCardProps {
  match: Match;
}

export function LiveMatchCard({ match }: LiveMatchCardProps) {
  return (
    <Link href={`/live/${match.match_id}`}>
      <div className="card-hover p-6 cursor-pointer border-red-500/30 bg-gradient-to-br from-arena-card to-red-950/20">
        {/* Live Badge */}
        <div className="flex items-center gap-2 mb-4">
          <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          <span className="text-xs font-medium text-red-400 uppercase tracking-wider">
            Live
          </span>
        </div>

        {/* Models */}
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="text-lg font-bold text-white truncate">
              {match.model_a}
            </div>
          </div>

          <div className="px-6 text-center">
            <div className="text-2xl font-bold font-mono">
              <span className="text-white">{match.model_a_score}</span>
              <span className="text-gray-500 mx-2">-</span>
              <span className="text-white">{match.model_b_score}</span>
            </div>
            <div className="text-xs text-gray-400 mt-1">
              Game {match.total_games + 1}
            </div>
          </div>

          <div className="flex-1 text-right">
            <div className="text-lg font-bold text-white truncate">
              {match.model_b}
            </div>
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

