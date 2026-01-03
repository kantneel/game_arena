"use client";

import Link from "next/link";
import { Match } from "@/lib/api";

interface MatchCardProps {
  match: Match;
}

export function MatchCard({ match }: MatchCardProps) {
  const isModelAWinner = match.winner === "model_a";
  const isModelBWinner = match.winner === "model_b";
  const isDraw = match.winner === "draw";

  // Format date
  const date = new Date(match.started_at);
  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  return (
    <Link href={`/matches/${match.match_id}`}>
      <div className="card-hover p-4 cursor-pointer">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs text-gray-500">{formattedDate}</span>
          <span className="text-xs text-gray-500">{match.time_control}</span>
        </div>

        {/* Models & Score */}
        <div className="flex items-center justify-between">
          {/* Model A */}
          <div className="flex-1">
            <div
              className={`font-medium truncate ${
                isModelAWinner ? "text-arena-win" : "text-gray-200"
              }`}
            >
              {formatModelName(match.model_a)}
            </div>
          </div>

          {/* Score */}
          <div className="px-4 text-center">
            <div className="text-xl font-bold font-mono">
              <span className={isModelAWinner ? "text-arena-win" : "text-gray-300"}>
                {match.model_a_score}
              </span>
              <span className="text-gray-500 mx-1">-</span>
              <span className={isModelBWinner ? "text-arena-win" : "text-gray-300"}>
                {match.model_b_score}
              </span>
            </div>
            {match.draws > 0 && (
              <div className="text-xs text-gray-500">{match.draws} draws</div>
            )}
          </div>

          {/* Model B */}
          <div className="flex-1 text-right">
            <div
              className={`font-medium truncate ${
                isModelBWinner ? "text-arena-win" : "text-gray-200"
              }`}
            >
              {formatModelName(match.model_b)}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
          <span>{match.total_games} games</span>
          {isDraw ? (
            <span className="badge-draw">Draw</span>
          ) : (
            <span className={isModelAWinner ? "badge-win" : "badge-loss"}>
              {isModelAWinner ? match.model_a : match.model_b} wins
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}

function formatModelName(name: string): string {
  // Shorten long model names for display
  return name
    .replace("claude-", "")
    .replace("gemini-", "")
    .replace("gpt-", "GPT ")
    .replace("-preview", "");
}

