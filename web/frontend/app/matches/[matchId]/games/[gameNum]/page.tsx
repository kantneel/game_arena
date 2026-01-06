"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { api, GameDetail } from "@/lib/api";
import { GameReplay } from "@/components/chess/GameReplay";
import { MoveAnalysisChart } from "@/components/charts/MoveAnalysisChart";

export default function GameDetailPage() {
  const params = useParams();
  const matchId = params.matchId as string;
  const gameNum = parseInt(params.gameNum as string);
  const [game, setGame] = useState<GameDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchGame() {
      try {
        const data = await api.getGame(matchId, gameNum);
        setGame(data);
      } catch (error) {
        console.error("Failed to fetch game:", error);
      } finally {
        setLoading(false);
      }
    }

    if (matchId && gameNum) {
      fetchGame();
    }
  }, [matchId, gameNum]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-gray-400">Loading game...</div>
      </div>
    );
  }

  if (!game) {
    return (
      <div className="text-center py-16">
        <h1 className="text-2xl font-bold text-gray-300">Game not found</h1>
        <Link
          href={`/matches/${matchId}`}
          className="text-arena-accent hover:underline mt-4 block"
        >
          ← Back to match
        </Link>
      </div>
    );
  }

  const isWhiteWin = game.result === "1-0";
  const isBlackWin = game.result === "0-1";

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link
            href={`/matches/${matchId}`}
            className="text-sm text-gray-400 hover:text-white transition-colors mb-2 block"
          >
            ← Back to Match
          </Link>
          <h1 className="text-2xl font-bold">Game {game.game_number}</h1>
        </div>
        <div className="text-right">
          <div className="text-2xl font-mono font-bold">{game.result}</div>
          <div className="text-sm text-gray-400 capitalize">
            {game.termination.replace("_", " ")}
          </div>
        </div>
      </div>

      {/* Players */}
      <div className="card p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="w-6 h-6 bg-white rounded" />
            <div>
              <div
                className={`font-bold ${isWhiteWin ? "text-arena-win" : ""}`}
              >
                {game.white_model}
              </div>
              <div className="text-sm text-gray-400">White</div>
            </div>
          </div>

          <div className="text-3xl font-mono font-bold px-8">
            {game.result}
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <div
                className={`font-bold ${isBlackWin ? "text-arena-win" : ""}`}
              >
                {game.black_model}
              </div>
              <div className="text-sm text-gray-400">Black</div>
            </div>
            <span className="w-6 h-6 bg-gray-800 rounded border border-gray-600" />
          </div>
        </div>
      </div>

      {/* Game Replay */}
      <div>
        <h2 className="text-xl font-bold mb-4">Game Replay</h2>
        <GameReplay
          moves={game.moves}
          whiteModel={game.white_model}
          blackModel={game.black_model}
        />
      </div>

      {/* Move Analysis */}
      {game.moves.length > 0 && (
        <div>
          <h2 className="text-xl font-bold mb-4">Move Analysis</h2>
          <MoveAnalysisChart 
            moves={game.moves} 
            whiteModel={game.white_model}
            blackModel={game.black_model}
          />
        </div>
      )}

      {/* Game Stats */}
      <div>
        <h2 className="text-xl font-bold mb-4">Statistics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="stat-card">
            <div className="stat-value">{game.total_moves}</div>
            <div className="stat-label">Total Moves</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">
              {formatDuration(game.duration_seconds)}
            </div>
            <div className="stat-label">Duration</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">
              {game.moves.length > 0
                ? (
                    game.moves.reduce((sum, m) => sum + m.time_taken, 0) /
                    game.moves.length
                  ).toFixed(1)
                : 0}
              s
            </div>
            <div className="stat-label">Avg Move Time</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">
              {game.moves.filter((m) => m.thinking_tokens).length > 0
                ? Math.round(
                    game.moves
                      .filter((m) => m.thinking_tokens)
                      .reduce((sum, m) => sum + (m.thinking_tokens || 0), 0) /
                      game.moves.filter((m) => m.thinking_tokens).length
                  ).toLocaleString()
                : "N/A"}
            </div>
            <div className="stat-label">Avg Thinking Tokens</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

