"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { api, MatchDetail, GameSummary } from "@/lib/api";
import { MatchAnalysisTab } from "@/components/analysis/MatchAnalysisTab";

type TabId = "games" | "analysis";

export default function MatchDetailPage() {
  const params = useParams();
  const matchId = params.matchId as string;
  const [match, setMatch] = useState<MatchDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabId>("games");

  useEffect(() => {
    async function fetchMatch() {
      try {
        const data = await api.getMatch(matchId);
        setMatch(data);
      } catch (error) {
        console.error("Failed to fetch match:", error);
      } finally {
        setLoading(false);
      }
    }

    if (matchId) {
      fetchMatch();
    }
  }, [matchId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-gray-400">Loading match...</div>
      </div>
    );
  }

  if (!match) {
    return (
      <div className="text-center py-16">
        <h1 className="text-2xl font-bold text-gray-300">Match not found</h1>
        <Link href="/matches" className="text-arena-accent hover:underline mt-4 block">
          ← Back to matches
        </Link>
      </div>
    );
  }

  const isModelAWinner = match.winner === "model_a";
  const isModelBWinner = match.winner === "model_b";

  const tabs: { id: TabId; label: string; icon: string }[] = [
    { id: "games", label: "Games", icon: "♟️" },
    { id: "analysis", label: "Time Pressure Analysis", icon: "📊" },
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link
            href="/matches"
            className="text-sm text-gray-400 hover:text-white transition-colors mb-2 block"
          >
            ← All Matches
          </Link>
          <h1 className="text-2xl font-bold">Match Details</h1>
        </div>
        <div className="text-right text-sm text-gray-400">
          <div>{new Date(match.started_at).toLocaleString()}</div>
          <div>{match.time_control}</div>
        </div>
      </div>

      {/* Score Card */}
      <div className="card p-8">
        <div className="flex items-center justify-between max-w-2xl mx-auto">
          {/* Model A */}
          <div className="text-center flex-1">
            <Link
              href={`/models/${encodeURIComponent(match.model_a)}`}
              className="text-xl font-bold hover:text-arena-accent transition-colors block"
            >
              <span className={isModelAWinner ? "text-arena-win" : "text-gray-200"}>
                {match.model_a}
              </span>
            </Link>
            {isModelAWinner && (
              <div className="text-arena-win text-sm mt-1">👑 Winner</div>
            )}
          </div>

          {/* Score */}
          <div className="text-center px-8">
            <div className="text-5xl font-bold font-mono">
              <span className={isModelAWinner ? "text-arena-win" : "text-white"}>
                {match.model_a_score}
              </span>
              <span className="text-gray-500 mx-3">-</span>
              <span className={isModelBWinner ? "text-arena-win" : "text-white"}>
                {match.model_b_score}
              </span>
            </div>
            {match.draws > 0 && (
              <div className="text-gray-400 mt-2">{match.draws} draws</div>
            )}
          </div>

          {/* Model B */}
          <div className="text-center flex-1">
            <Link
              href={`/models/${encodeURIComponent(match.model_b)}`}
              className="text-xl font-bold hover:text-arena-accent transition-colors block"
            >
              <span className={isModelBWinner ? "text-arena-win" : "text-gray-200"}>
                {match.model_b}
              </span>
            </Link>
            {isModelBWinner && (
              <div className="text-arena-win text-sm mt-1">👑 Winner</div>
            )}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-arena-border">
        <div className="flex gap-4">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-3 font-medium transition-colors relative ${
                activeTab === tab.id
                  ? "text-arena-accent"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
              {activeTab === tab.id && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-arena-accent" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === "games" && (
        <GamesTab match={match} matchId={matchId} />
      )}
      {activeTab === "analysis" && (
        <MatchAnalysisTab matchId={matchId} />
      )}
    </div>
  );
}

function GamesTab({ match, matchId }: { match: MatchDetail; matchId: string }) {
  return (
    <div className="space-y-8 animate-fade-in">
      {/* Games Table */}
      <div>
        <h2 className="text-xl font-bold mb-4">Games</h2>
        <div className="card overflow-hidden">
          <table className="w-full">
            <thead className="bg-arena-border/50">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                  Game
                </th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                  White
                </th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                  Black
                </th>
                <th className="px-4 py-3 text-center text-sm font-medium text-gray-400">
                  Result
                </th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                  Termination
                </th>
                <th className="px-4 py-3 text-right text-sm font-medium text-gray-400">
                  Moves
                </th>
                <th className="px-4 py-3 text-right text-sm font-medium text-gray-400">
                  Duration
                </th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-arena-border">
              {match.games.map((game) => (
                <GameRow
                  key={game.game_number}
                  game={game}
                  matchId={matchId}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Match Stats */}
      <div>
        <h2 className="text-xl font-bold mb-4">Match Statistics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="stat-card">
            <div className="stat-value">{match.total_games}</div>
            <div className="stat-label">Total Games</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{match.rethinking_enabled ? "Yes" : "No"}</div>
            <div className="stat-label">Rethinking</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{match.time_control}</div>
            <div className="stat-label">Time Control</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">
              {match.games.reduce((sum, g) => sum + g.total_moves, 0)}
            </div>
            <div className="stat-label">Total Moves</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function GameRow({ game, matchId }: { game: GameSummary; matchId: string }) {
  const isWhiteWin = game.result === "1-0";
  const isBlackWin = game.result === "0-1";

  return (
    <tr className="hover:bg-arena-border/30 transition-colors">
      <td className="px-4 py-3 font-mono text-gray-400">#{game.game_number}</td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-white rounded-sm" />
          <span className={isWhiteWin ? "text-arena-win font-medium" : ""}>
            {game.white_model}
          </span>
        </div>
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-gray-800 rounded-sm border border-gray-600" />
          <span className={isBlackWin ? "text-arena-win font-medium" : ""}>
            {game.black_model}
          </span>
        </div>
      </td>
      <td className="px-4 py-3 text-center font-mono font-bold">
        {game.result}
      </td>
      <td className="px-4 py-3 text-sm text-gray-400 capitalize">
        {game.termination.replace("_", " ")}
      </td>
      <td className="px-4 py-3 text-right font-mono">{game.total_moves}</td>
      <td className="px-4 py-3 text-right text-gray-400">
        {formatDuration(game.duration_seconds)}
      </td>
      <td className="px-4 py-3">
        <Link
          href={`/matches/${matchId}/games/${game.game_number}`}
          className="text-arena-accent hover:text-arena-accent-dim transition-colors text-sm"
        >
          View →
        </Link>
      </td>
    </tr>
  );
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}
