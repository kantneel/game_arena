"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api, Match } from "@/lib/api";

type FilterOption = "all" | "completed" | "with_games";
type ViewMode = "grid" | "table";

export default function MatchesPage() {
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterOption>("with_games");
  const [viewMode, setViewMode] = useState<ViewMode>("table");
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const fetchMatches = async () => {
    try {
      const data = await api.getMatches(100);
      setMatches(data);
    } catch (error) {
      console.error("Failed to fetch matches:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMatches();
  }, []);

  const handleDelete = async (matchId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (deleteConfirm !== matchId) {
      setDeleteConfirm(matchId);
      // Auto-dismiss after 3 seconds
      setTimeout(() => setDeleteConfirm(null), 3000);
      return;
    }
    
    setDeleting(matchId);
    try {
      await api.deleteMatch(matchId);
      setMatches(prev => prev.filter(m => m.match_id !== matchId));
    } catch (error) {
      console.error("Failed to delete match:", error);
    } finally {
      setDeleting(null);
      setDeleteConfirm(null);
    }
  };

  // Filter and sort matches
  const filteredMatches = useMemo(() => {
    let result = [...matches];
    
    // Apply filter
    if (filter === "completed") {
      result = result.filter(m => m.status === "completed" && m.winner);
    } else if (filter === "with_games") {
      result = result.filter(m => m.total_games > 0);
    }
    
    // Sort: live first, then by date (newest first)
    result.sort((a, b) => {
      if (a.status === "live" && b.status !== "live") return -1;
      if (b.status === "live" && a.status !== "live") return 1;
      return new Date(b.started_at).getTime() - new Date(a.started_at).getTime();
    });
    
    return result;
  }, [matches, filter]);

  // Group matches by month
  const groupedMatches = useMemo(() => {
    const groups: { [key: string]: Match[] } = {};
    
    filteredMatches.forEach(match => {
      const date = new Date(match.started_at);
      const key = date.toLocaleDateString("en-US", { month: "long", year: "numeric" });
      if (!groups[key]) groups[key] = [];
      groups[key].push(match);
    });
    
    return groups;
  }, [filteredMatches]);

  const stats = useMemo(() => {
    const completed = matches.filter(m => m.total_games > 0 && m.winner).length;
    const incomplete = matches.filter(m => m.total_games === 0).length;
    const live = matches.filter(m => m.status === "live").length;
    return { completed, incomplete, live, total: matches.length };
  }, [matches]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-gray-400">Loading matches...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">All Matches</h1>
          <p className="text-gray-400">
            {stats.total} matches • {stats.completed} completed • {stats.live} live
          </p>
        </div>
        
        {/* View Mode Toggle */}
        <div className="flex items-center gap-2 bg-arena-border rounded-lg p-1">
          <button
            onClick={() => setViewMode("table")}
            className={`p-2 rounded transition-colors ${
              viewMode === "table" ? "bg-arena-accent text-white" : "text-gray-400 hover:text-white"
            }`}
            title="Table view"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
            </svg>
          </button>
          <button
            onClick={() => setViewMode("grid")}
            className={`p-2 rounded transition-colors ${
              viewMode === "grid" ? "bg-arena-accent text-white" : "text-gray-400 hover:text-white"
            }`}
            title="Grid view"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zM14 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
            </svg>
          </button>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">Show:</span>
          <div className="flex gap-1">
            <FilterButton 
              active={filter === "with_games"} 
              onClick={() => setFilter("with_games")}
              label={`With Games (${stats.completed})`}
            />
            <FilterButton 
              active={filter === "completed"} 
              onClick={() => setFilter("completed")}
              label="Completed Only"
            />
            <FilterButton 
              active={filter === "all"} 
              onClick={() => setFilter("all")}
              label={`All (${stats.total})`}
            />
          </div>
        </div>
      </div>

      {/* Table View */}
      {viewMode === "table" && (
        <div className="space-y-6">
          {Object.entries(groupedMatches).map(([month, monthMatches]) => (
            <div key={month}>
              <h2 className="text-sm font-medium text-gray-400 mb-3 sticky top-0 bg-arena-bg py-2">
                {month}
              </h2>
              <div className="card overflow-hidden">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-arena-border text-xs text-gray-500 uppercase">
                      <th className="text-left py-3 px-4 font-medium">Date</th>
                      <th className="text-left py-3 px-4 font-medium">Model A</th>
                      <th className="text-center py-3 px-4 font-medium">Score</th>
                      <th className="text-right py-3 px-4 font-medium">Model B</th>
                      <th className="text-center py-3 px-4 font-medium">Games</th>
                      <th className="text-center py-3 px-4 font-medium">Time</th>
                      <th className="text-left py-3 px-4 font-medium">Notes</th>
                      <th className="text-center py-3 px-4 font-medium">Status</th>
                      <th className="text-center py-3 px-4 font-medium w-12"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {monthMatches.map((match) => (
                      <MatchRow 
                        key={match.match_id} 
                        match={match}
                        isDeleting={deleting === match.match_id}
                        isConfirming={deleteConfirm === match.match_id}
                        onDelete={(e) => handleDelete(match.match_id, e)}
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Grid View */}
      {viewMode === "grid" && (
        <div className="space-y-6">
          {Object.entries(groupedMatches).map(([month, monthMatches]) => (
            <div key={month}>
              <h2 className="text-sm font-medium text-gray-400 mb-3">
                {month}
              </h2>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {monthMatches.map((match, i) => (
                  <MatchCard 
                    key={match.match_id} 
                    match={match}
                    delay={i * 30}
                    isDeleting={deleting === match.match_id}
                    isConfirming={deleteConfirm === match.match_id}
                    onDelete={(e) => handleDelete(match.match_id, e)}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {filteredMatches.length === 0 && (
        <div className="text-center py-12 text-gray-400">
          {filter === "all" 
            ? "No matches found. Run some games to see them here!"
            : "No matches match this filter. Try selecting 'All'."}
        </div>
      )}
    </div>
  );
}

function MatchRow({ 
  match, 
  isDeleting, 
  isConfirming, 
  onDelete 
}: { 
  match: Match; 
  isDeleting: boolean;
  isConfirming: boolean;
  onDelete: (e: React.MouseEvent) => void;
}) {
  const router = useRouter();
  const isModelAWinner = match.winner === "model_a";
  const isModelBWinner = match.winner === "model_b";
  const isIncomplete = match.total_games === 0;
  
  const date = new Date(match.started_at);
  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  const handleRowClick = (e: React.MouseEvent) => {
    // Don't navigate if clicking on delete button or notes tooltip
    if ((e.target as HTMLElement).closest('button') || (e.target as HTMLElement).closest('.group')) {
      return;
    }
    router.push(`/matches/${match.match_id}`);
  };

  return (
    <tr 
      onClick={handleRowClick}
      className={`border-b border-arena-border/50 last:border-b-0 hover:bg-arena-border/30 transition-colors cursor-pointer ${
        isIncomplete ? "opacity-50" : ""
      }`}
    >
      <td className="py-3 px-4 text-sm text-gray-400">{formattedDate}</td>
      <td className="py-3 px-4">
        <span className={isModelAWinner ? "text-arena-win font-medium" : ""}>
          {formatModelName(match.model_a)}
        </span>
      </td>
      <td className="py-3 px-4 text-center">
        <span className="font-mono font-bold">
          <span className={isModelAWinner ? "text-arena-win" : ""}>{match.model_a_score}</span>
          <span className="text-gray-500 mx-1">-</span>
          <span className={isModelBWinner ? "text-arena-win" : ""}>{match.model_b_score}</span>
        </span>
      </td>
      <td className="py-3 px-4 text-right">
        <span className={isModelBWinner ? "text-arena-win font-medium" : ""}>
          {formatModelName(match.model_b)}
        </span>
      </td>
      <td className="py-3 px-4 text-center text-sm text-gray-400">
        {match.total_games}
      </td>
      <td className="py-3 px-4 text-center text-sm text-gray-500 font-mono">
        {match.time_control}
      </td>
      <td className="py-3 px-4 text-left">
        {match.notes ? (
          <span className="relative group cursor-help">
            <span className="text-xs text-gray-400 bg-arena-border/50 px-2 py-1 rounded truncate max-w-[150px] inline-block">
              📝 {match.notes.length > 20 ? match.notes.slice(0, 20) + "..." : match.notes}
            </span>
            <span className="absolute bottom-full left-0 mb-2 px-3 py-2 text-xs text-white bg-gray-800 rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-20 max-w-[300px] break-words">
              {match.notes}
            </span>
          </span>
        ) : (
          <span className="text-gray-600 text-xs">—</span>
        )}
      </td>
      <td className="py-3 px-4 text-center">
        <StatusBadge match={match} />
      </td>
      <td className="py-3 px-4 text-center">
        <button
          onClick={onDelete}
          disabled={isDeleting}
          className={`p-1.5 rounded transition-all ${
            isConfirming 
              ? "bg-red-500 text-white" 
              : "text-gray-500 hover:text-red-400 hover:bg-red-500/10"
          }`}
          title={isConfirming ? "Click again to confirm" : "Delete match"}
        >
          {isDeleting ? (
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          )}
        </button>
      </td>
    </tr>
  );
}

function MatchCard({ 
  match, 
  delay,
  isDeleting, 
  isConfirming, 
  onDelete 
}: { 
  match: Match; 
  delay: number;
  isDeleting: boolean;
  isConfirming: boolean;
  onDelete: (e: React.MouseEvent) => void;
}) {
  const isModelAWinner = match.winner === "model_a";
  const isModelBWinner = match.winner === "model_b";
  const isDraw = match.winner === "draw";
  const isIncomplete = match.total_games === 0;

  const date = new Date(match.started_at);
  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  return (
    <div
      className="animate-slide-up relative group"
      style={{ animationDelay: `${delay}ms` }}
    >
      <Link href={`/matches/${match.match_id}`}>
        <div className={`card-hover p-4 cursor-pointer ${isIncomplete ? "opacity-60" : ""}`}>
          {/* Delete Button */}
          <button
            onClick={onDelete}
            disabled={isDeleting}
            className={`absolute top-2 right-2 p-1.5 rounded opacity-0 group-hover:opacity-100 transition-all ${
              isConfirming 
                ? "bg-red-500 text-white opacity-100" 
                : "text-gray-500 hover:text-red-400 hover:bg-red-500/10"
            }`}
            title={isConfirming ? "Click again to confirm" : "Delete match"}
          >
            {isDeleting ? (
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            )}
          </button>

          {/* Header */}
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs text-gray-500">{formattedDate}</span>
            <span className="text-xs text-gray-500 font-mono">{match.time_control}</span>
          </div>

          {/* Models & Score */}
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className={`font-medium truncate ${isModelAWinner ? "text-arena-win" : ""}`}>
                {formatModelName(match.model_a)}
              </div>
            </div>
            <div className="px-4 text-center">
              <div className="text-xl font-bold font-mono">
                <span className={isModelAWinner ? "text-arena-win" : ""}>{match.model_a_score}</span>
                <span className="text-gray-500 mx-1">-</span>
                <span className={isModelBWinner ? "text-arena-win" : ""}>{match.model_b_score}</span>
              </div>
            </div>
            <div className="flex-1 text-right">
              <div className={`font-medium truncate ${isModelBWinner ? "text-arena-win" : ""}`}>
                {formatModelName(match.model_b)}
              </div>
            </div>
          </div>

          {/* Notes */}
          {match.notes && (
            <div className="mt-2 relative group cursor-help">
              <div className="text-xs text-gray-500 bg-arena-border/30 px-2 py-1 rounded truncate">
                📝 {match.notes}
              </div>
              <div className="absolute bottom-full left-0 mb-2 px-3 py-2 text-xs text-white bg-gray-800 rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-normal pointer-events-none z-20 max-w-[250px]">
                {match.notes}
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
            <span>{match.total_games} game{match.total_games !== 1 ? 's' : ''}</span>
            <StatusBadge match={match} />
          </div>
        </div>
      </Link>
    </div>
  );
}

function StatusBadge({ match }: { match: Match }) {
  const isModelAWinner = match.winner === "model_a";
  const isModelBWinner = match.winner === "model_b";
  const isDraw = match.winner === "draw";
  
  if (match.status === "live") {
    return <span className="badge-live text-xs">🔴 Live</span>;
  }
  if (match.total_games === 0) {
    return <span className="text-gray-500 text-xs">No games</span>;
  }
  if (isDraw) {
    return <span className="badge-draw text-xs">Draw</span>;
  }
  if (isModelAWinner) {
    return <span className="badge-win text-xs">{formatModelName(match.model_a)} wins</span>;
  }
  if (isModelBWinner) {
    return <span className="badge-win text-xs">{formatModelName(match.model_b)} wins</span>;
  }
  return <span className="text-gray-500 text-xs">Incomplete</span>;
}

function FilterButton({ 
  active, 
  onClick, 
  label 
}: { 
  active: boolean; 
  onClick: () => void; 
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
        active 
          ? "bg-arena-accent text-white" 
          : "bg-arena-border text-gray-400 hover:bg-arena-accent/20 hover:text-white"
      }`}
    >
      {label}
    </button>
  );
}

function formatModelName(name: string): string {
  return name
    .replace("claude-", "")
    .replace("gemini-", "")
    .replace("gpt-", "GPT ")
    .replace("-preview", "");
}
