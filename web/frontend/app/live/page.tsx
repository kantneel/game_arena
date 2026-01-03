"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { LiveMatchCard } from "@/components/ui/LiveMatchCard";
import { api, Match } from "@/lib/api";

export default function LivePage() {
  const [liveMatches, setLiveMatches] = useState<Match[]>([]);
  const [staleMatches, setStaleMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchMatches = async () => {
    try {
      const [live, stale] = await Promise.all([
        api.getLiveMatches(),
        api.getStaleMatches(),
      ]);
      setLiveMatches(live);
      setStaleMatches(stale);
    } catch (error) {
      console.error("Failed to fetch matches:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMatches();
    
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchMatches, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleAbandon = async (matchId: string) => {
    try {
      await api.abandonMatch(matchId);
      fetchMatches();
    } catch (error) {
      console.error("Failed to abandon match:", error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-gray-400">Checking for live matches...</div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
          <h1 className="text-3xl font-bold">Live Matches</h1>
        </div>
        <Link
          href="/new-match"
          className="px-4 py-2 bg-arena-accent text-white rounded-lg hover:opacity-90 transition-opacity"
        >
          + New Match
        </Link>
      </div>

      {/* Live Matches */}
      {liveMatches.length > 0 ? (
        <div className="grid gap-6 md:grid-cols-2">
          {liveMatches.map((match) => (
            <LiveMatchCard key={match.match_id} match={match} />
          ))}
        </div>
      ) : (
        <div className="text-center py-16 card">
          <div className="text-6xl mb-4">😴</div>
          <h2 className="text-xl font-medium text-gray-300 mb-2">
            No live matches right now
          </h2>
          <p className="text-gray-500 mb-6">
            Start a match to see it here in real-time
          </p>
          <Link
            href="/new-match"
            className="inline-block px-6 py-3 bg-gradient-to-r from-arena-accent to-purple-500 text-white rounded-lg hover:opacity-90 transition-opacity"
          >
            ⚔️ Start a Match
          </Link>
        </div>
      )}

      {/* Stale/Abandoned Matches */}
      {staleMatches.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-gray-400 flex items-center gap-2">
            <span className="text-yellow-500">⚠️</span> Stale Matches
            <span className="text-sm font-normal text-gray-500">
              (No activity for 5+ minutes)
            </span>
          </h2>
          <div className="grid gap-4 md:grid-cols-2">
            {staleMatches.map((match) => (
              <div
                key={match.match_id}
                className="card p-4 border-yellow-500/30 bg-yellow-500/5"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">
                      {match.model_a} vs {match.model_b}
                    </div>
                    <div className="text-sm text-gray-500">
                      {match.model_a_score} - {match.model_b_score} • {match.time_control}
                    </div>
                  </div>
                  <button
                    onClick={() => handleAbandon(match.match_id)}
                    className="px-3 py-1 text-sm bg-yellow-500/20 text-yellow-400 rounded hover:bg-yellow-500/30 transition-colors"
                  >
                    Mark Abandoned
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

