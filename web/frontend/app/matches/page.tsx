"use client";

import { useEffect, useState } from "react";
import { MatchCard } from "@/components/ui/MatchCard";
import { api, Match } from "@/lib/api";

export default function MatchesPage() {
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchMatches() {
      try {
        const data = await api.getMatches(100);
        setMatches(data);
      } catch (error) {
        console.error("Failed to fetch matches:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchMatches();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-gray-400">Loading matches...</div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold mb-2">All Matches</h1>
        <p className="text-gray-400">
          Browse all completed and live chess matches between AI models
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {matches.map((match, i) => (
          <div
            key={match.match_id}
            className="animate-slide-up"
            style={{ animationDelay: `${i * 30}ms` }}
          >
            <MatchCard match={match} />
          </div>
        ))}
      </div>

      {matches.length === 0 && (
        <div className="text-center py-12 text-gray-400">
          No matches found. Run some games to see them here!
        </div>
      )}
    </div>
  );
}

