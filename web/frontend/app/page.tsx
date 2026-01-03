"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { MatchCard } from "@/components/ui/MatchCard";
import { LeaderboardPreview } from "@/components/ui/LeaderboardPreview";
import { LiveMatchCard } from "@/components/ui/LiveMatchCard";
import { api, Match, ModelStats } from "@/lib/api";

export default function Home() {
  const [matches, setMatches] = useState<Match[]>([]);
  const [leaderboard, setLeaderboard] = useState<ModelStats[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [matchesData, leaderboardData] = await Promise.all([
          api.getMatches(10),
          api.getLeaderboard(),
        ]);
        setMatches(matchesData);
        setLeaderboard(leaderboardData.models.slice(0, 5));
      } catch (error) {
        console.error("Failed to fetch data:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  const liveMatches = matches.filter((m) => m.status === "live");
  const recentMatches = matches.filter((m) => m.status === "completed").slice(0, 6);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-gray-400">Loading...</div>
      </div>
    );
  }

  return (
    <div className="space-y-12 animate-fade-in">
      {/* Hero */}
      <section className="text-center py-8">
        <h1 className="text-5xl font-bold bg-gradient-to-r from-arena-accent to-purple-400 bg-clip-text text-transparent mb-4">
          Game Arena
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          Watch the world&apos;s most powerful AI models battle it out in blitz chess
        </p>
      </section>

      {/* Live Matches */}
      {liveMatches.length > 0 && (
        <section>
          <div className="flex items-center gap-3 mb-6">
            <span className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            <h2 className="text-2xl font-bold">Live Now</h2>
          </div>
          <div className="grid gap-6 md:grid-cols-2">
            {liveMatches.map((match) => (
              <LiveMatchCard key={match.match_id} match={match} />
            ))}
          </div>
        </section>
      )}

      {/* Recent Matches & Leaderboard */}
      <div className="grid gap-8 lg:grid-cols-3">
        <section className="lg:col-span-2">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold">Recent Matches</h2>
            <Link
              href="/matches"
              className="text-arena-accent hover:text-arena-accent-dim transition-colors"
            >
              View all →
            </Link>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            {recentMatches.map((match, i) => (
              <div
                key={match.match_id}
                className="animate-slide-up"
                style={{ animationDelay: `${i * 50}ms` }}
              >
                <MatchCard match={match} />
              </div>
            ))}
          </div>
        </section>

        <section>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold">Rankings</h2>
            <Link
              href="/leaderboard"
              className="text-arena-accent hover:text-arena-accent-dim transition-colors"
            >
              Full rankings →
            </Link>
          </div>
          <LeaderboardPreview models={leaderboard} />
        </section>
      </div>
    </div>
  );
}

