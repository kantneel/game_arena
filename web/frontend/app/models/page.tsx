"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";

interface ModelListItem {
  model_id: string;
  display_name: string;
  matches: number;
  wins: number;
  losses: number;
  win_rate: number;
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchModels() {
      try {
        const data = await api.getModels();
        setModels(data);
      } catch (error) {
        console.error("Failed to fetch models:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchModels();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-gray-400">Loading models...</div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Model Profiles</h1>
        <div className="text-gray-400">{models.length} models</div>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {models.map((model) => (
          <Link
            key={model.model_id}
            href={`/models/${encodeURIComponent(model.model_id)}`}
            className="card p-6 hover:border-arena-accent transition-all group"
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="font-bold text-lg group-hover:text-arena-accent transition-colors">
                  {model.display_name}
                </h3>
                <div className="text-xs text-gray-500 font-mono">{model.model_id}</div>
              </div>
              <div
                className={`text-2xl font-bold ${
                  model.win_rate > 0.5
                    ? "text-arena-win"
                    : model.win_rate < 0.5
                    ? "text-arena-loss"
                    : "text-gray-400"
                }`}
              >
                {(model.win_rate * 100).toFixed(0)}%
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-xl font-mono font-bold text-arena-win">
                  {model.wins}
                </div>
                <div className="text-xs text-gray-500">Wins</div>
              </div>
              <div>
                <div className="text-xl font-mono font-bold text-arena-loss">
                  {model.losses}
                </div>
                <div className="text-xs text-gray-500">Losses</div>
              </div>
              <div>
                <div className="text-xl font-mono font-bold text-gray-400">
                  {model.matches}
                </div>
                <div className="text-xs text-gray-500">Matches</div>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-arena-border text-sm text-arena-accent opacity-0 group-hover:opacity-100 transition-opacity text-center">
              View Profile →
            </div>
          </Link>
        ))}
      </div>

      {models.length === 0 && (
        <div className="text-center py-16 text-gray-400">
          <div className="text-4xl mb-4">🤖</div>
          <div>No models found. Run some matches first!</div>
        </div>
      )}
    </div>
  );
}

