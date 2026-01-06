/**
 * API client for Game Arena backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export interface Match {
  match_id: string;
  model_a: string;
  model_b: string;
  model_a_score: number;
  model_b_score: number;
  draws: number;
  winner: string;
  total_games: number;
  started_at: string;
  ended_at: string | null;
  time_control: string;
  status: "completed" | "live";
  notes?: string | null;
}

export interface MatchDetail extends Match {
  rethinking_enabled: boolean;
  games: GameSummary[];
}

export interface GameSummary {
  game_number: number;
  white_model: string;
  black_model: string;
  result: string;
  winner: string;
  termination: string;
  total_moves: number;
  duration_seconds: number;
}

export interface GameDetail extends GameSummary {
  match_id: string;
  moves: MoveRecord[];
}

export interface MoveRecord {
  move_number: number;
  player: string;
  color: string;
  move: string;
  fen_before: string;
  time_taken: number;
  time_remaining: number;
  thinking_tokens: number | null;
  // Stockfish analysis (populated if move analysis was run)
  centipawn_loss: number | null;
  is_best_move: boolean | null;
  is_blunder: boolean | null;  // True if CP loss >= 100
  best_move: string | null;  // The engine's preferred move
  win_probability_loss: number | null;  // WP loss from 0-1
  // Position complexity metrics
  num_legal_moves: number | null;  // Number of legal moves available
  eval_sharpness: number | null;  // CP diff between best and 2nd best move
  position_eval_abs: number | null;  // Absolute evaluation in CP
}

export interface ModelStats {
  model_id: string;
  display_name: string;
  elo: number;
  games_played: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  elo_change: number;
}

export interface Leaderboard {
  models: ModelStats[];
  last_updated: string;
}

// Analysis types
export interface PressureStats {
  pressure_level: string;
  move_count: number;
  avg_move_time: number;
  std_move_time: number;
  avg_thinking_tokens: number | null;
  avg_centipawn_loss: number | null;
  blunder_rate: number;
}

export interface ModelPressureProfile {
  name: string;
  total_moves: number;
  speed_adaptation_ratio: number;
  quality_degradation_ratio: number;
  thinking_reduction_ratio: number;
  pressure_stats: PressureStats[];
}

export interface MatchAnalysis {
  match_id: string;
  model_a: ModelPressureProfile;
  model_b: ModelPressureProfile;
  insights: string[];
}

export interface ScatterPoint {
  model: string;
  time_remaining: number;
  move_time: number;
  game_number: number;
  move_number: number;
  thinking_tokens: number | null;
}

export interface ScatterData {
  model_a: string;
  model_b: string;
  points: ScatterPoint[];
}

export interface ThinkingByPressure {
  model_a: string;
  model_b: string;
  data: {
    pressure: string;
    model_a_avg_tokens: number;
    model_b_avg_tokens: number;
    model_a_avg_time: number;
    model_b_avg_time: number;
    model_a_count: number;
    model_b_count: number;
  }[];
}

// Offline Evaluation types
export interface OfflineEvalSession {
  session_id: string;
  model_id: string;
  dataset_name: string;
  status: string;
  start_time: string | null;
  end_time: string | null;
  result_count: number;
  prompt_style: string;
}

export interface OfflineEvalSummary {
  total_sessions: number;
  total_evaluations: number;
  models: string[];
  prompt_styles: string[];
  time_levels: number[];
  overall_timeout_rate: number;
  has_move_quality: boolean;
}

export interface OfflineTimeoutData {
  model_id: string;
  time_remaining: number;
  timeouts: number;
  total: number;
  rate: number;
}

export interface OfflineStyleTimeoutData {
  model_id: string;
  prompt_style: string;
  timeouts: number;
  total: number;
  rate: number;
}

export interface OfflineTimeoutAnalysis {
  by_model_time: OfflineTimeoutData[];
  by_prompt_style: OfflineStyleTimeoutData[];
}

export interface OfflineResponseTimeData {
  model_id: string;
  time_remaining: number;
  avg_response_time: number | null;
  std_response_time: number | null;
  avg_thinking_tokens: number | null;
}

export interface OfflineMoveQualityData {
  model_id: string;
  time_remaining: number;
  avg_centipawn_loss: number | null;
  blunder_rate: number | null;
  best_move_rate: number | null;
}

export interface OfflineMoveQualityAnalysis {
  available: boolean;
  total_analyzed?: number;
  by_model_time: OfflineMoveQualityData[];
  by_prompt_style: {
    model_id: string;
    prompt_style: string;
    avg_centipawn_loss: number | null;
    blunder_rate: number | null;
  }[];
}

export interface OfflineAblationComparison {
  available: boolean;
  styles?: string[];
  models?: {
    [model_id: string]: {
      [style: string]: {
        evaluations: number;
        timeout_rate: number | null;
        avg_response_time: number;
        avg_thinking_tokens: number;
        avg_centipawn_loss?: number;
        blunder_rate?: number;
      };
    };
  };
}

export interface ModelProfile {
  model_id: string;
  display_name: string;
  total_matches: number;
  total_games: number;
  total_moves: number;
  wins: number;
  losses: number;
  draws: number;
  elo: number;
  win_rate: number;
  avg_move_time: number;
  avg_thinking_tokens: number | null;
  speed_adaptation_ratio: number;
  quality_degradation_ratio: number;
  thinking_reduction_ratio: number;
  pressure_stats: PressureStats[];
  recent_matches: {
    match_id: string;
    opponent: string;
    wins: number;
    losses: number;
    draws: number;
    result: string;
    date: string;
  }[];
}

// New Match Configuration types
export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
}

export interface TimeControlPreset {
  id: string;
  name: string;
  initial_time: number;
  increment: number;
}

export interface MatchConfig {
  model_a: string;
  model_b: string;
  initial_time_seconds: number;
  increment_seconds: number;
  first_to: number;
  use_rethinking: boolean;
  max_rethinks: number;
  max_parsing_failures: number;
  reasoning_budget_a: number;
  reasoning_budget_b: number;
  show_reasoning_a: boolean;
  show_reasoning_b: boolean;
  notes?: string;
}

export interface ProcessDetail {
  pid: number;
  status: "running" | "completed" | "failed" | "stopped";
  model_a: string;
  model_b: string;
  running_seconds: number;
  error: string | null;
  logs: { time: string; line: string }[];
}

export interface ConfigResponse {
  models: ModelInfo[];
  time_control_presets: TimeControlPreset[];
}

async function fetchJson<T>(endpoint: string): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  // Matches
  async getMatches(limit = 50, offset = 0): Promise<Match[]> {
    return fetchJson<Match[]>(`/matches?limit=${limit}&offset=${offset}`);
  },

  async getMatch(matchId: string): Promise<MatchDetail> {
    return fetchJson<MatchDetail>(`/matches/${matchId}`);
  },

  async deleteMatch(matchId: string): Promise<{ status: string; match_id: string }> {
    const res = await fetch(`${API_BASE}/matches/${matchId}`, { method: "DELETE" });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: "Failed to delete match" }));
      throw new Error(error.detail || "Failed to delete match");
    }
    return res.json();
  },

  async getGame(matchId: string, gameNumber: number): Promise<GameDetail> {
    return fetchJson<GameDetail>(`/matches/${matchId}/games/${gameNumber}`);
  },

  async getLiveGame(matchId: string, gameNumber: number): Promise<GameDetail> {
    return fetchJson<GameDetail>(`/matches/${matchId}/live/${gameNumber}`);
  },

  // Leaderboard
  async getLeaderboard(): Promise<Leaderboard> {
    return fetchJson<Leaderboard>("/leaderboard");
  },

  // Live
  async getLiveMatches(): Promise<Match[]> {
    return fetchJson<Match[]>("/live/matches");
  },

  async getStaleMatches(): Promise<Match[]> {
    return fetchJson<Match[]>("/live/stale");
  },

  async abandonMatch(matchId: string): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/live/abandon/${matchId}`, { method: "POST" });
    return res.json();
  },

  async refreshMatches(): Promise<{ status: string; matches_loaded: number }> {
    const res = await fetch(`${API_BASE}/matches/refresh`, { method: "POST" });
    return res.json();
  },

  // Analysis
  async getMatchAnalysis(matchId: string): Promise<MatchAnalysis> {
    return fetchJson<MatchAnalysis>(`/analysis/matches/${matchId}`);
  },

  async getPressureScatter(matchId: string): Promise<ScatterData> {
    return fetchJson<ScatterData>(`/analysis/matches/${matchId}/scatter`);
  },

  async getThinkingByPressure(matchId: string): Promise<ThinkingByPressure> {
    return fetchJson<ThinkingByPressure>(`/analysis/matches/${matchId}/thinking`);
  },

  // Models
  async getModels(): Promise<{ model_id: string; display_name: string; matches: number; wins: number; losses: number; win_rate: number }[]> {
    return fetchJson(`/models`);
  },

  async getModelProfile(modelId: string): Promise<ModelProfile> {
    return fetchJson<ModelProfile>(`/models/${modelId}`);
  },

  // Offline Evaluation
  async getOfflineEvalSessions(): Promise<{ sessions: OfflineEvalSession[] }> {
    return fetchJson(`/offline-eval/sessions`);
  },

  async getOfflineEvalSummary(): Promise<OfflineEvalSummary> {
    return fetchJson(`/offline-eval/summary`);
  },

  async getOfflineEvalTimeouts(): Promise<OfflineTimeoutAnalysis> {
    return fetchJson(`/offline-eval/analysis/timeouts`);
  },

  async getOfflineEvalResponseTimes(): Promise<{ data: OfflineResponseTimeData[] }> {
    return fetchJson(`/offline-eval/analysis/response-times`);
  },

  async getOfflineEvalMoveQuality(): Promise<OfflineMoveQualityAnalysis> {
    return fetchJson(`/offline-eval/analysis/move-quality`);
  },

  async getOfflineEvalAblation(): Promise<OfflineAblationComparison> {
    return fetchJson(`/offline-eval/analysis/ablation`);
  },

  // Match Configuration & Process Management
  async getConfig(): Promise<ConfigResponse> {
    return fetchJson<ConfigResponse>("/config");
  },

  async startMatch(config: MatchConfig): Promise<{ status: string; process_id?: number; error?: string; message?: string }> {
    const res = await fetch(`${API_BASE}/matches/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    return res.json();
  },

  async getProcessDetail(pid: number): Promise<ProcessDetail> {
    return fetchJson<ProcessDetail>(`/matches/processes/${pid}`);
  },

  async stopProcess(pid: number): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/matches/processes/${pid}/stop`, { method: "POST" });
    return res.json();
  },

  async getRunningProcesses(): Promise<{
    pid: number;
    status: string;
    model_a: string;
    model_b: string;
    started_at: string;
    running_seconds: number;
    log_count: number;
    last_log: string | null;
  }[]> {
    return fetchJson(`/matches/processes`);
  },
};

// WebSocket connection for live updates
export function createLiveConnection(
  matchId?: string,
  onMessage?: (data: any) => void
): WebSocket | null {
  if (typeof window === "undefined") return null;

  const wsBase = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/api/live";
  const url = matchId ? `${wsBase}/ws/${matchId}` : `${wsBase}/ws`;

  const ws = new WebSocket(url);

  ws.onopen = () => {
    console.log("WebSocket connected");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage?.(data);
    } catch (e) {
      // Handle pong messages
      if (event.data === "pong") return;
      console.error("Failed to parse WebSocket message:", e);
    }
  };

  ws.onerror = (error) => {
    console.error("WebSocket error:", error);
  };

  // Keepalive ping
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send("ping");
    }
  }, 30000);

  ws.onclose = () => {
    clearInterval(pingInterval);
    console.log("WebSocket disconnected");
  };

  return ws;
}
