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
}

export interface MatchDetail extends Match {
  rethinking_enabled: boolean;
  games: GameSummary[];
  current_game?: number;  // Current game being played (for live matches)
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

async function fetchJson<T>(endpoint: string): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

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

export interface ConfigResponse {
  models: ModelInfo[];
  time_control_presets: TimeControlPreset[];
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
  // Per-model reasoning configuration
  reasoning_budget_a: number;
  reasoning_budget_b: number;
  show_reasoning_a?: boolean;
  show_reasoning_b?: boolean;
}

export interface StartMatchResponse {
  status: string;
  process_id?: number;
  message: string;
  command?: string;
  error?: string;
}

export interface ProcessStatus {
  pid: number;
  status: "starting" | "running" | "completed" | "failed" | "stopped";
  exit_code: number | null;
  model_a: string;
  model_b: string;
  started_at: string;
  running_seconds: number;
  log_count: number;
  last_log: string | null;
  error: string | null;
}

export interface ProcessDetail extends ProcessStatus {
  logs: { time: string; line: string }[];
}

export const api = {
  async getMatches(limit = 50, offset = 0): Promise<Match[]> {
    return fetchJson<Match[]>(`/matches?limit=${limit}&offset=${offset}`);
  },

  async getMatch(matchId: string): Promise<MatchDetail> {
    return fetchJson<MatchDetail>(`/matches/${matchId}`);
  },

  async getGame(matchId: string, gameNumber: number): Promise<GameDetail> {
    return fetchJson<GameDetail>(`/matches/${matchId}/games/${gameNumber}`);
  },

  async getLiveGame(matchId: string, gameNumber: number): Promise<GameDetail> {
    return fetchJson<GameDetail>(`/matches/${matchId}/live/${gameNumber}`);
  },

  async getLeaderboard(): Promise<Leaderboard> {
    return fetchJson<Leaderboard>("/leaderboard");
  },

  async getLiveMatches(): Promise<Match[]> {
    return fetchJson<Match[]>("/live/matches");
  },

  async getStaleMatches(): Promise<Match[]> {
    return fetchJson<Match[]>("/live/stale");
  },

  async abandonMatch(matchId: string): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/live/abandon/${matchId}`, {
      method: "POST",
    });
    return res.json();
  },

  async refreshMatches(): Promise<{ status: string; matches_loaded: number }> {
    const res = await fetch(`${API_BASE}/matches/refresh`, { method: "POST" });
    return res.json();
  },

  async getConfig(): Promise<ConfigResponse> {
    return fetchJson<ConfigResponse>("/config");
  },

  async startMatch(config: MatchConfig): Promise<StartMatchResponse> {
    const res = await fetch(`${API_BASE}/matches/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    if (!res.ok) {
      throw new Error(`Failed to start match: ${res.status}`);
    }
    return res.json();
  },

  async getProcesses(): Promise<ProcessStatus[]> {
    return fetchJson<ProcessStatus[]>("/matches/processes");
  },

  async getProcessDetail(pid: number): Promise<ProcessDetail> {
    const res = await fetch(`${API_BASE}/matches/processes/${pid}`);
    if (!res.ok) {
      throw new Error(`Process not found: ${res.status}`);
    }
    return res.json();
  },

  async stopProcess(pid: number): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/matches/processes/${pid}/stop`, {
      method: "POST",
    });
    return res.json();
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

