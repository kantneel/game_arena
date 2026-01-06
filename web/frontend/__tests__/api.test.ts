import { describe, it, expect, vi, beforeEach } from "vitest";
import { api } from "@/lib/api";

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("API Client", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  describe("getMatches", () => {
    it("should fetch matches from the API", async () => {
      const mockMatches = [
        {
          match_id: "test-match",
          model_a: "claude-sonnet-4.5",
          model_b: "gemini-3-flash",
          model_a_score: 3,
          model_b_score: 2,
          draws: 0,
          winner: "model_a",
          total_games: 5,
          started_at: "2023-12-25T12:00:00",
          ended_at: "2023-12-25T13:00:00",
          time_control: "300+3",
          status: "completed",
        },
      ];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockMatches,
      });

      const result = await api.getMatches();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/matches?limit=50&offset=0")
      );
      expect(result).toEqual(mockMatches);
    });

    it("should throw on API error", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
      });

      await expect(api.getMatches()).rejects.toThrow("API error");
    });
  });

  describe("getMatchAnalysis", () => {
    it("should fetch analysis data for a match", async () => {
      const mockAnalysis = {
        match_id: "test-match",
        model_a: {
          name: "claude-sonnet-4.5",
          total_moves: 100,
          speed_adaptation_ratio: 0.7,
          quality_degradation_ratio: 1.2,
          thinking_reduction_ratio: 0.5,
          pressure_stats: [],
        },
        model_b: {
          name: "gemini-3-flash",
          total_moves: 100,
          speed_adaptation_ratio: 0.9,
          quality_degradation_ratio: 1.5,
          thinking_reduction_ratio: 0.8,
          pressure_stats: [],
        },
        insights: ["Test insight"],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockAnalysis,
      });

      const result = await api.getMatchAnalysis("test-match");

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/analysis/matches/test-match")
      );
      expect(result).toEqual(mockAnalysis);
    });
  });

  describe("getModelProfile", () => {
    it("should fetch model profile", async () => {
      const mockProfile = {
        model_id: "claude-sonnet-4.5",
        display_name: "Claude Sonnet 4.5",
        total_matches: 10,
        total_games: 50,
        total_moves: 1500,
        wins: 30,
        losses: 15,
        draws: 5,
        elo: 1650,
        win_rate: 0.6,
        avg_move_time: 5.5,
        avg_thinking_tokens: 8000,
        speed_adaptation_ratio: 0.7,
        quality_degradation_ratio: 1.1,
        thinking_reduction_ratio: 0.6,
        pressure_stats: [],
        recent_matches: [],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockProfile,
      });

      const result = await api.getModelProfile("claude-sonnet-4.5");

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/models/claude-sonnet-4.5")
      );
      expect(result).toEqual(mockProfile);
    });
  });

  describe("getPressureScatter", () => {
    it("should fetch scatter plot data", async () => {
      const mockScatter = {
        model_a: "claude-sonnet-4.5",
        model_b: "gemini-3-flash",
        points: [
          {
            model: "claude-sonnet-4.5",
            time_remaining: 250,
            move_time: 5.0,
            game_number: 1,
            move_number: 1,
            thinking_tokens: 10000,
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockScatter,
      });

      const result = await api.getPressureScatter("test-match");

      expect(result.points).toHaveLength(1);
      expect(result.points[0].model).toBe("claude-sonnet-4.5");
    });
  });
});

