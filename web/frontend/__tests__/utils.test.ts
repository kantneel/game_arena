import { describe, it, expect } from "vitest";

// Test utility functions that we use in components
describe("Utility Functions", () => {
  describe("formatDuration", () => {
    // Inline implementation for testing (same as in match page)
    function formatDuration(seconds: number): string {
      const mins = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      return `${mins}:${secs.toString().padStart(2, "0")}`;
    }

    it("formats seconds correctly", () => {
      expect(formatDuration(0)).toBe("0:00");
      expect(formatDuration(30)).toBe("0:30");
      expect(formatDuration(60)).toBe("1:00");
      expect(formatDuration(90)).toBe("1:30");
      expect(formatDuration(125)).toBe("2:05");
      expect(formatDuration(599)).toBe("9:59");
      expect(formatDuration(600)).toBe("10:00");
    });
  });

  describe("pressure level formatting", () => {
    // Matches the PRESSURE_THRESHOLDS in backend
    function categorizePressure(timeRemaining: number): string {
      if (timeRemaining < 30) return "critical";
      if (timeRemaining < 60) return "high";
      if (timeRemaining < 120) return "medium";
      return "comfortable";
    }

    it("categorizes time remaining correctly", () => {
      expect(categorizePressure(0)).toBe("critical");
      expect(categorizePressure(15)).toBe("critical");
      expect(categorizePressure(29)).toBe("critical");
      expect(categorizePressure(30)).toBe("high");
      expect(categorizePressure(45)).toBe("high");
      expect(categorizePressure(59)).toBe("high");
      expect(categorizePressure(60)).toBe("medium");
      expect(categorizePressure(90)).toBe("medium");
      expect(categorizePressure(119)).toBe("medium");
      expect(categorizePressure(120)).toBe("comfortable");
      expect(categorizePressure(300)).toBe("comfortable");
    });
  });

  describe("win rate calculation", () => {
    function calculateWinRate(wins: number, losses: number, draws: number): number {
      const total = wins + losses + draws;
      return total > 0 ? wins / total : 0;
    }

    it("calculates win rate correctly", () => {
      expect(calculateWinRate(5, 3, 2)).toBe(0.5);
      expect(calculateWinRate(10, 0, 0)).toBe(1);
      expect(calculateWinRate(0, 10, 0)).toBe(0);
      expect(calculateWinRate(0, 0, 0)).toBe(0);
      expect(calculateWinRate(3, 1, 1)).toBe(0.6);
    });
  });

  describe("adaptation ratio interpretation", () => {
    function interpretSpeedAdaptation(ratio: number): string {
      if (ratio < 0.7) return "significantly sped up";
      if (ratio < 0.9) return "moderately adapted";
      if (ratio > 1.1) return "slowed down";
      return "maintained pace";
    }

    it("interprets speed adaptation correctly", () => {
      expect(interpretSpeedAdaptation(0.5)).toBe("significantly sped up");
      expect(interpretSpeedAdaptation(0.8)).toBe("moderately adapted");
      expect(interpretSpeedAdaptation(1.0)).toBe("maintained pace");
      expect(interpretSpeedAdaptation(1.2)).toBe("slowed down");
    });
  });
});

