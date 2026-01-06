import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AdaptationMetrics } from "@/components/charts/AdaptationMetrics";
import { ModelPressureProfile } from "@/lib/api";

describe("AdaptationMetrics", () => {
  const modelA: ModelPressureProfile = {
    name: "claude-sonnet-4.5",
    total_moves: 100,
    speed_adaptation_ratio: 0.6,
    quality_degradation_ratio: 1.2,
    thinking_reduction_ratio: 0.4,
    pressure_stats: [],
  };

  const modelB: ModelPressureProfile = {
    name: "gemini-3-flash",
    total_moves: 100,
    speed_adaptation_ratio: 0.95,
    quality_degradation_ratio: 2.5,
    thinking_reduction_ratio: 0.9,
    pressure_stats: [],
  };

  it("renders all metrics", () => {
    render(<AdaptationMetrics modelA={modelA} modelB={modelB} />);

    expect(screen.getByText("Speed Adaptation")).toBeInTheDocument();
    expect(screen.getByText("Quality Preservation")).toBeInTheDocument();
    expect(screen.getByText("Thinking Reduction")).toBeInTheDocument();
  });

  it("displays model names", () => {
    render(<AdaptationMetrics modelA={modelA} modelB={modelB} />);

    expect(screen.getAllByText("claude-sonnet-4.5").length).toBeGreaterThan(0);
    expect(screen.getAllByText("gemini-3-flash").length).toBeGreaterThan(0);
  });

  it("displays percentage values", () => {
    render(<AdaptationMetrics modelA={modelA} modelB={modelB} />);

    // Speed adaptation: 60% vs 95%
    expect(screen.getByText("60%")).toBeInTheDocument();
    expect(screen.getByText("95%")).toBeInTheDocument();
  });

  it("highlights better performer for speed adaptation", () => {
    render(<AdaptationMetrics modelA={modelA} modelB={modelB} />);

    // Model A has lower speed ratio (0.6 < 0.95), so it should be highlighted
    const modelASpeedCard = screen.getByText("60%").closest("div");
    expect(modelASpeedCard).toHaveClass("bg-arena-win/10");
  });
});

