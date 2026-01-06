import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

// Need to mock next/navigation before importing Navigation
vi.mock("next/navigation", () => ({
  usePathname: () => "/",
}));

import { Navigation } from "@/components/ui/Navigation";

describe("Navigation", () => {
  it("renders the logo", () => {
    render(<Navigation />);

    expect(screen.getByText("Game Arena")).toBeInTheDocument();
  });

  it("renders all navigation links", () => {
    render(<Navigation />);

    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getByText("Live")).toBeInTheDocument();
    expect(screen.getByText("Matches")).toBeInTheDocument();
    expect(screen.getByText("Models")).toBeInTheDocument();
    expect(screen.getByText("Rankings")).toBeInTheDocument();
    expect(screen.getByText("New Match")).toBeInTheDocument();
  });

  it("shows live badge on Live link", () => {
    render(<Navigation />);

    // The Live link should have a pulsing badge (w-2 h-2 bg-red-500)
    const liveLink = screen.getByText("Live").closest("a");
    const badge = liveLink?.querySelector(".animate-pulse");
    expect(badge).toBeInTheDocument();
  });

  it("highlights New Match link", () => {
    render(<Navigation />);

    const newMatchLink = screen.getByText("New Match").closest("a");
    expect(newMatchLink).toHaveClass("border");
  });
});

