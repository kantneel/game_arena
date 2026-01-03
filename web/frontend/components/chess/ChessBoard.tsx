"use client";

import { useState, useEffect } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";

interface ChessBoardProps {
  fen?: string;
  lastMove?: string;
  onMove?: (move: string) => void;
  interactive?: boolean;
  orientation?: "white" | "black";
  size?: number;
}

export function ChessBoardComponent({
  fen = "start",
  lastMove,
  onMove,
  interactive = false,
  orientation = "white",
  size = 400,
}: ChessBoardProps) {
  const [game] = useState(new Chess());
  const [position, setPosition] = useState(fen);

  useEffect(() => {
    if (fen === "start") {
      game.reset();
    } else {
      try {
        game.load(fen);
      } catch (e) {
        console.error("Invalid FEN:", fen);
      }
    }
    setPosition(game.fen());
  }, [fen, game]);

  // Highlight last move
  const customSquareStyles: Record<string, React.CSSProperties> = {};
  if (lastMove && lastMove.length >= 4) {
    const from = lastMove.slice(0, 2);
    const to = lastMove.slice(2, 4);
    customSquareStyles[from] = {
      backgroundColor: "rgba(255, 255, 0, 0.3)",
    };
    customSquareStyles[to] = {
      backgroundColor: "rgba(255, 255, 0, 0.4)",
    };
  }

  return (
    <div className="rounded-lg overflow-hidden shadow-2xl">
      <Chessboard
        position={position}
        boardOrientation={orientation}
        boardWidth={size}
        arePiecesDraggable={interactive}
        customSquareStyles={customSquareStyles}
        customDarkSquareStyle={{ backgroundColor: "#4a5568" }}
        customLightSquareStyle={{ backgroundColor: "#a0aec0" }}
      />
    </div>
  );
}

