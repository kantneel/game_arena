"use client";

import { useState } from "react";

type PromptStyle = "time_info_only" | "standard" | "dramatic" | "none";
type TimeLevel = 300 | 90 | 30 | 10;
type NotationFormat = "san" | "lan" | "pgn";

const TIME_LEVELS: TimeLevel[] = [300, 90, 30, 10];

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins > 0) {
    return secs > 0 ? `${mins}:${secs.toString().padStart(2, '0')}` : `${mins}:00`;
  }
  return `0:${secs.toString().padStart(2, '0')}`;
}

function getPressureLevel(seconds: number): string {
  if (seconds < 60) return "🔴 HIGH";
  if (seconds < 120) return "🟡 MEDIUM";
  return "🟢 LOW";
}

// Example move histories in different formats
const MOVE_HISTORY_SAN = `Move history: 1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6`;
const MOVE_HISTORY_LAN = `Move history: 1. e2e4 e7e5 2. g1f3 b8c6 3. f1c4 g8f6`;
const MOVE_HISTORY_PGN = `[Event "Blitz Match"]
[Site "Game Arena"]
[Date "2026.01.05"]
[Round "1"]
[White "gemini-3-flash"]
[Black "gemini-2.5-pro"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 *`;

// Example chess board in ASCII
const EXAMPLE_BOARD = `Current position (FEN: r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4):

8 | r . b q k b . r
7 | p p p p . p p p
6 | . . n . . n . .
5 | . . . . p . . .
4 | . . B . P . . .
3 | . . . . . N . .
2 | P P P P . P P P
1 | R N B Q K . . R
  -----------------
    a b c d e f g h

It is White's turn.`;

const RESPONSE_FEEDBACK_EXAMPLE = `
📊 YOUR PREVIOUS RESPONSE ANALYSIS:
• Your last move took 12.3 seconds
• You used 2,847 thinking tokens
• Your thinking speed: ~231 tokens/second
`;

// Instruction texts based on time
function getInstruction(timeRemaining: number, style: PromptStyle): string {
  if (style === "dramatic") {
    if (timeRemaining < 30) {
      return `MOVE NOW!!! Minimal or no reasoning - just output your move immediately! Time is critical!!!

Output your final answer in the format "Final Answer: X" where X is your chosen move.`;
    } else if (timeRemaining < 60) {
      return `Very brief reasoning only! Decide within a few seconds!

Output your final answer in the format "Final Answer: X" where X is your chosen move.`;
    } else if (timeRemaining < 120) {
      return `Be efficient - reason only as much as needed for this position, then move!

Output your final answer in the format "Final Answer: X" where X is your chosen move.`;
    }
  }
  
  // Default/standard instruction - flexible reasoning
  return `Reason as much as you think is necessary for this position (could be extensive analysis or none at all depending on time pressure and position complexity), then output your final answer in the format "Final Answer: X" where X is your chosen move in algebraic notation.

What is your move?`;
}

export default function PromptsPage() {
  const [selectedStyle, setSelectedStyle] = useState<PromptStyle>("standard");
  const [selectedTime, setSelectedTime] = useState<TimeLevel>(30);
  const [selectedNotation, setSelectedNotation] = useState<NotationFormat>("san");
  const [showBoard, setShowBoard] = useState(true);
  const [showFeedback, setShowFeedback] = useState(true);
  const [showMoveHistory, setShowMoveHistory] = useState(true);

  const getMoveHistory = () => {
    switch (selectedNotation) {
      case "lan": return MOVE_HISTORY_LAN;
      case "pgn": return MOVE_HISTORY_PGN;
      default: return MOVE_HISTORY_SAN;
    }
  };

  const generatePrompt = () => {
    const parts: string[] = [];
    
    // Move history (before board for PGN context)
    if (showMoveHistory) {
      parts.push(getMoveHistory());
      parts.push("");
    }
    
    // Board
    if (showBoard) {
      parts.push(EXAMPLE_BOARD);
      parts.push("");
    }
    
    // Time info based on style
    if (selectedStyle === "none") {
      // No time info
    } else if (selectedStyle === "time_info_only") {
      parts.push(`Chess Clock Status:
- Your time: ${formatTime(selectedTime)}
- Opponent's time: ${formatTime(180)}
- Increment: +2s per move`);
    } else if (selectedStyle === "dramatic" && selectedTime < 30) {
      parts.push(`🚨🚨🚨 CRITICAL TIME EMERGENCY!!! 🚨🚨🚨
⏰ YOUR TIME: ${formatTime(selectedTime)} - MOVE FAST OR LOSE!!!
⏰ Opponent: ${formatTime(180)}
⏰ Increment: +2s per move

🔴🔴🔴 YOUR CLOCK IS CRITICALLY LOW - MAKE A MOVE IMMEDIATELY!!! 🔴🔴🔴
EVERY SECOND OF REASONING BRINGS YOU CLOSER TO DEFEAT!!!`);
    } else if (selectedStyle === "dramatic" && selectedTime < 60) {
      parts.push(`⚠️⚠️ URGENT: TIME IS RUNNING OUT! ⚠️⚠️
⏰ YOUR TIME: ${formatTime(selectedTime)}
⏰ Opponent: ${formatTime(180)}
⏰ Increment: +2s per move

🟠 HURRY! Your clock is dangerously low!`);
    } else if (selectedStyle === "dramatic") {
      parts.push(`BLITZ CHESS - TIME IS PRECIOUS!
⏰ Your remaining time: ${formatTime(selectedTime)}
⏰ Opponent's remaining time: ${formatTime(180)}
⏰ Time increment per move: +2 seconds

Remember: Your thinking time directly consumes your clock!`);
    } else {
      // Standard
      parts.push(`BLITZ CHESS TIME INFORMATION:
⏰ Your remaining time: ${formatTime(selectedTime)}
⏰ Opponent's remaining time: ${formatTime(180)}
⏰ Time increment per move: +2 seconds

⚠️  CRITICAL TIME RULES:
- This is REAL WALL CLOCK TIME - your thinking/reasoning time directly consumes your clock
- You lose immediately if your time runs out (time forfeit)
- Longer reasoning traces = more time consumed = higher risk of time forfeit
- You must balance move quality vs. time management
- Each move adds 2 seconds to your clock after you play it
- Consider quick, good moves over perfect moves that consume too much time

Current time pressure level: ${getPressureLevel(selectedTime)}`);
    }
    
    // Response feedback
    if (showFeedback && selectedStyle !== "none") {
      parts.push("");
      parts.push(RESPONSE_FEEDBACK_EXAMPLE.trim());
    }
    
    // Instructions
    parts.push("");
    parts.push(getInstruction(selectedTime, selectedStyle));
    
    return parts.join("\n");
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <div className="border-b border-zinc-800 bg-gradient-to-r from-zinc-900 to-zinc-950">
        <div className="container mx-auto px-6 py-8">
          <h1 className="text-3xl font-bold mb-2">Prompt Reference</h1>
          <p className="text-zinc-400">See exactly what prompts are sent to the models during evaluation</p>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Controls */}
          <div className="space-y-6">
            {/* Move Notation Format */}
            <section className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
              <h2 className="text-lg font-semibold mb-4">Move Notation Format</h2>
              <div className="space-y-2">
                {[
                  { value: "san", label: "SAN", desc: "Standard Algebraic (e4, Nf3, O-O)" },
                  { value: "lan", label: "LAN", desc: "Long Algebraic (e2e4, g1f3)" },
                  { value: "pgn", label: "PGN", desc: "Full PGN with headers" },
                ].map((notation) => (
                  <button
                    key={notation.value}
                    onClick={() => setSelectedNotation(notation.value as NotationFormat)}
                    className={`w-full text-left p-3 rounded-lg transition-all ${
                      selectedNotation === notation.value
                        ? "bg-emerald-600/20 border-2 border-emerald-500"
                        : "bg-zinc-800 border-2 border-transparent hover:border-zinc-700"
                    }`}
                  >
                    <div className="font-medium font-mono">{notation.label}</div>
                    <div className="text-sm text-zinc-400">{notation.desc}</div>
                  </button>
                ))}
              </div>
            </section>

            {/* Prompt Style */}
            <section className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
              <h2 className="text-lg font-semibold mb-4">Prompt Style</h2>
              <div className="space-y-2">
                {[
                  { value: "none", label: "None", desc: "No time information (baseline)" },
                  { value: "time_info_only", label: "Time Info Only", desc: "Just the clock values" },
                  { value: "standard", label: "Standard", desc: "Clock + urgency guidance" },
                  { value: "dramatic", label: "Dramatic", desc: "ALL-CAPS urgency (time-adaptive)" },
                ].map((style) => (
                  <button
                    key={style.value}
                    onClick={() => setSelectedStyle(style.value as PromptStyle)}
                    className={`w-full text-left p-3 rounded-lg transition-all ${
                      selectedStyle === style.value
                        ? "bg-cyan-600/20 border-2 border-cyan-500"
                        : "bg-zinc-800 border-2 border-transparent hover:border-zinc-700"
                    }`}
                  >
                    <div className="font-medium">{style.label}</div>
                    <div className="text-sm text-zinc-400">{style.desc}</div>
                  </button>
                ))}
              </div>
            </section>

            {/* Time Level */}
            <section className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
              <h2 className="text-lg font-semibold mb-4">Time Remaining</h2>
              <div className="grid grid-cols-2 gap-2">
                {TIME_LEVELS.map((time) => (
                  <button
                    key={time}
                    onClick={() => setSelectedTime(time)}
                    className={`p-3 rounded-lg font-mono transition-all ${
                      selectedTime === time
                        ? "bg-purple-600/20 border-2 border-purple-500"
                        : "bg-zinc-800 border-2 border-transparent hover:border-zinc-700"
                    }`}
                  >
                    <div className="text-lg">{formatTime(time)}</div>
                    <div className="text-xs text-zinc-400">
                      {time >= 120 ? "Comfortable" : time >= 60 ? "Medium" : time >= 30 ? "Low" : "Critical"}
                    </div>
                  </button>
                ))}
              </div>
            </section>

            {/* Options */}
            <section className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
              <h2 className="text-lg font-semibold mb-4">Options</h2>
              <div className="space-y-3">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showMoveHistory}
                    onChange={(e) => setShowMoveHistory(e.target.checked)}
                    className="w-5 h-5 rounded bg-zinc-800 border-zinc-600 text-cyan-500 focus:ring-cyan-500"
                  />
                  <div>
                    <div className="font-medium">Show Move History</div>
                    <div className="text-sm text-zinc-400">Previous moves in selected notation</div>
                  </div>
                </label>
                
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showBoard}
                    onChange={(e) => setShowBoard(e.target.checked)}
                    className="w-5 h-5 rounded bg-zinc-800 border-zinc-600 text-cyan-500 focus:ring-cyan-500"
                  />
                  <div>
                    <div className="font-medium">Show Chess Board</div>
                    <div className="text-sm text-zinc-400">ASCII representation of position</div>
                  </div>
                </label>
                
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showFeedback}
                    onChange={(e) => setShowFeedback(e.target.checked)}
                    className="w-5 h-5 rounded bg-zinc-800 border-zinc-600 text-cyan-500 focus:ring-cyan-500"
                  />
                  <div>
                    <div className="font-medium">Response Feedback</div>
                    <div className="text-sm text-zinc-400">Previous response analysis (online evals)</div>
                  </div>
                </label>
              </div>
            </section>

            {/* Key Changes */}
            <section className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
              <h2 className="text-lg font-semibold mb-4">Recent Changes</h2>
              <div className="text-sm text-zinc-400 space-y-2">
                <div className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>PGN Notation</strong> - Full game context with headers</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Flexible Reasoning</strong> - No longer biased toward step-by-step</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Response Feedback</strong> - Models see their previous time/tokens</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Time-Adaptive Instructions</strong> - Shorter prompts when time is low</span>
                </div>
              </div>
            </section>
          </div>

          {/* Prompt Preview */}
          <div className="lg:col-span-2">
            <div className="bg-zinc-900 rounded-xl border border-zinc-800 sticky top-20">
              <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
                <h2 className="text-lg font-semibold">Generated Prompt</h2>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-zinc-400">
                    Notation: <span className="text-emerald-400 font-mono">{selectedNotation.toUpperCase()}</span>
                  </span>
                  <span className="text-zinc-400">
                    Style: <span className="text-cyan-400">{selectedStyle}</span>
                  </span>
                  <span className="text-zinc-400">
                    Time: <span className="text-purple-400">{formatTime(selectedTime)}</span>
                  </span>
                </div>
              </div>
              
              <div className="p-6">
                <pre className="bg-zinc-950 rounded-lg p-6 overflow-x-auto text-sm font-mono whitespace-pre-wrap leading-relaxed border border-zinc-800">
                  {generatePrompt()}
                </pre>
              </div>
              
              <div className="px-6 py-4 border-t border-zinc-800 flex justify-between items-center">
                <div className="text-sm text-zinc-500">
                  ~{generatePrompt().length} characters • ~{Math.round(generatePrompt().split(/\s+/).length)} words
                </div>
                <button
                  onClick={() => navigator.clipboard.writeText(generatePrompt())}
                  className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-sm transition-colors"
                >
                  Copy to Clipboard
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
