#!/usr/bin/env python3
"""Offline evaluator for controlled time pressure experiments.

This module runs models on fixed positions with varying time constraints
to measure how time pressure affects move quality and thinking depth.
"""

import datetime
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
from typing import Optional

# Try to import tqdm for progress bars, fall back to simple progress
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from game_arena.harness import tournament_util
from game_arena.blitz.models.registry import get_model_from_registry
from game_arena.blitz.models.wrappers import NoRetryModelWrapper
from game_arena.blitz.display.formatting import format_time
from game_arena.blitz.offline_eval.position_dataset import ChessPosition, PositionDataset


class PromptStyle:
    """Enum-like class for prompt styles."""
    NONE = "none"  # No time information at all
    TIME_INFO_ONLY = "time_info_only"  # Just the clock values, no urgency
    STANDARD = "standard"  # Time info + urgency warnings
    DRAMATIC = "dramatic"  # ALL-CAPS dramatic pressure


@dataclass
class EvaluationConfig:
    """Configuration for offline evaluation."""
    
    # Time pressure levels to test (seconds remaining)
    time_levels: list[float] = field(default_factory=lambda: [300, 120, 60, 30, 15])
    
    # Number of samples per (position, time_level) pair
    samples_per_condition: int = 3
    
    # Opponent time (constant)
    opponent_time: float = 180.0
    
    # Time increment
    increment: int = 3
    
    # Model reasoning budget
    reasoning_budget: int = 8000
    
    # Prompt style (replaces old boolean flags)
    prompt_style: str = PromptStyle.STANDARD  # "none", "time_info_only", "standard", "dramatic"
    
    # Legacy flags (for backwards compatibility, derived from prompt_style)
    @property
    def enable_time_pressure_prompt(self) -> bool:
        return self.prompt_style != PromptStyle.NONE
    
    @property
    def use_dramatic_prompts(self) -> bool:
        return self.prompt_style == PromptStyle.DRAMATIC
    
    # Response feedback (separate from prompt style)
    enable_response_feedback: bool = False
    
    # Timeout simulation - treat response as forfeit if it exceeds time_remaining
    simulate_clock: bool = True
    
    # Latency calibration
    calibrate_latency: bool = True
    calibration_samples: int = 3
    
    # Stockfish settings for move quality analysis
    stockfish_depth: int = 15


@dataclass
class EvaluationResult:
    """Result from evaluating a single (position, time_level, sample) condition."""
    
    # Identifiers
    position_id: str
    model_id: str
    time_remaining: float
    sample_index: int
    
    # Model outputs
    move_played: str
    thinking_tokens: int
    output_tokens: int
    response_time_seconds: float
    full_response: str
    
    # Timeout/forfeit tracking
    would_timeout: bool = False  # True if response_time > time_remaining
    time_after_move: float = 0.0  # Simulated clock after move (with increment)
    
    # Quality metrics (computed post-hoc with Stockfish)
    centipawn_loss: Optional[float] = None
    is_best_move: bool = False
    is_blunder: bool = False  # >100 cp loss
    move_rank: Optional[int] = None  # Rank among top N moves
    
    # Metadata
    timestamp: str = ""
    config_hash: str = ""
    prompt_style: str = ""  # Track which prompt style was used
    network_latency: float = 0.0  # Calibrated network latency


@dataclass
class EvaluationSession:
    """A complete evaluation session."""
    
    session_id: str
    model_id: str
    dataset_name: str
    config: EvaluationConfig
    results: list[EvaluationResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    status: str = "in_progress"  # "in_progress", "completed", "interrupted"
    
    def add_result(self, result: EvaluationResult):
        self.results.append(result)
    
    def save(self, path: Path):
        """Save session to JSON."""
        data = {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "dataset_name": self.dataset_name,
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "EvaluationSession":
        """Load session from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        config = EvaluationConfig(**data["config"])
        results = [EvaluationResult(**r) for r in data["results"]]
        
        return cls(
            session_id=data["session_id"],
            model_id=data["model_id"],
            dataset_name=data["dataset_name"],
            config=config,
            results=results,
            start_time=data["start_time"],
            end_time=data.get("end_time", ""),
            status=data.get("status", "completed"),
        )
    
    def get_completion_stats(self) -> dict:
        """Get statistics about session completion."""
        total_expected = (
            len(set(r.position_id for r in self.results)) or 1
        ) * len(self.config.time_levels) * self.config.samples_per_condition
        
        return {
            "completed": len(self.results),
            "expected": total_expected,
            "completion_pct": len(self.results) / total_expected * 100 if total_expected > 0 else 0,
            "status": self.status,
        }


class ProgressTracker:
    """Tracks progress with ETA estimation."""
    
    def __init__(self, total: int, desc: str = "Evaluating"):
        self.total = total
        self.completed = 0
        self.desc = desc
        self.start_time = time.time()
        self.times = []  # Track individual task times for ETA
        self._lock = Lock()
        
        # Use tqdm if available
        if HAS_TQDM:
            self.pbar = tqdm(total=total, desc=desc, unit="eval", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            self.pbar = None
    
    def update(self, task_time: float = None):
        """Update progress by 1."""
        with self._lock:
            self.completed += 1
            if task_time:
                self.times.append(task_time)
            
            if self.pbar:
                self.pbar.update(1)
            else:
                self._print_progress()
    
    def _print_progress(self):
        """Print simple progress without tqdm."""
        elapsed = time.time() - self.start_time
        pct = self.completed / self.total * 100
        
        # Calculate ETA
        if self.completed > 0:
            avg_time = elapsed / self.completed
            remaining = (self.total - self.completed) * avg_time
            eta_str = self._format_time(remaining)
        else:
            eta_str = "calculating..."
        
        # Clear line and print
        sys.stdout.write(f"\r{self.desc}: {self.completed}/{self.total} ({pct:.1f}%) | ETA: {eta_str}    ")
        sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def close(self):
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()
        else:
            elapsed = time.time() - self.start_time
            print(f"\n✅ Completed {self.completed}/{self.total} in {self._format_time(elapsed)}")
    
    def get_stats(self) -> dict:
        """Get progress statistics."""
        elapsed = time.time() - self.start_time
        return {
            "completed": self.completed,
            "total": self.total,
            "elapsed_seconds": elapsed,
            "avg_time_per_eval": elapsed / self.completed if self.completed > 0 else 0,
            "estimated_remaining": (self.total - self.completed) * (elapsed / self.completed) if self.completed > 0 else 0,
        }


class OfflineEvaluator:
    """Evaluates models on fixed positions with varying time constraints."""
    
    def __init__(
        self,
        model_id: str,
        config: Optional[EvaluationConfig] = None,
        output_dir: Optional[Path] = None,
        max_workers: int = 4,
        save_interval: int = 10,  # Save partial results every N completions
    ):
        self.model_id = model_id
        self.config = config or EvaluationConfig()
        self.output_dir = output_dir or Path("_results/offline_eval")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.save_interval = save_interval
        
        # Initialize model with NoRetryModelWrapper for consistency with online matches
        print(f"Loading model: {model_id}")
        raw_model = get_model_from_registry(model_id)
        
        # Apply reasoning budget
        if hasattr(raw_model, '_model_options'):
            if raw_model._model_options is None:
                raw_model._model_options = {}
            raw_model._model_options['thinking_budget'] = self.config.reasoning_budget
        
        # Wrap with NoRetryModelWrapper for consistent API with blitz matches
        self.model = NoRetryModelWrapper(raw_model)
        
        # Stockfish for move quality (lazy loaded)
        self._stockfish = None
        
        # Thread safety for results collection
        self._results_lock = Lock()
        self._save_counter = 0
        
        # Network latency (calibrated before evaluation)
        self._network_latency = 0.0
    
    def _calibrate_latency(self) -> float:
        """Calibrate network latency with warmup calls (same as blitz matches)."""
        print("🌐 Calibrating network latency...")
        
        # Simple prompt for calibration
        calibration_prompt = "What is 2+2? Answer with just the number."
        model_input = tournament_util.ModelTextInput(prompt_text=calibration_prompt)
        
        latencies = []
        for i in range(self.config.calibration_samples):
            start = time.time()
            try:
                self.model.generate_with_text_input(model_input)
                latency = time.time() - start
                latencies.append(latency)
                print(f"   Calibration {i+1}/{self.config.calibration_samples}: {latency:.2f}s")
            except Exception as e:
                print(f"   Calibration {i+1} failed: {e}")
        
        if latencies:
            # Use median for robustness
            latencies.sort()
            median_latency = latencies[len(latencies) // 2]
            print(f"   📊 Calibrated latency: {median_latency:.2f}s")
            return median_latency
        else:
            print("   ⚠️ Calibration failed, using 0s latency")
            return 0.0
    
    def evaluate_dataset(
        self,
        dataset: PositionDataset,
        progress_callback=None,
        parallel: bool = True,
    ) -> EvaluationSession:
        """Evaluate all positions in dataset across all time levels.
        
        Args:
            dataset: Position dataset to evaluate
            progress_callback: Optional callback(completed, total) for progress
            parallel: If True, run evaluations in parallel (default: True)
            
        Returns:
            EvaluationSession with all results
        """
        # Calibrate latency first if enabled
        if self.config.calibrate_latency:
            self._network_latency = self._calibrate_latency()
        
        session_id = f"{self.model_id}_{self.config.prompt_style}_{dataset.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = EvaluationSession(
            session_id=session_id,
            model_id=self.model_id,
            dataset_name=dataset.name,
            config=self.config,
            start_time=datetime.datetime.now().isoformat(),
        )
        
        total_evals = (
            len(dataset.positions)
            * len(self.config.time_levels)
            * self.config.samples_per_condition
        )
        
        print(f"\n{'='*60}")
        print(f"Starting offline evaluation: {session_id}")
        print(f"Model: {self.model_id}")
        print(f"Prompt style: {self.config.prompt_style}")
        print(f"Positions: {len(dataset.positions)}")
        print(f"Time levels: {self.config.time_levels}")
        print(f"Samples per condition: {self.config.samples_per_condition}")
        print(f"Total evaluations: {total_evals}")
        print(f"Parallel workers: {self.max_workers if parallel else 1}")
        print(f"Network latency: {self._network_latency:.2f}s")
        print(f"Simulate clock/timeouts: {self.config.simulate_clock}")
        print(f"{'='*60}\n")
        
        # Store session path for incremental saving
        self._session_path = self.output_dir / f"{session_id}.json"
        self._current_session = session
        
        try:
            if parallel:
                session = self._evaluate_parallel(dataset, session, total_evals, progress_callback)
            else:
                session = self._evaluate_sequential(dataset, session, total_evals, progress_callback)
            
            session.status = "completed"
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted! Saving partial results...")
            session.status = "interrupted"
        
        session.end_time = datetime.datetime.now().isoformat()
        
        # Print timeout summary
        if session.results:
            timeouts = sum(1 for r in session.results if r.would_timeout)
            print(f"\n⏰ Timeout summary: {timeouts}/{len(session.results)} ({100*timeouts/len(session.results):.1f}%) would have timed out")
        
        # Final save
        session.save(self._session_path)
        print(f"\n✅ Session saved: {self._session_path}")
        print(f"   Status: {session.status} ({len(session.results)}/{total_evals} evaluations)")
        
        return session
    
    def _save_incremental(self, session: EvaluationSession):
        """Save session incrementally for partial results."""
        with self._results_lock:
            self._save_counter += 1
            if self._save_counter % self.save_interval == 0:
                session.save(self._session_path)
                # Also save a backup
                backup_path = self._session_path.with_suffix('.partial.json')
                session.save(backup_path)
    
    def _evaluate_sequential(
        self,
        dataset: PositionDataset,
        session: EvaluationSession,
        total_evals: int,
        progress_callback=None,
    ) -> EvaluationSession:
        """Run evaluations sequentially with progress tracking."""
        progress = ProgressTracker(total_evals, desc=f"Evaluating {self.model_id}")
        
        try:
            for position in dataset.positions:
                for time_level in self.config.time_levels:
                    for sample_idx in range(self.config.samples_per_condition):
                        start = time.time()
                        try:
                            result = self._evaluate_single(
                                position, time_level, sample_idx
                            )
                            session.add_result(result)
                            self._save_incremental(session)
                            
                        except Exception as e:
                            print(f"\n  ❌ Error {position.position_id} @ {time_level}s: {e}")
                        
                        task_time = time.time() - start
                        progress.update(task_time)
                        
                        if progress_callback:
                            progress_callback(progress.completed, total_evals)
        finally:
            progress.close()
        
        return session
    
    def _evaluate_parallel(
        self,
        dataset: PositionDataset,
        session: EvaluationSession,
        total_evals: int,
        progress_callback=None,
    ) -> EvaluationSession:
        """Run evaluations in parallel using ThreadPoolExecutor."""
        # Build list of all evaluation tasks
        tasks = []
        for position in dataset.positions:
            for time_level in self.config.time_levels:
                for sample_idx in range(self.config.samples_per_condition):
                    tasks.append((position, time_level, sample_idx))
        
        print(f"🚀 Submitting {len(tasks)} tasks to {self.max_workers} parallel workers...\n")
        
        progress = ProgressTracker(total_evals, desc=f"Evaluating {self.model_id}")
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._evaluate_single_safe, pos, time_lvl, sample_idx): (pos, time_lvl, sample_idx)
                    for pos, time_lvl, sample_idx in tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_task):
                    pos, time_lvl, sample_idx = future_to_task[future]
                    
                    try:
                        result = future.result()
                        if result:
                            with self._results_lock:
                                session.add_result(result)
                            self._save_incremental(session)
                            
                    except Exception as e:
                        # Error already logged in _evaluate_single_safe
                        pass
                    
                    progress.update()
                    
                    if progress_callback:
                        progress_callback(progress.completed, total_evals)
        finally:
            progress.close()
        
        return session
    
    def _evaluate_single_safe(
        self,
        position: ChessPosition,
        time_remaining: float,
        sample_index: int,
    ) -> Optional[EvaluationResult]:
        """Thread-safe wrapper for _evaluate_single."""
        try:
            return self._evaluate_single(position, time_remaining, sample_index)
        except Exception as e:
            print(f"  ⚠️ Exception in evaluation: {e}")
            return None
    
    def _evaluate_single(
        self,
        position: ChessPosition,
        time_remaining: float,
        sample_index: int,
    ) -> EvaluationResult:
        """Evaluate a single (position, time_level) condition."""
        
        # Build prompt directly without pyspiel (simpler for offline eval)
        prompt_text = self._build_prompt(position, time_remaining)
        
        # Create model input
        model_input = tournament_util.ModelTextInput(prompt_text=prompt_text)
        
        # Call model via NoRetryModelWrapper (returns tuple like in blitz matches)
        start_time = time.time()
        response, retry_count, total_retry_time = self.model.generate_with_text_input(model_input)
        response_time = time.time() - start_time
        
        # Subtract retry time from response time (same as blitz matches)
        actual_thinking_time = response_time - total_retry_time
        
        # Subtract network latency from thinking time (same as blitz matches)
        thinking_time_minus_latency = max(0, actual_thinking_time - self._network_latency)
        
        # Extract move from response
        move_played = self._extract_move(response.main_response)
        
        # Get token counts
        thinking_tokens = getattr(response, 'reasoning_tokens', 0) or 0
        output_tokens = getattr(response, 'generation_tokens', 0) or 0
        
        # Compute timeout status
        would_timeout = False
        time_after_move = time_remaining
        if self.config.simulate_clock:
            # Would this response cause a timeout?
            would_timeout = thinking_time_minus_latency > time_remaining
            # What would the clock show after this move (with increment)?
            time_after_move = time_remaining - thinking_time_minus_latency + self.config.increment
        
        # Create result
        result = EvaluationResult(
            position_id=position.position_id,
            model_id=self.model_id,
            time_remaining=time_remaining,
            sample_index=sample_index,
            move_played=move_played,
            thinking_tokens=thinking_tokens,
            output_tokens=output_tokens,
            response_time_seconds=thinking_time_minus_latency,  # Use thinking time minus latency
            full_response=response.main_response[:1000],  # Truncate for storage
            would_timeout=would_timeout,
            time_after_move=time_after_move,
            timestamp=datetime.datetime.now().isoformat(),
            prompt_style=self.config.prompt_style,
            network_latency=self._network_latency,
        )
        
        # Compute move quality if we have ground truth
        if position.best_move:
            result.is_best_move = (move_played.lower() == position.best_move.lower())
        
        return result
    
    def _build_prompt(self, position: ChessPosition, time_remaining: float) -> str:
        """Build a complete prompt for evaluating a position.
        
        Prompt styles:
        - NONE: No time information at all (baseline)
        - TIME_INFO_ONLY: Just the clock values, neutral language
        - STANDARD: Time info + urgency warnings
        - DRAMATIC: ALL-CAPS dramatic pressure
        """
        
        # Convert FEN to readable board
        board_str = self._fen_to_readable(position.fen)
        
        # Determine side to move from FEN
        parts = position.fen.split()
        side_to_move = "White" if len(parts) < 2 or parts[1] == "w" else "Black"
        
        # Build time info based on prompt style
        style = self.config.prompt_style
        
        if style == PromptStyle.NONE:
            # No time information at all - pure chess analysis
            time_info = ""
            
        elif style == PromptStyle.TIME_INFO_ONLY:
            # Just the factual clock values, no urgency or warnings
            time_info = f"""
Chess Clock Status:
- Your time: {format_time(time_remaining)}
- Opponent's time: {format_time(self.config.opponent_time)}
- Increment: +{self.config.increment}s per move
"""
            
        elif style == PromptStyle.DRAMATIC:
            # Dramatic ALL-CAPS urgency
            if time_remaining < 30:
                time_info = f"""
🚨🚨🚨 CRITICAL TIME EMERGENCY!!! 🚨🚨🚨
⏰ YOUR TIME: {format_time(time_remaining)} - MOVE FAST OR LOSE!!!
⏰ Opponent: {format_time(self.config.opponent_time)}
⏰ Increment: +{self.config.increment}s per move

🔴🔴🔴 YOUR CLOCK IS CRITICALLY LOW - MAKE A MOVE IMMEDIATELY!!! 🔴🔴🔴
EVERY SECOND OF REASONING BRINGS YOU CLOSER TO DEFEAT!!!
"""
            elif time_remaining < 60:
                time_info = f"""
⚠️⚠️ URGENT: TIME IS RUNNING OUT! ⚠️⚠️
⏰ YOUR TIME: {format_time(time_remaining)}
⏰ Opponent: {format_time(self.config.opponent_time)}
⏰ Increment: +{self.config.increment}s per move

🟠 HURRY! Your clock is dangerously low!
"""
            else:
                time_info = f"""
BLITZ CHESS - TIME IS PRECIOUS!
⏰ Your remaining time: {format_time(time_remaining)}
⏰ Opponent's remaining time: {format_time(self.config.opponent_time)}
⏰ Time increment per move: +{self.config.increment} seconds

Remember: Your thinking time directly consumes your clock!
"""
            
        else:  # STANDARD (default)
            pressure = "🔴 HIGH" if time_remaining < 60 else "🟡 MEDIUM" if time_remaining < 120 else "🟢 LOW"
            time_info = f"""
BLITZ CHESS TIME INFORMATION:
⏰ Your remaining time: {format_time(time_remaining)}
⏰ Opponent's remaining time: {format_time(self.config.opponent_time)}
⏰ Time increment per move: +{self.config.increment} seconds

⚠️  CRITICAL TIME RULES:
- This is REAL WALL CLOCK TIME - your thinking/reasoning time directly consumes your clock
- You lose immediately if your time runs out (time forfeit)
- Longer reasoning traces = more time consumed = higher risk of time forfeit
- You must balance move quality vs. time management

Current time pressure level: {pressure}
"""
        
        prompt = f"""You are playing chess as {side_to_move}.

{board_str}
{time_info}
Analyze the position and determine the best move. Reason step by step, then output your final answer in the format "Final Answer: X" where X is your chosen move in algebraic notation (e.g., "Final Answer: e4" or "Final Answer: Nf3").

What is your move?"""
        
        return prompt
    
    def _fen_to_readable(self, fen: str) -> str:
        """Convert FEN to readable board string."""
        parts = fen.split()
        board_part = parts[0]
        
        # Convert FEN to visual board
        rows = board_part.split("/")
        board_lines = []
        
        for rank_idx, row in enumerate(rows):
            rank_num = 8 - rank_idx
            line = f"{rank_num} |"
            for char in row:
                if char.isdigit():
                    line += " ." * int(char)
                else:
                    line += f" {char}"
            board_lines.append(line)
        
        board_lines.append("   ----------------")
        board_lines.append("    a b c d e f g h")
        
        # Add side to move
        side = "White" if len(parts) < 2 or parts[1] == "w" else "Black"
        board_lines.insert(0, f"Position (FEN): {fen}")
        board_lines.insert(1, f"Side to move: {side}")
        board_lines.insert(2, "")
        
        return "\n".join(board_lines)
    
    def _extract_move(self, response: str) -> str:
        """Extract chess move from model response."""
        import re
        
        # Look for "Final Answer: X" pattern
        match = re.search(r"Final Answer:\s*([A-Za-z0-9\-\+\#\=]+)", response)
        if match:
            return match.group(1)
        
        # Look for common move patterns
        move_patterns = [
            r"\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+\#]?)\b",
            r"\b(O-O-O|O-O)\b",
        ]
        
        for pattern in move_patterns:
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1]  # Return last match (usually the final answer)
        
        return "unknown"
    
    def analyze_with_stockfish(self, session: EvaluationSession, dataset: PositionDataset) -> EvaluationSession:
        """Add Stockfish analysis to evaluation results.
        
        This computes centipawn loss for each move played.
        
        Args:
            session: EvaluationSession with results to analyze
            dataset: PositionDataset containing the FENs for each position
            
        Returns:
            Updated session with centipawn loss and move quality metrics
        """
        try:
            from game_arena.blitz.analysis.stockfish import MoveQualityAnalyzer
        except ImportError:
            print("⚠️ Stockfish analyzer not available (install python-chess)")
            return session
        
        if self._stockfish is None:
            try:
                self._stockfish = MoveQualityAnalyzer(
                    default_depth=self.config.stockfish_depth
                )
            except Exception as e:
                print(f"⚠️ Could not initialize Stockfish: {e}")
                return session
        
        # Build position lookup
        position_fens = {p.position_id: p.fen for p in dataset.positions}
        
        print(f"📊 Running Stockfish analysis on {len(session.results)} results...")
        
        analyzed = 0
        errors = 0
        
        for result in session.results:
            if result.move_played == "unknown":
                continue
                
            fen = position_fens.get(result.position_id)
            if not fen:
                continue
            
            try:
                analysis = self._stockfish.evaluate_move(
                    fen=fen,
                    move_str=result.move_played,
                    depth=self.config.stockfish_depth,
                )
                
                result.centipawn_loss = analysis.get("centipawn_loss", 0)
                result.is_best_move = analysis.get("is_best_move", False)
                result.is_blunder = result.centipawn_loss >= 100 if result.centipawn_loss else False
                result.move_rank = analysis.get("move_rank")
                
                analyzed += 1
                
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  ⚠️ Analysis error for {result.position_id}: {e}")
        
        print(f"✅ Analyzed {analyzed} moves ({errors} errors)")
        
        return session

