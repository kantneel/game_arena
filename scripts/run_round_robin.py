#!/usr/bin/env python3
"""
Round-Robin Tournament: All vs All for Gemini Models

Sets up a complete round-robin where each model pair plays N games.
Time control: 3+2 (3 minutes + 2 second increment)
"""

import argparse
import subprocess
import sys
from itertools import combinations
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal

# The 4 Gemini models
MODELS = [
    "gemini-3-pro",
    "gemini-3-flash", 
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

@dataclass
class Matchup:
    model_a: str
    model_b: str
    games: int
    time_control: str
    notes: str
    notation: str = "san"
    
    @property
    def name(self) -> str:
        a_short = self.model_a.replace("gemini-", "g").replace("-", "")
        b_short = self.model_b.replace("gemini-", "g").replace("-", "")
        return f"{a_short}_vs_{b_short}"


def generate_matchups(games_per_matchup: int, time_control: str, notation: str = "san") -> list[Matchup]:
    """Generate all unique matchups between models."""
    matchups = []
    for model_a, model_b in combinations(MODELS, 2):
        matchups.append(Matchup(
            model_a=model_a,
            model_b=model_b,
            games=games_per_matchup,
            time_control=time_control,
            notes=f"Round-robin: {model_a} vs {model_b}, {games_per_matchup} games @ {time_control}",
            notation=notation,
        ))
    return matchups


# Global state for graceful shutdown
active_processes: dict[int, subprocess.Popen] = {}
shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    if shutdown_requested:
        print("\n🚨 Force killing all processes...")
        for pid, proc in list(active_processes.items()):
            try:
                proc.kill()
            except:
                pass
        sys.exit(1)
    else:
        print("\n🛑 Cancellation requested - gracefully stopping...")
        print("   (Press Ctrl+C again to force kill)")
        shutdown_requested = True
        for pid, proc in list(active_processes.items()):
            if proc.poll() is None:
                proc.terminate()


def run_matchup(matchup: Matchup) -> tuple[str, bool, str]:
    """Run a single matchup."""
    global shutdown_requested
    
    if shutdown_requested:
        return (matchup.name, False, "cancelled")
    
    # Parse time control
    base_time, increment = matchup.time_control.split("+")
    base_seconds = int(base_time) * 60
    increment_seconds = int(increment)
    
    cmd = [
        sys.executable, "-m", "game_arena.blitz.match",
        "--model_a", matchup.model_a,
        "--model_b", matchup.model_b,
        "--initial_time_seconds", str(base_seconds),
        "--increment_seconds", str(increment_seconds),
        "--total_games", str(matchup.games),  # play exactly N games
        "--notes", matchup.notes,
        "--move_notation_format", matchup.notation,
        "--enable_response_feedback",  # Enable previous response analysis
    ]
    
    print(f"🎮 Starting: {matchup.name} ({matchup.games} games @ {matchup.time_control})")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/Users/neelkant/Desktop/Work/game_arena"
        )
        active_processes[process.pid] = process
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            if shutdown_requested:
                process.terminate()
                break
            if line:
                print(f"   [{matchup.name}] {line.rstrip()}")
        
        process.wait()
        del active_processes[process.pid]
        
        if process.returncode == 0:
            return (matchup.name, True, "success")
        else:
            return (matchup.name, False, f"exit code {process.returncode}")
            
    except Exception as e:
        return (matchup.name, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Run round-robin tournament between Gemini models")
    parser.add_argument("--games", type=int, default=10, help="Games per matchup (default: 10)")
    parser.add_argument("--time", type=str, default="3+2", help="Time control (default: 3+2)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel matchups (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Show matchups without running")
    parser.add_argument("--matchup", type=str, help="Run only a specific matchup (e.g., 'g3pro_vs_g3flash')")
    parser.add_argument("--notation", type=str, default="san", choices=["san", "lan", "pgn"],
                       help="Move notation format: san (Standard), lan (Long), pgn (Full PGN)")
    args = parser.parse_args()
    
    matchups = generate_matchups(args.games, args.time, args.notation)
    
    # Filter to specific matchup if requested
    if args.matchup:
        matchups = [m for m in matchups if args.matchup.lower() in m.name.lower()]
        if not matchups:
            print(f"❌ No matchup found matching '{args.matchup}'")
            print("Available matchups:")
            for m in generate_matchups(args.games, args.time):
                print(f"  - {m.name}")
            return
    
    print("=" * 60)
    print("🏆 ROUND-ROBIN TOURNAMENT")
    print("=" * 60)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Matchups: {len(matchups)}")
    print(f"Games per matchup: {args.games}")
    print(f"Time control: {args.time}")
    print(f"Move notation: {args.notation.upper()}")
    print(f"Total games: {len(matchups) * args.games}")
    print("=" * 60)
    print()
    
    for i, m in enumerate(matchups, 1):
        print(f"  {i}. {m.model_a} vs {m.model_b}")
    print()
    
    if args.dry_run:
        print("[DRY RUN] Would run the above matchups.")
        return
    
    # Set up signal handler
    original_handler = signal.signal(signal.SIGINT, signal_handler)
    
    results = {}
    try:
        if args.parallel > 1:
            print(f"🚀 Running {args.parallel} matchups in parallel\n")
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {executor.submit(run_matchup, m): m for m in matchups}
                for future in as_completed(futures):
                    if shutdown_requested:
                        break
                    name, success, msg = future.result()
                    results[name] = success
                    status = "✅" if success else "❌"
                    print(f"\n{status} {name}: {msg}")
        else:
            for matchup in matchups:
                if shutdown_requested:
                    break
                name, success, msg = run_matchup(matchup)
                results[name] = success
                status = "✅" if success else "❌"
                print(f"\n{status} {name}: {msg}")
                
    finally:
        signal.signal(signal.SIGINT, original_handler)
    
    # Summary
    print()
    print("=" * 60)
    print("📋 TOURNAMENT SUMMARY")
    print("=" * 60)
    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    cancelled = len(matchups) - len(results)
    
    print(f"✅ Completed: {succeeded}/{len(matchups)}")
    print(f"❌ Failed: {failed}/{len(matchups)}")
    if cancelled:
        print(f"🛑 Cancelled: {cancelled}/{len(matchups)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

