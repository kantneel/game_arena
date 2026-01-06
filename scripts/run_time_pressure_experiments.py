#!/usr/bin/env python3
"""
Time Pressure Experiment Runner

Runs the planned experiments for analyzing LLM behavior under time pressure.
Configure which experiments to run by uncommenting/commenting the EXPERIMENTS list.

Usage:
    python scripts/run_time_pressure_experiments.py --experiment 1
    python scripts/run_time_pressure_experiments.py --all
    python scripts/run_time_pressure_experiments.py --list
    
    # Offline evaluation experiments
    python scripts/run_time_pressure_experiments.py --offline 40
    python scripts/run_time_pressure_experiments.py --offline-all
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Experiment:
    """Defines a live game experiment configuration."""
    id: int
    name: str
    model_a: str
    model_b: str
    purpose: str
    games: int = 6
    time_control: str = "300+3"  # 5 minutes + 3 second increment
    # Experiment flags
    enable_time_pressure_prompt: bool = True
    use_dramatic_prompts: bool = False
    enable_response_feedback: bool = False
    enable_efficiency_guidance: bool = False
    # Notes for experiment tracking
    notes: str = ""


@dataclass
class OfflineExperiment:
    """Defines an offline position evaluation experiment."""
    id: int
    name: str
    model: str
    purpose: str
    dataset: str = "standard"  # "standard" or path to JSON
    category: Optional[str] = None  # Filter by category
    samples_per_condition: int = 3
    time_levels: list = field(default_factory=lambda: [300, 120, 60, 30, 15])
    # Experiment flags
    enable_time_pressure_prompt: bool = True
    use_dramatic_prompts: bool = False


# Experiment definitions
EXPERIMENTS = [
    # ===== PHASE 1: BASELINES (Self-Play) =====
    Experiment(
        id=1,
        name="G3F_baseline",
        model_a="gemini-3-flash",
        model_b="gemini-3-flash",
        purpose="Baseline variance, white/black asymmetry for Gemini 3 Flash",
        games=8,
        notes="Phase 1: G3 Flash self-play baseline",
    ),
    Experiment(
        id=2,
        name="G3P_baseline",
        model_a="gemini-3-pro",
        model_b="gemini-3-pro",
        purpose="Baseline for Gemini 3 Pro tier",
        games=6,
        notes="Phase 1: G3 Pro self-play baseline",
    ),
    Experiment(
        id=3,
        name="G25F_baseline",
        model_a="gemini-2.5-flash",
        model_b="gemini-2.5-flash",
        purpose="Prior-gen baseline for Flash tier",
        games=6,
        notes="Phase 1: G2.5 Flash self-play baseline",
    ),
    
    # ===== PHASE 2: WITHIN-GENERATION (Pro vs Flash) =====
    Experiment(
        id=4,
        name="G3_pro_vs_flash",
        model_a="gemini-3-pro",
        model_b="gemini-3-flash",
        purpose="Gen 3: Pro vs Flash pressure handling",
        games=8,
        notes="Phase 2: G3 Pro vs Flash (same gen, diff tier)",
    ),
    Experiment(
        id=5,
        name="G25_pro_vs_flash",
        model_a="gemini-2.5-pro",
        model_b="gemini-2.5-flash",
        purpose="Gen 2.5: Pro vs Flash pressure handling",
        games=6,
        notes="Phase 2: G2.5 Pro vs Flash (same gen, diff tier)",
    ),
    
    # ===== PHASE 3: CROSS-GENERATION (Same Tier) =====
    Experiment(
        id=6,
        name="flash_evolution",
        model_a="gemini-3-flash",
        model_b="gemini-2.5-flash",
        purpose="Flash tier evolution: G3F vs G25F",
        games=8,
        notes="Phase 3: Flash evolution (G3 vs G2.5)",
    ),
    Experiment(
        id=7,
        name="pro_evolution",
        model_a="gemini-3-pro",
        model_b="gemini-2.5-pro",
        purpose="Pro tier evolution: G3P vs G25P",
        games=6,
        notes="Phase 3: Pro evolution (G3 vs G2.5)",
    ),
    
    # ===== PHASE 4: CROSS-TIER CROSS-GENERATION =====
    Experiment(
        id=8,
        name="new_flash_vs_old_pro",
        model_a="gemini-3-flash",
        model_b="gemini-2.5-pro",
        purpose="Is new Flash better than old Pro?",
        games=6,
        notes="Phase 4: Cross-tier cross-gen (G3 Flash vs G2.5 Pro)",
    ),
    
    # ===== PHASE 5: PROMPT ABLATION EXPERIMENTS =====
    Experiment(
        id=10,
        name="G3F_no_time_prompt",
        model_a="gemini-3-flash",
        model_b="gemini-3-flash",
        purpose="ABLATION: G3F self-play WITHOUT time pressure prompts",
        games=8,
        enable_time_pressure_prompt=False,  # Key difference
        notes="Phase 5: Ablation - no time pressure prompts",
    ),
    Experiment(
        id=11,
        name="G3F_dramatic_prompts",
        model_a="gemini-3-flash",
        model_b="gemini-3-flash",
        purpose="G3F self-play WITH dramatic ALL-CAPS time pressure prompts",
        games=8,
        use_dramatic_prompts=True,  # Key difference
        notes="Phase 5: Ablation - dramatic ALL-CAPS prompts",
    ),
    
    # ===== PHASE 6: RESPONSE FEEDBACK EXPERIMENTS =====
    Experiment(
        id=20,
        name="G3F_response_feedback",
        model_a="gemini-3-flash",
        model_b="gemini-3-flash",
        purpose="G3F self-play WITH previous response feedback (recurrent awareness)",
        games=8,
        enable_response_feedback=True,  # Key difference
        notes="Phase 6: Response feedback only",
    ),
    Experiment(
        id=21,
        name="G3F_efficiency_guidance",
        model_a="gemini-3-flash",
        model_b="gemini-3-flash",
        purpose="G3F self-play WITH response feedback + efficiency guidance",
        games=8,
        enable_response_feedback=True,
        enable_efficiency_guidance=True,  # Also enable guidance
        notes="Phase 6: Response feedback + efficiency guidance",
    ),
    Experiment(
        id=22,
        name="G3F_full_awareness",
        model_a="gemini-3-flash",
        model_b="gemini-3-flash",
        purpose="G3F self-play WITH all awareness features (dramatic + feedback + guidance)",
        games=8,
        use_dramatic_prompts=True,
        enable_response_feedback=True,
        enable_efficiency_guidance=True,
        notes="Phase 6: Full awareness (dramatic + feedback + guidance)",
    ),
    
    # ===== PHASE 7: COMPARISON WITH AWARENESS FEATURES =====
    Experiment(
        id=30,
        name="G3P_vs_G3F_with_feedback",
        model_a="gemini-3-pro",
        model_b="gemini-3-flash",
        purpose="Pro vs Flash with response feedback enabled",
        games=8,
        enable_response_feedback=True,
        enable_efficiency_guidance=True,
        notes="Phase 7: Pro vs Flash with full awareness",
    ),
]

# ===== OFFLINE EVALUATION EXPERIMENTS =====
OFFLINE_EXPERIMENTS = [
    # Phase 8: Controlled position evaluation
    OfflineExperiment(
        id=40,
        name="G3F_offline_full",
        model="gemini-3-flash",
        purpose="Full offline evaluation of G3 Flash across all time levels",
        samples_per_condition=3,
        time_levels=[300, 120, 60, 30, 15],
    ),
    OfflineExperiment(
        id=41,
        name="G3P_offline_full",
        model="gemini-3-pro",
        purpose="Full offline evaluation of G3 Pro across all time levels",
        samples_per_condition=3,
        time_levels=[300, 120, 60, 30, 15],
    ),
    OfflineExperiment(
        id=42,
        name="G25F_offline_full",
        model="gemini-2.5-flash",
        purpose="Full offline evaluation of G2.5 Flash for comparison",
        samples_per_condition=3,
        time_levels=[300, 120, 60, 30, 15],
    ),
    OfflineExperiment(
        id=43,
        name="G3F_offline_no_time",
        model="gemini-3-flash",
        purpose="ABLATION: G3 Flash without time pressure prompts",
        samples_per_condition=3,
        time_levels=[300, 60, 15],
        enable_time_pressure_prompt=False,
    ),
    OfflineExperiment(
        id=44,
        name="G3F_offline_dramatic",
        model="gemini-3-flash",
        purpose="G3 Flash with dramatic time pressure prompts",
        samples_per_condition=3,
        time_levels=[300, 60, 15],
        use_dramatic_prompts=True,
    ),
    OfflineExperiment(
        id=45,
        name="G3F_offline_tactical",
        model="gemini-3-flash",
        purpose="G3 Flash on tactical positions only",
        samples_per_condition=3,
        category="tactical",
        time_levels=[300, 60, 15],
    ),
    OfflineExperiment(
        id=46,
        name="G3F_offline_endgame",
        model="gemini-3-flash",
        purpose="G3 Flash on endgame positions only",
        samples_per_condition=3,
        category="endgame",
        time_levels=[300, 60, 15],
    ),
]


def get_experiment(exp_id: int) -> Optional[Experiment]:
    """Get live game experiment by ID."""
    for exp in EXPERIMENTS:
        if exp.id == exp_id:
            return exp
    return None


def get_offline_experiment(exp_id: int) -> Optional[OfflineExperiment]:
    """Get offline experiment by ID."""
    for exp in OFFLINE_EXPERIMENTS:
        if exp.id == exp_id:
            return exp
    return None


def list_experiments():
    """Print all available experiments."""
    print("\n📋 Available Experiments:\n")
    
    # Group by phase
    phases = {
        "Baselines (1-3)": [e for e in EXPERIMENTS if e.id <= 3],
        "Within-Gen (4-5)": [e for e in EXPERIMENTS if 4 <= e.id <= 5],
        "Cross-Gen (6-8)": [e for e in EXPERIMENTS if 6 <= e.id <= 8],
        "Prompt Ablation (10-11)": [e for e in EXPERIMENTS if 10 <= e.id <= 11],
        "Response Feedback (20-22)": [e for e in EXPERIMENTS if 20 <= e.id <= 22],
        "Comparison w/ Features (30+)": [e for e in EXPERIMENTS if e.id >= 30],
    }
    
    for phase_name, experiments in phases.items():
        if not experiments:
            continue
        print(f"\n{phase_name}")
        print("-" * 80)
        print(f"{'ID':<4} {'Name':<25} {'Games':<6} {'Flags':<20} Purpose")
        
        for exp in experiments:
            flags = []
            if not exp.enable_time_pressure_prompt:
                flags.append("NoTime")
            if exp.use_dramatic_prompts:
                flags.append("Dramatic")
            if exp.enable_response_feedback:
                flags.append("Feedback")
            if exp.enable_efficiency_guidance:
                flags.append("Guidance")
            flags_str = ",".join(flags) if flags else "Standard"
            
            print(f"{exp.id:<4} {exp.name:<25} {exp.games:<6} {flags_str:<20} {exp.purpose[:40]}")
    
    # Offline experiments
    print("\n📊 Offline Position Evaluation Experiments:")
    print("-" * 80)
    print(f"{'ID':<4} {'Name':<25} {'Model':<18} {'Samples':<8} Purpose")
    
    for exp in OFFLINE_EXPERIMENTS:
        flags = []
        if not exp.enable_time_pressure_prompt:
            flags.append("NoTime")
        if exp.use_dramatic_prompts:
            flags.append("Dramatic")
        if exp.category:
            flags.append(f"cat:{exp.category}")
        flags_str = f" [{','.join(flags)}]" if flags else ""
        
        print(f"{exp.id:<4} {exp.name:<25} {exp.model:<18} {exp.samples_per_condition:<8} {exp.purpose[:35]}{flags_str}")
    
    print("\n")
    print("Run live game:   python scripts/run_time_pressure_experiments.py --experiment <ID>")
    print("Run offline:     python scripts/run_time_pressure_experiments.py --offline <ID>")
    print("Run all live:    python scripts/run_time_pressure_experiments.py --all")
    print("Run all offline: python scripts/run_time_pressure_experiments.py --offline-all")
    print("Run phase:       python scripts/run_time_pressure_experiments.py --range 1-3")


def run_experiment(exp: Experiment, dry_run: bool = False):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"🧪 Experiment {exp.id}: {exp.name}")
    print(f"   {exp.model_a} vs {exp.model_b}")
    print(f"   {exp.games} games, {exp.time_control} time control")
    print(f"   Purpose: {exp.purpose}")
    
    # Print special flags
    if not exp.enable_time_pressure_prompt:
        print(f"   🔬 TIME PRESSURE PROMPTS: DISABLED")
    if exp.use_dramatic_prompts:
        print(f"   🔬 DRAMATIC PROMPTS: ENABLED")
    if exp.enable_response_feedback:
        print(f"   🔬 RESPONSE FEEDBACK: ENABLED")
    if exp.enable_efficiency_guidance:
        print(f"   🔬 EFFICIENCY GUIDANCE: ENABLED")
    
    print(f"{'='*60}\n")
    
    # Parse time control
    base_time, increment = exp.time_control.split("+")
    base_time_seconds = int(base_time)
    increment_seconds = int(increment)
    
    # Build command
    cmd = [
        "python", "-m", "game_arena.blitz.match",
        "--model_a", exp.model_a,
        "--model_b", exp.model_b,
        "--first_to", str((exp.games + 1) // 2),  # First to N wins
        "--initial_time_seconds", str(base_time_seconds),
        "--increment_seconds", str(increment_seconds),
        "--use_rethinking",
    ]
    
    # Add experiment-specific flags
    if not exp.enable_time_pressure_prompt:
        cmd.append("--noenable_time_pressure_prompt")
    
    if exp.use_dramatic_prompts:
        cmd.append("--use_dramatic_prompts")
    
    if exp.enable_response_feedback:
        cmd.append("--enable_response_feedback")
    
    if exp.enable_efficiency_guidance:
        cmd.append("--enable_efficiency_guidance")
    
    # Add notes for experiment tracking
    if exp.notes:
        cmd.extend(["--notes", exp.notes])
    
    if dry_run:
        print(f"[DRY RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        return True
    
    print(f"Executing: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Experiment {exp.id} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment {exp.id} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment {exp.id} interrupted by user")
        return False


def run_offline_experiment(exp: OfflineExperiment, dry_run: bool = False):
    """Run an offline position evaluation experiment."""
    print(f"\n{'='*60}")
    print(f"📊 Offline Experiment {exp.id}: {exp.name}")
    print(f"   Model: {exp.model}")
    print(f"   Dataset: {exp.dataset}" + (f" (category: {exp.category})" if exp.category else ""))
    print(f"   Samples per condition: {exp.samples_per_condition}")
    print(f"   Time levels: {exp.time_levels}")
    print(f"   Purpose: {exp.purpose}")
    
    # Print special flags
    if not exp.enable_time_pressure_prompt:
        print(f"   🔬 TIME PRESSURE PROMPTS: DISABLED")
    if exp.use_dramatic_prompts:
        print(f"   🔬 DRAMATIC PROMPTS: ENABLED")
    
    print(f"{'='*60}\n")
    
    # Build command
    time_levels_str = ",".join(str(t) for t in exp.time_levels)
    cmd = [
        "python", "scripts/run_offline_eval.py",
        "--model", exp.model,
        "--samples", str(exp.samples_per_condition),
        "--time_levels", time_levels_str,
        "--dataset", exp.dataset,
    ]
    
    if exp.category:
        cmd.extend(["--category", exp.category])
    
    if not exp.enable_time_pressure_prompt:
        cmd.append("--noenable_time_pressure_prompt")
    
    if exp.use_dramatic_prompts:
        cmd.append("--use_dramatic_prompts")
    
    if dry_run:
        print(f"[DRY RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        return True
    
    print(f"Executing: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Offline experiment {exp.id} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Offline experiment {exp.id} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Offline experiment {exp.id} interrupted by user")
        return False


def _build_experiment_command(exp: Experiment) -> list[str]:
    """Build the command for running an experiment."""
    base_time, increment = exp.time_control.split("+")
    
    cmd = [
        "python", "-m", "game_arena.blitz.match",
        "--model_a", exp.model_a,
        "--model_b", exp.model_b,
        "--first_to", str((exp.games + 1) // 2),
        "--initial_time_seconds", str(int(base_time)),
        "--increment_seconds", str(int(increment)),
        "--use_rethinking",
    ]
    
    if not exp.enable_time_pressure_prompt:
        cmd.append("--noenable_time_pressure_prompt")
    if exp.use_dramatic_prompts:
        cmd.append("--use_dramatic_prompts")
    if exp.enable_response_feedback:
        cmd.append("--enable_response_feedback")
    if exp.enable_efficiency_guidance:
        cmd.append("--enable_efficiency_guidance")
    if exp.notes:
        cmd.extend(["--notes", exp.notes])
    
    return cmd


def _run_single_experiment_subprocess(exp: Experiment) -> tuple[int, str, bool, str]:
    """Run a single experiment in a subprocess. Module-level for pickling."""
    cmd = _build_experiment_command(exp)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return (exp.id, exp.name, True, "success")
    except subprocess.CalledProcessError as e:
        return (exp.id, exp.name, False, f"exit code {e.returncode}")
    except Exception as e:
        return (exp.id, exp.name, False, str(e))


# Global tracking for graceful shutdown
_running_processes: dict[int, subprocess.Popen] = {}
_shutdown_requested = False


def _graceful_shutdown_handler(signum, frame):
    """Handle Ctrl+C by gracefully terminating all running processes."""
    global _shutdown_requested
    
    if _shutdown_requested:
        # Second Ctrl+C - force kill
        print("\n\n⚠️  Force killing all processes...")
        for exp_id, proc in list(_running_processes.items()):
            try:
                proc.kill()
                print(f"   💀 Killed experiment {exp_id}")
            except:
                pass
        sys.exit(1)
    
    _shutdown_requested = True
    print("\n\n🛑 Cancellation requested - gracefully stopping experiments...")
    print("   (Press Ctrl+C again to force kill)\n")
    
    for exp_id, proc in list(_running_processes.items()):
        try:
            proc.terminate()
            print(f"   ⏹️  Terminating experiment {exp_id}...")
        except:
            pass


def _run_experiment_with_tracking(exp: Experiment) -> tuple[int, str, bool, str]:
    """Run experiment with process tracking for graceful shutdown."""
    global _shutdown_requested
    
    if _shutdown_requested:
        return (exp.id, exp.name, False, "cancelled before start")
    
    cmd = _build_experiment_command(exp)
    
    try:
        # Start process and track it
        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        _running_processes[exp.id] = proc
        
        # Wait for completion
        stdout, stderr = proc.communicate()
        
        # Remove from tracking
        _running_processes.pop(exp.id, None)
        
        if _shutdown_requested:
            return (exp.id, exp.name, False, "cancelled")
        
        if proc.returncode == 0:
            return (exp.id, exp.name, True, "success")
        else:
            return (exp.id, exp.name, False, f"exit code {proc.returncode}")
            
    except Exception as e:
        _running_processes.pop(exp.id, None)
        return (exp.id, exp.name, False, str(e))


def run_experiments_parallel(experiments: list[Experiment], max_workers: int = 2, dry_run: bool = False):
    """Run multiple experiments in parallel using subprocess with graceful cancellation."""
    import signal
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    global _shutdown_requested, _running_processes
    _shutdown_requested = False
    _running_processes = {}
    
    if dry_run:
        print(f"\n[DRY RUN] Would run {len(experiments)} experiments in parallel (max {max_workers} workers)")
        for exp in experiments:
            print(f"  - Experiment {exp.id}: {exp.name}")
        return
    
    print(f"\n{'='*60}")
    print(f"🚀 PARALLEL EXECUTION: {len(experiments)} experiments")
    print(f"   Max workers: {max_workers}")
    print(f"   Press Ctrl+C to gracefully stop all experiments")
    print(f"{'='*60}\n")
    
    # Set up signal handler for graceful shutdown
    original_handler = signal.signal(signal.SIGINT, _graceful_shutdown_handler)
    
    results = {}
    cancelled_count = 0
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_exp = {
                executor.submit(_run_experiment_with_tracking, exp): exp 
                for exp in experiments
            }
            
            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    exp_id, exp_name, success, msg = future.result()
                    if msg == "cancelled" or msg == "cancelled before start":
                        print(f"🛑 Experiment {exp_id} ({exp_name}): {msg}")
                        cancelled_count += 1
                        results[exp_id] = None  # Mark as cancelled
                    else:
                        status = "✅" if success else "❌"
                        print(f"{status} Experiment {exp_id} ({exp_name}): {msg}")
                        results[exp_id] = success
                except Exception as e:
                    print(f"❌ Experiment {exp.id} ({exp.name}): {e}")
                    results[exp.id] = False
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)
        
        # Clean up any remaining processes
        for exp_id, proc in list(_running_processes.items()):
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass
        _running_processes.clear()
    
    # Summary
    succeeded = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    
    print(f"\n{'='*60}")
    if _shutdown_requested:
        print(f"📋 PARALLEL EXECUTION CANCELLED")
    else:
        print(f"📋 PARALLEL EXECUTION COMPLETE")
    print(f"   ✅ Succeeded: {succeeded}/{len(experiments)}")
    print(f"   ❌ Failed: {failed}/{len(experiments)}")
    if cancelled_count > 0:
        print(f"   🛑 Cancelled: {cancelled_count}/{len(experiments)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run time pressure analysis experiments"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=int,
        help="Run specific experiment by ID"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all experiments in sequence"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available experiments"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be run without executing"
    )
    parser.add_argument(
        "--range", "-r",
        type=str,
        help="Run experiments in range, e.g., '1-4' or '4,5,6'"
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=0,
        metavar="N",
        help="Run experiments in parallel with N workers (use with --range or --all)"
    )
    parser.add_argument(
        "--offline", "-o",
        type=int,
        help="Run specific offline experiment by ID"
    )
    parser.add_argument(
        "--offline-all",
        action="store_true",
        help="Run all offline experiments"
    )
    parser.add_argument(
        "--offline-range",
        type=str,
        help="Run offline experiments in range, e.g., '40-43'"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    if args.experiment:
        exp = get_experiment(args.experiment)
        if not exp:
            print(f"❌ Experiment {args.experiment} not found")
            list_experiments()
            sys.exit(1)
        run_experiment(exp, args.dry_run)
        return
    
    if args.range:
        # Parse range like "1-4" or "1,3,5"
        exp_ids = []
        if "-" in args.range:
            start, end = map(int, args.range.split("-"))
            exp_ids = list(range(start, end + 1))
        else:
            exp_ids = [int(x) for x in args.range.split(",")]
        
        experiments = [get_experiment(exp_id) for exp_id in exp_ids]
        experiments = [e for e in experiments if e is not None]
        
        if args.parallel > 0:
            run_experiments_parallel(experiments, max_workers=args.parallel, dry_run=args.dry_run)
        else:
            for exp in experiments:
                success = run_experiment(exp, args.dry_run)
                if not success and not args.dry_run:
                    print(f"Stopping due to failure in experiment {exp.id}")
                    break
        return
    
    if args.all:
        print("\n🚀 Running all live game experiments...\n")
        if args.parallel > 0:
            run_experiments_parallel(EXPERIMENTS, max_workers=args.parallel, dry_run=args.dry_run)
        else:
            for exp in EXPERIMENTS:
                success = run_experiment(exp, args.dry_run)
                if not success and not args.dry_run:
                    print(f"Stopping due to failure in experiment {exp.id}")
                    break
        return
    
    # Offline experiment handling
    if args.offline:
        exp = get_offline_experiment(args.offline)
        if not exp:
            print(f"❌ Offline experiment {args.offline} not found")
            list_experiments()
            sys.exit(1)
        run_offline_experiment(exp, args.dry_run)
        return
    
    if args.offline_range:
        exp_ids = []
        if "-" in args.offline_range:
            start, end = map(int, args.offline_range.split("-"))
            exp_ids = list(range(start, end + 1))
        else:
            exp_ids = [int(x) for x in args.offline_range.split(",")]
        
        for exp_id in exp_ids:
            exp = get_offline_experiment(exp_id)
            if exp:
                success = run_offline_experiment(exp, args.dry_run)
                if not success and not args.dry_run:
                    print(f"Stopping due to failure in offline experiment {exp_id}")
                    break
        return
    
    if args.offline_all:
        print("\n📊 Running all offline experiments...\n")
        for exp in OFFLINE_EXPERIMENTS:
            success = run_offline_experiment(exp, args.dry_run)
            if not success and not args.dry_run:
                print(f"Stopping due to failure in offline experiment {exp.id}")
                break
        return
    
    # Default: show help
    parser.print_help()
    print("\n")
    list_experiments()


if __name__ == "__main__":
    main()
