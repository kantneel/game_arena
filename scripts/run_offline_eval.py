#!/usr/bin/env python3
"""Script to run offline evaluation experiments.

Usage:
    # Basic evaluation
    python scripts/run_offline_eval.py --model gemini-3-pro
    python scripts/run_offline_eval.py --model gemini-3-flash --samples 5
    
    # Prompt style options
    python scripts/run_offline_eval.py --model gemini-3-flash --prompt_style time_info_only  # Just clock values
    python scripts/run_offline_eval.py --model gemini-3-flash --prompt_style standard       # Clock + urgency guidance
    
    # Run ablation study (compares time_info_only vs standard)
    python scripts/run_offline_eval.py --model gemini-3-flash --ablation
    
    # Compare models
    python scripts/run_offline_eval.py --compare gemini-3-pro gemini-3-flash
    
    # Analyze existing results
    python scripts/run_offline_eval.py --analyze
"""

import argparse
from pathlib import Path

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# Model selection
flags.DEFINE_string("model", None, "Model to evaluate (e.g., gemini-3-pro)")
flags.DEFINE_list("compare", None, "List of models to compare")

# Dataset options
flags.DEFINE_enum(
    "dataset", 
    "standard", 
    ["standard", "stress", "combined"],
    "Dataset to use: 'standard' (balanced), 'stress' (blunder-prone positions), 'combined' (all)"
)
flags.DEFINE_string("dataset_file", None, "Custom dataset JSON file (overrides --dataset)")
flags.DEFINE_string("category", None, "Filter dataset by category (opening, middlegame, etc.)")
flags.DEFINE_string("difficulty", None, "Filter dataset by difficulty (easy, medium, hard)")

# Evaluation options
flags.DEFINE_list("time_levels", "300,120,60,30,15", "Time remaining levels to test (seconds)")
flags.DEFINE_integer("samples", 3, "Samples per (position, time) condition")
flags.DEFINE_integer("reasoning_budget", 8000, "Reasoning token budget")

# Prompt options
flags.DEFINE_enum(
    "prompt_style", 
    "standard", 
    ["none", "time_info_only", "standard", "dramatic"],
    "Prompt style: 'time_info_only' (just clock values), "
    "'standard' (clock values + time pressure guidance). "
    "Also available: 'none' (no time info), 'dramatic' (ALL-CAPS urgency)"
)
flags.DEFINE_bool("enable_response_feedback", False, "Include previous response feedback")

# Timeout/calibration options
flags.DEFINE_bool("simulate_clock", True, "Track whether responses would cause timeouts")
flags.DEFINE_bool("calibrate_latency", True, "Calibrate network latency before evaluation")
flags.DEFINE_integer("calibration_samples", 3, "Number of calibration warmup calls")

# Parallelization options
flags.DEFINE_integer("workers", 4, "Number of parallel workers for API calls")
flags.DEFINE_bool("sequential", False, "Run sequentially instead of parallel")

# Output options
flags.DEFINE_string("output_dir", "_results/offline_eval", "Output directory")
flags.DEFINE_bool("analyze", False, "Only analyze existing results")
flags.DEFINE_bool("generate_notebook", False, "Generate visualization notebook")

# Convenience options
flags.DEFINE_bool("all_gemini", False, "Run evaluation for all Gemini models")
flags.DEFINE_bool("ablation", False, "Run prompt style ablation study (all 4 styles)")
flags.DEFINE_bool("clean_failed", False, "Remove interrupted/failed sessions from output directory")

# Live monitoring
flags.DEFINE_string("watch", None, "Watch a running session file for live updates")
flags.DEFINE_float("watch_interval", 5.0, "Refresh interval for watch mode (seconds)")
flags.DEFINE_bool("status", False, "Show status of all sessions in output directory")

# Stockfish analysis
flags.DEFINE_bool("stockfish", False, "Run Stockfish analysis on existing results to compute centipawn loss")
flags.DEFINE_integer("stockfish_depth", 15, "Stockfish analysis depth")


def run_evaluation(model_id: str):
    """Run offline evaluation for a single model."""
    from game_arena.blitz.offline_eval import (
        OfflineEvaluator, 
        EvaluationConfig,
        create_standard_dataset,
        create_stress_test_dataset,
        create_combined_dataset,
        load_dataset,
    )
    
    # Load or create dataset
    if FLAGS.dataset_file:
        dataset_path = Path(FLAGS.dataset_file)
        print(f"Loading dataset from {dataset_path}")
        dataset = load_dataset(dataset_path)
    elif FLAGS.dataset == "standard":
        print("Using standard evaluation dataset")
        dataset = create_standard_dataset()
    elif FLAGS.dataset == "stress":
        print("Using stress test dataset (blunder-prone positions)")
        dataset = create_stress_test_dataset()
    elif FLAGS.dataset == "combined":
        print("Using combined dataset (standard + stress)")
        dataset = create_combined_dataset()
    
    # Apply filters
    if FLAGS.category:
        dataset = dataset.filter_by_category(FLAGS.category)
        print(f"Filtered to {FLAGS.category}: {len(dataset)} positions")
    
    if FLAGS.difficulty:
        dataset = dataset.filter_by_difficulty(FLAGS.difficulty)
        print(f"Filtered to {FLAGS.difficulty}: {len(dataset)} positions")
    
    # Create config
    time_levels = [float(t) for t in FLAGS.time_levels]
    config = EvaluationConfig(
        time_levels=time_levels,
        samples_per_condition=FLAGS.samples,
        reasoning_budget=FLAGS.reasoning_budget,
        prompt_style=FLAGS.prompt_style,
        enable_response_feedback=FLAGS.enable_response_feedback,
        simulate_clock=FLAGS.simulate_clock,
        calibrate_latency=FLAGS.calibrate_latency,
        calibration_samples=FLAGS.calibration_samples,
    )
    
    total_evals = len(dataset) * len(time_levels) * FLAGS.samples
    parallel = not FLAGS.sequential
    
    print(f"\n{'='*60}")
    print(f"Offline Evaluation Configuration")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset.name} ({len(dataset)} positions)")
    print(f"Time levels: {time_levels}")
    print(f"Samples per condition: {FLAGS.samples}")
    print(f"Total evaluations: {total_evals}")
    print(f"Prompt style: {FLAGS.prompt_style}")
    print(f"Simulate clock/timeouts: {FLAGS.simulate_clock}")
    print(f"Latency calibration: {FLAGS.calibrate_latency}")
    print(f"Parallel: {parallel} (workers: {FLAGS.workers})")
    print(f"{'='*60}\n")
    
    # Create evaluator and run
    output_dir = Path(FLAGS.output_dir)
    evaluator = OfflineEvaluator(
        model_id, 
        config, 
        output_dir,
        max_workers=FLAGS.workers,
    )
    
    def progress(completed, total):
        pct = completed / total * 100
        print(f"Progress: {completed}/{total} ({pct:.1f}%)", end="\r")
    
    session = evaluator.evaluate_dataset(
        dataset, 
        progress_callback=progress,
        parallel=parallel,
    )
    
    print(f"\n✅ Evaluation complete! Session: {session.session_id}")
    print(f"Results saved to: {output_dir / f'{session.session_id}.json'}")
    
    return session


def run_all_gemini_models():
    """Run evaluation for all Gemini models in parallel batches."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import subprocess
    
    gemini_models = [
        "gemini-3-flash",
        "gemini-3-pro",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]
    
    print(f"\n{'='*60}")
    print(f"🚀 Running offline eval for ALL Gemini models")
    print(f"{'='*60}")
    print(f"Models: {', '.join(gemini_models)}")
    print(f"Workers per model: {FLAGS.workers}")
    print(f"{'='*60}\n")
    
    # Build base command args
    base_args = [
        "--time_levels", ",".join(FLAGS.time_levels),
        "--samples", str(FLAGS.samples),
        "--output_dir", FLAGS.output_dir,
        "--workers", str(FLAGS.workers),
        "--prompt_style", FLAGS.prompt_style,
    ]
    
    if FLAGS.category:
        base_args.extend(["--category", FLAGS.category])
    if FLAGS.difficulty:
        base_args.extend(["--difficulty", FLAGS.difficulty])
    
    # Run each model (can be parallelized further with ProcessPoolExecutor)
    # For now, run sequentially to avoid API rate limits
    results = {}
    for model in gemini_models:
        print(f"\n{'='*60}")
        print(f"📊 Starting evaluation: {model}")
        print(f"{'='*60}\n")
        
        try:
            session = run_evaluation(model)
            results[model] = {"status": "success", "session_id": session.session_id}
        except Exception as e:
            print(f"❌ Failed: {model} - {e}")
            results[model] = {"status": "failed", "error": str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 SUMMARY: All Gemini Models")
    print(f"{'='*60}")
    for model, result in results.items():
        status = "✅" if result["status"] == "success" else "❌"
        print(f"  {status} {model}: {result.get('session_id', result.get('error'))}")
    
    # Run analysis if all completed
    successful = sum(1 for r in results.values() if r["status"] == "success")
    if successful > 0:
        print(f"\n🔍 Running analysis on {successful} completed evaluations...")
        analyze_results()


def analyze_results():
    """Analyze existing offline evaluation results."""
    from game_arena.blitz.offline_eval import OfflineAnalyzer
    
    output_dir = Path(FLAGS.output_dir)
    if not output_dir.exists():
        print(f"No results found in {output_dir}")
        return
    
    print(f"Loading sessions from {output_dir}")
    analyzer = OfflineAnalyzer(session_dir=output_dir, include_partial=True)
    
    if not analyzer.sessions:
        print("No sessions found!")
        return
    
    # Generate report
    report_path = output_dir / "analysis_report.md"
    report = analyzer.generate_report(report_path)
    print(f"\n{report}")
    
    # Generate notebook if requested
    if FLAGS.generate_notebook:
        from game_arena.blitz.offline_eval.analysis import create_visualization_notebook
        notebook_path = output_dir / "visualization.ipynb"
        create_visualization_notebook(output_dir, notebook_path)
        print(f"\nNotebook generated: {notebook_path}")
    
    # Print comparison if multiple models
    df = analyzer.to_dataframe()
    if not df.empty and df['model_id'].nunique() > 1:
        print("\n📊 Model Comparison:\n")
        comparison = analyzer.compare_models()
        print(comparison.to_string())


def run_stockfish_analysis():
    """Run Stockfish analysis on existing evaluation results."""
    from game_arena.blitz.offline_eval import (
        OfflineAnalyzer,
        create_standard_dataset,
        create_stress_test_dataset,
        create_combined_dataset,
        load_dataset,
    )
    from game_arena.blitz.offline_eval.evaluator import EvaluationSession
    
    output_dir = Path(FLAGS.output_dir)
    if not output_dir.exists():
        print(f"No results found in {output_dir}")
        return
    
    # Load dataset for FEN lookup
    if FLAGS.dataset_file:
        dataset = load_dataset(Path(FLAGS.dataset_file))
    elif FLAGS.dataset == "standard":
        dataset = create_standard_dataset()
    elif FLAGS.dataset == "stress":
        dataset = create_stress_test_dataset()
    elif FLAGS.dataset == "combined":
        dataset = create_combined_dataset()
    else:
        dataset = create_standard_dataset()
    
    position_fens = {p.position_id: p.fen for p in dataset.positions}
    
    # Try to import Stockfish analyzer
    try:
        from game_arena.blitz.analysis.stockfish import MoveQualityAnalyzer
        analyzer = MoveQualityAnalyzer(default_depth=FLAGS.stockfish_depth)
    except Exception as e:
        print(f"❌ Could not initialize Stockfish: {e}")
        print("   Make sure Stockfish is installed: brew install stockfish")
        return
    
    print(f"♟️ Running Stockfish analysis (depth={FLAGS.stockfish_depth})...")
    
    # Process each session
    for session_file in output_dir.glob("*.json"):
        if "partial" in session_file.name:
            continue
            
        try:
            session = EvaluationSession.load(session_file)
        except Exception as e:
            print(f"  ⚠️ Skipping {session_file.name}: {e}")
            continue
        
        if session.status != "completed":
            print(f"  ⏭️ Skipping {session.session_id} ({session.status})")
            continue
        
        # Check if already analyzed
        already_analyzed = sum(1 for r in session.results if r.centipawn_loss is not None)
        if already_analyzed == len(session.results):
            print(f"  ✅ Already analyzed: {session.session_id}")
            continue
        
        print(f"\n📊 Analyzing: {session.session_id} ({len(session.results)} moves)")
        
        analyzed = 0
        errors = 0
        
        for result in session.results:
            if result.move_played == "unknown":
                continue
            
            fen = position_fens.get(result.position_id)
            if not fen:
                continue
            
            try:
                analysis = analyzer.evaluate_move(
                    fen=fen,
                    move_str=result.move_played,
                    depth=FLAGS.stockfish_depth,
                )
                
                result.centipawn_loss = analysis.get("centipawn_loss", 0)
                result.is_best_move = analysis.get("is_best_move", False)
                result.is_blunder = result.centipawn_loss >= 100 if result.centipawn_loss else False
                result.move_rank = analysis.get("move_rank")
                
                analyzed += 1
                
                if analyzed % 50 == 0:
                    print(f"    Progress: {analyzed}/{len(session.results)}")
                
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"    ⚠️ Error on {result.position_id}: {e}")
        
        # Save updated session
        session.save(session_file)
        print(f"  ✅ Analyzed {analyzed} moves, saved to {session_file.name}")
    
    print("\n🔍 Run --analyze to see updated report with move quality metrics")


def show_status():
    """Show status of all sessions in output directory."""
    from game_arena.blitz.offline_eval import OfflineAnalyzer
    
    output_dir = Path(FLAGS.output_dir)
    if not output_dir.exists():
        print(f"No results directory found: {output_dir}")
        return
    
    print(f"\n📋 Session Status: {output_dir}\n")
    
    analyzer = OfflineAnalyzer(session_dir=output_dir, include_partial=True)
    
    if not analyzer.sessions:
        print("No sessions found.")
        return
    
    status_df = analyzer.get_session_status()
    
    # Format nicely
    print(f"{'Session ID':<45} {'Model':<20} {'Status':<12} {'Progress':<15}")
    print("-" * 95)
    
    for _, row in status_df.iterrows():
        status_icon = "✅" if row['status'] == "completed" else "⏳" if row['status'] == "in_progress" else "⚠️"
        progress = f"{row['completed']}/{row['expected']} ({row['completion_pct']:.0f}%)"
        print(f"{row['session_id']:<45} {row['model_id']:<20} {status_icon} {row['status']:<10} {progress:<15}")
    
    print()
    
    # Summary
    completed = sum(1 for s in analyzer.sessions if s.status == "completed")
    total_results = sum(len(s.results) for s in analyzer.sessions)
    print(f"Total: {len(analyzer.sessions)} sessions, {completed} completed, {total_results} evaluations")


def watch_session():
    """Watch a running session for live updates."""
    from game_arena.blitz.offline_eval import OfflineAnalyzer
    
    session_path = Path(FLAGS.watch)
    
    if not session_path.exists():
        # Try relative to output dir
        session_path = Path(FLAGS.output_dir) / FLAGS.watch
    
    if not session_path.exists():
        print(f"Session file not found: {FLAGS.watch}")
        print(f"Looked in: {Path(FLAGS.watch).absolute()}")
        print(f"       and: {Path(FLAGS.output_dir) / FLAGS.watch}")
        return
    
    analyzer = OfflineAnalyzer()
    analyzer.watch_live(session_path, refresh_interval=FLAGS.watch_interval)


def clean_failed_sessions():
    """Remove interrupted/failed sessions from output directory."""
    import json
    
    output_dir = Path(FLAGS.output_dir)
    if not output_dir.exists():
        print(f"No results directory found: {output_dir}")
        return
    
    removed = 0
    for session_file in output_dir.glob("*.json"):
        try:
            with open(session_file) as f:
                data = json.load(f)
            status = data.get("status", "unknown")
            if status in ["interrupted", "failed"]:
                session_file.unlink()
                print(f"🗑️  Removed: {session_file.name} ({status})")
                removed += 1
        except Exception as e:
            print(f"⚠️  Error reading {session_file.name}: {e}")
    
    if removed:
        print(f"\n✅ Removed {removed} failed/interrupted sessions")
    else:
        print("✅ No failed sessions to clean up")


def run_ablation_study(model_id: str):
    """Run prompt style ablation study for a single model.
    
    Compares two prompt styles:
    - time_info_only: Just clock values (e.g., "Your time: 1:30")
    - standard: Clock values + time pressure guidance (urgency level, time management tips)
    """
    from game_arena.blitz.offline_eval import (
        OfflineEvaluator, 
        EvaluationConfig,
        PromptStyle,
        create_standard_dataset,
        create_stress_test_dataset,
        create_combined_dataset,
        load_dataset,
    )
    
    prompt_styles = [
        PromptStyle.TIME_INFO_ONLY,
        PromptStyle.STANDARD,
    ]
    
    print(f"\n{'='*60}")
    print(f"🧪 PROMPT ABLATION STUDY")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Comparing: time_info_only vs standard")
    print(f"  - time_info_only: Just clock values")
    print(f"  - standard: Clock values + time pressure guidance")
    print(f"{'='*60}\n")
    
    # Load dataset once
    if FLAGS.dataset_file:
        dataset = load_dataset(Path(FLAGS.dataset_file))
    elif FLAGS.dataset == "standard":
        dataset = create_standard_dataset()
    elif FLAGS.dataset == "stress":
        dataset = create_stress_test_dataset()
    elif FLAGS.dataset == "combined":
        dataset = create_combined_dataset()
    else:
        dataset = create_standard_dataset()
    
    if FLAGS.category:
        dataset = dataset.filter_by_category(FLAGS.category)
    if FLAGS.difficulty:
        dataset = dataset.filter_by_difficulty(FLAGS.difficulty)
    
    time_levels = [float(t) for t in FLAGS.time_levels]
    output_dir = Path(FLAGS.output_dir)
    
    results = {}
    for style in prompt_styles:
        print(f"\n{'='*60}")
        print(f"📊 Testing prompt style: {style}")
        print(f"{'='*60}\n")
        
        config = EvaluationConfig(
            time_levels=time_levels,
            samples_per_condition=FLAGS.samples,
            reasoning_budget=FLAGS.reasoning_budget,
            prompt_style=style,
            simulate_clock=FLAGS.simulate_clock,
            calibrate_latency=FLAGS.calibrate_latency,
            calibration_samples=FLAGS.calibration_samples,
        )
        
        try:
            evaluator = OfflineEvaluator(
                model_id, 
                config, 
                output_dir,
                max_workers=FLAGS.workers,
            )
            
            session = evaluator.evaluate_dataset(
                dataset, 
                parallel=not FLAGS.sequential,
            )
            
            results[style] = {
                "status": "success",
                "session_id": session.session_id,
                "timeouts": sum(1 for r in session.results if r.would_timeout),
                "total": len(session.results),
            }
            print(f"\n✅ Completed: {style}")
            
        except Exception as e:
            print(f"\n❌ Failed: {style} - {e}")
            results[style] = {"status": "failed", "error": str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 ABLATION STUDY SUMMARY: {model_id}")
    print(f"{'='*60}")
    print(f"{'Prompt Style':<20} {'Status':<10} {'Timeout Rate':<15}")
    print("-" * 50)
    
    for style, result in results.items():
        if result["status"] == "success":
            timeout_pct = 100 * result["timeouts"] / result["total"] if result["total"] > 0 else 0
            print(f"{style:<20} ✅         {timeout_pct:.1f}%")
        else:
            print(f"{style:<20} ❌         {result.get('error', 'unknown')[:30]}")
    
    print(f"\n🔍 Run --analyze to see full comparison")


def main(argv):
    del argv  # Unused
    
    # Clean up command
    if FLAGS.clean_failed:
        clean_failed_sessions()
        return
    
    # Status and monitoring commands
    if FLAGS.status:
        show_status()
        return
    
    if FLAGS.watch:
        watch_session()
        return
    
    if FLAGS.analyze:
        analyze_results()
        return
    
    if FLAGS.stockfish:
        run_stockfish_analysis()
        return
    
    if FLAGS.all_gemini:
        run_all_gemini_models()
        return
    
    # Ablation study
    if FLAGS.ablation:
        if not FLAGS.model:
            print("Error: --ablation requires --model to specify which model to test")
            return
        run_ablation_study(FLAGS.model)
        return
    
    if FLAGS.compare:
        # Run evaluation for multiple models
        for model_id in FLAGS.compare:
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_id}")
            print(f"{'='*60}")
            run_evaluation(model_id)
        
        # Analyze after all evaluations
        analyze_results()
    
    elif FLAGS.model:
        run_evaluation(FLAGS.model)
        
        # Quick analysis
        if FLAGS.samples >= 3:
            print("\nRunning quick analysis...")
            analyze_results()
    
    else:
        print("Please specify --model or --compare or --all_gemini or --analyze")
        print("\nExamples:")
        print("  # Run single model with parallel API calls:")
        print("  python scripts/run_offline_eval.py --model gemini-3-flash --workers 4")
        print("")
        print("  # Run with specific prompt style:")
        print("  python scripts/run_offline_eval.py --model gemini-3-flash --prompt_style time_info_only")
        print("  # Styles: time_info_only (just clocks), standard (clocks + urgency)")
        print("")
        print("  # Run prompt ablation study (time_info_only vs standard):")
        print("  python scripts/run_offline_eval.py --model gemini-3-flash --ablation")
        print("")
        print("  # Run ALL Gemini models:")
        print("  python scripts/run_offline_eval.py --all_gemini --workers 4")
        print("")
        print("  # Compare specific models:")
        print("  python scripts/run_offline_eval.py --compare gemini-3-pro gemini-3-flash")
        print("")
        print("  # Run sequentially (no parallelization):")
        print("  python scripts/run_offline_eval.py --model gemini-3-flash --sequential")
        print("")
        print("  # Analyze existing results:")
        print("  python scripts/run_offline_eval.py --analyze --generate_notebook")
        print("")
        print("  # Check status of all sessions:")
        print("  python scripts/run_offline_eval.py --status")
        print("")
        print("  # Clean up failed/interrupted sessions:")
        print("  python scripts/run_offline_eval.py --clean_failed")
        print("")
        print("  # Watch a running session live:")
        print("  python scripts/run_offline_eval.py --watch gemini-3-flash_standard_eval_v1_20240101_120000.json")


if __name__ == "__main__":
    app.run(main)

