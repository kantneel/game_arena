#!/usr/bin/env python3
"""Batch Stockfish analysis for all existing matches.

Usage:
    python scripts/analyze_all_matches.py
    python scripts/analyze_all_matches.py --depth 18 --force
    python scripts/analyze_all_matches.py --match g3f_vs_g3f_20260103_182559
"""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Run Stockfish analysis on all matches")
    parser.add_argument("--results-dir", default="_results", help="Results directory")
    parser.add_argument("--depth", type=int, default=15, help="Stockfish search depth")
    parser.add_argument("--multipv", type=int, default=3, help="Number of principal variations")
    parser.add_argument("--force", action="store_true", help="Re-analyze even if already done")
    parser.add_argument("--match", type=str, help="Analyze specific match only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be analyzed")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return 1
    
    # Find all match directories (exclude offline_eval)
    match_dirs = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "offline_eval":
            continue
        if args.match and d.name != args.match:
            continue
        
        # Check if it has game move files
        move_files = list(d.glob("game_*_moves.csv"))
        if not move_files:
            continue
        
        # Check if already analyzed (unless --force)
        analysis_file = d / "complete_move_analysis.csv"
        if analysis_file.exists() and not args.force:
            print(f"⏭️  Skipping (already analyzed): {d.name}")
            continue
        
        match_dirs.append(d)
    
    if not match_dirs:
        print("✅ No matches to analyze (all already done or no matches found)")
        return 0
    
    print(f"\n{'='*60}")
    print(f"♟️  BATCH STOCKFISH ANALYSIS")
    print(f"{'='*60}")
    print(f"Matches to analyze: {len(match_dirs)}")
    print(f"Depth: {args.depth}, MultiPV: {args.multipv}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("[DRY RUN] Would analyze:")
        for d in match_dirs:
            move_count = len(list(d.glob("game_*_moves.csv")))
            print(f"  - {d.name} ({move_count} games)")
        return 0
    
    # Import analyzer
    try:
        from game_arena.blitz.analysis.stockfish import MoveQualityAnalyzer
    except ImportError as e:
        print(f"❌ Could not import analyzer: {e}")
        print("   Make sure you're in the game_arena directory with venv activated")
        return 1
    
    try:
        analyzer = MoveQualityAnalyzer(
            default_depth=args.depth,
            default_multipv=args.multipv,
        )
        print(f"🏃 Using Stockfish at: {analyzer.engine_path}\n")
    except Exception as e:
        print(f"❌ Could not initialize Stockfish: {e}")
        print("   Install with: brew install stockfish")
        return 1
    
    # Analyze each match
    success = 0
    failed = 0
    total_moves = 0
    
    for i, match_dir in enumerate(match_dirs, 1):
        print(f"\n[{i}/{len(match_dirs)}] Analyzing: {match_dir.name}")
        print("-" * 50)
        
        try:
            results = analyzer.analyze_match_directory(
                match_dir,
                depth=args.depth,
                multipv=args.multipv,
                save_results=True,
            )
            
            moves_analyzed = sum(len(analyses) for analyses in results.values())
            total_moves += moves_analyzed
            
            print(f"✅ Complete: {len(results)} games, {moves_analyzed} moves")
            success += 1
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 BATCH ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Succeeded: {success}/{len(match_dirs)}")
    print(f"❌ Failed: {failed}/{len(match_dirs)}")
    print(f"📊 Total moves analyzed: {total_moves}")
    print(f"{'='*60}\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

