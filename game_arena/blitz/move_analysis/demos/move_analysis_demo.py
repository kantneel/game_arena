#!/usr/bin/env python3
"""Demonstration script for chess move quality analysis.

This script shows how to use the MoveQualityAnalyzer to analyze moves
from blitz chess games.
"""

import sys
from pathlib import Path
import pandas as pd

# Add the project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from game_arena.blitz.move_analysis.move_analyzer import MoveQualityAnalyzer


def demo_single_move_analysis():
    """Demonstrate analyzing a single chess move."""
    print("🔍 Demo: Single Move Analysis")
    print("=" * 40)
    
    try:
        analyzer = MoveQualityAnalyzer(default_depth=15, default_multipv=3)
        print(f"✅ Initialized analyzer with Stockfish at: {analyzer.engine_path}")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        print("💡 Run 'python setup_stockfish.py' to set up Stockfish first")
        return
    
    # Analyze the opening move 1.e4
    print("\n📝 Analyzing 1.e4 from starting position...")
    
    try:
        result = analyzer.evaluate_move(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            move_str="e4",
            depth=15
        )
        
        print("📊 Analysis Results:")
        print(f"  Best move: {result['pretty']['best_move_san']}")
        print(f"  Best evaluation: {result['pretty']['best_eval_str']}")
        print(f"  Played evaluation: {result['pretty']['played_eval_str']}")
        print(f"  Centipawn loss: {result['pretty']['cp_loss_str']}")
        print(f"  Move rank: {result['played_move_rank_among_top']}")
        
    except Exception as e:
        print(f"❌ Error analyzing move: {e}")


def demo_match_analysis():
    """Demonstrate analyzing a complete match."""
    print("\n\n🎯 Demo: Complete Match Analysis")
    print("=" * 40)
    
    # Look for available match directories
    results_dir = Path(__file__).parent / "_results"
    if not results_dir.exists():
        print("❌ No _results directory found")
        return
    
    match_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not match_dirs:
        print("❌ No match directories found in _results")
        return
    
    # Use the first available match
    match_dir = match_dirs[0]
    print(f"📁 Analyzing match: {match_dir.name}")
    
    # Check if there are game move files
    move_files = list(match_dir.glob("game_*_moves.csv"))
    if not move_files:
        print(f"❌ No game move files found in {match_dir}")
        return
    
    print(f"🎮 Found {len(move_files)} games to analyze")
    
    try:
        analyzer = MoveQualityAnalyzer(default_depth=12, default_multipv=3)  # Faster for demo
        
        print("⚡ Running analysis (using depth=12 for speed)...")
        results = analyzer.analyze_match_directory(
            match_dir, 
            depth=12, 
            multipv=3, 
            save_results=True
        )
        
        print(f"\n✅ Analysis complete!")
        print(f"📈 Analyzed {sum(len(analyses) for analyses in results.values())} total moves")
        
        # Show where results were saved
        print(f"\n📁 Results saved to:")
        print(f"  Complete analysis: {match_dir}/complete_move_analysis.csv")
        print(f"  Summary statistics: {match_dir}/move_analysis_summary.json")
        
    except Exception as e:
        print(f"❌ Error during match analysis: {e}")


def demo_quality_categories():
    """Demonstrate move quality categorization."""
    print("\n\n📊 Demo: Move Quality Categories")
    print("=" * 40)
    
    # Example centipawn losses and their classifications
    examples = [
        (0, "Perfect/Best move"),
        (5, "Excellent move"),
        (15, "Good move"),
        (35, "Inaccuracy (25-49 cp loss)"),
        (75, "Mistake (50-99 cp loss)"),
        (150, "Blunder (100+ cp loss)"),
        (300, "Major blunder"),
    ]
    
    print("📏 Move Quality Scale:")
    for cp_loss, description in examples:
        print(f"  {cp_loss:3d} cp loss: {description}")
    
    print("\n💡 Tips for interpretation:")
    print("  - 0-10 cp: Excellent moves, engine would play similarly")
    print("  - 10-25 cp: Good moves, reasonable choices")
    print("  - 25-49 cp: Inaccuracies, suboptimal but not terrible")
    print("  - 50-99 cp: Mistakes, clearly worse moves")
    print("  - 100+ cp: Blunders, significant errors")


def demo_analysis_results():
    """Show example of how to read analysis results."""
    print("\n\n📋 Demo: Reading Analysis Results")
    print("=" * 40)
    
    # Look for existing analysis files
    results_dir = Path(__file__).parent / "_results"
    if not results_dir.exists():
        print("❌ No analysis results found. Run match analysis first.")
        return
    
    for match_dir in results_dir.iterdir():
        if not match_dir.is_dir():
            continue
            
        analysis_file = match_dir / "complete_move_analysis.csv"
        if analysis_file.exists():
            print(f"📖 Reading analysis from: {analysis_file.name}")
            
            df = pd.read_csv(analysis_file)
            
            # Show basic statistics
            print(f"\n📈 Basic Statistics:")
            print(f"  Total moves analyzed: {len(df)}")
            print(f"  Average centipawn loss: {df['centipawn_loss'].mean():.1f}")
            print(f"  Median centipawn loss: {df['centipawn_loss'].median():.1f}")
            
            # Quality breakdown
            perfect = len(df[df['centipawn_loss'] == 0])
            good = len(df[(df['centipawn_loss'] > 0) & (df['centipawn_loss'] < 25)])
            inaccuracies = len(df[(df['centipawn_loss'] >= 25) & (df['centipawn_loss'] < 50)])
            mistakes = len(df[(df['centipawn_loss'] >= 50) & (df['centipawn_loss'] < 100)])
            blunders = len(df[df['centipawn_loss'] >= 100])
            
            print(f"\n🎯 Move Quality Breakdown:")
            print(f"  Perfect moves (0 cp): {perfect} ({perfect/len(df)*100:.1f}%)")
            print(f"  Good moves (1-24 cp): {good} ({good/len(df)*100:.1f}%)")
            print(f"  Inaccuracies (25-49 cp): {inaccuracies} ({inaccuracies/len(df)*100:.1f}%)")
            print(f"  Mistakes (50-99 cp): {mistakes} ({mistakes/len(df)*100:.1f}%)")
            print(f"  Blunders (100+ cp): {blunders} ({blunders/len(df)*100:.1f}%)")
            
            # Player comparison
            if 'player' in df.columns:
                print(f"\n👥 Player Comparison:")
                for player in df['player'].unique():
                    player_df = df[df['player'] == player]
                    avg_loss = player_df['centipawn_loss'].mean()
                    blunder_rate = len(player_df[player_df['centipawn_loss'] >= 100]) / len(player_df) * 100
                    print(f"  {player}: {avg_loss:.1f} avg cp loss, {blunder_rate:.1f}% blunder rate")
            
            break
    else:
        print("❌ No complete analysis files found.")
        print("💡 Run the match analysis demo first to generate results.")


def main():
    """Run all demonstrations."""
    print("🎮 Game Arena - Move Analysis Demo")
    print("="*50)
    print("This demo shows how to analyze chess move quality using Stockfish.\n")
    
    # Run demos
    demo_single_move_analysis()
    demo_match_analysis() 
    demo_quality_categories()
    demo_analysis_results()
    
    print("\n\n🏁 Demo Complete!")
    print("\n📚 Next Steps:")
    print("1. Install dependencies: pip install chess pandas")
    print("2. Set up Stockfish: python setup_stockfish.py")
    print("3. Analyze your own matches:")
    print("   python -m game_arena.blitz.move_analysis.move_analyzer <match_directory>")
    print("4. Integrate with tournament analysis pipeline")


if __name__ == "__main__":
    main()
