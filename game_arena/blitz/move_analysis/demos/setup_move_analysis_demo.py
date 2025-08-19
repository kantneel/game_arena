#!/usr/bin/env python3
"""
Setup script to demonstrate move quality analysis integration with notebooks.

This script shows how to:
1. Run move analysis on a match directory 
2. Generate updated notebooks with move quality charts
3. View the enhanced analysis results
"""

import sys
from pathlib import Path
import os

# Add the project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from game_arena.blitz.move_analysis.move_analyzer import MoveQualityAnalyzer
from game_arena.blitz.data_collector import create_analysis_notebook


def find_available_matches():
    """Find available match directories for analysis."""
    results_dir = Path(__file__).parent / "_results"
    if not results_dir.exists():
        print("❌ No _results directory found")
        return []
    
    match_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not match_dirs:
        print("❌ No match directories found in _results")
        return []
    
    # Filter to matches with game move files
    valid_matches = []
    for match_dir in match_dirs:
        move_files = list(match_dir.glob("game_*_moves.csv"))
        if move_files:
            valid_matches.append(match_dir)
    
    return valid_matches


def demo_move_analysis_integration():
    """Demonstrate the complete move analysis + notebook integration."""
    print("🎮 Move Quality Analysis + Notebook Integration Demo")
    print("=" * 60)
    
    # Find available matches
    matches = find_available_matches()
    if not matches:
        print("❌ No valid match directories found with move data")
        print("💡 Run some blitz matches first to generate data")
        return False
    
    # Use the first available match
    match_dir = matches[0]
    print(f"📁 Analyzing match: {match_dir.name}")
    
    # Check if we have move files
    move_files = list(match_dir.glob("game_*_moves.csv"))
    print(f"🎮 Found {len(move_files)} games with move data")
    
    # Check if analysis already exists
    analysis_file = match_dir / "complete_move_analysis.csv"
    has_existing_analysis = analysis_file.exists()
    
    if has_existing_analysis:
        print(f"✅ Move analysis already exists: {analysis_file}")
        print("🔄 Skipping analysis generation (delete file to regenerate)")
    else:
        print("🔬 Running move quality analysis...")
        
        try:
            analyzer = MoveQualityAnalyzer(
                default_depth=12,  # Faster for demo
                default_multipv=3
            )
            
            print(f"🏃 Using Stockfish at: {analyzer.engine_path}")
            print("⚡ Running analysis (using depth=12 for speed)...")
            
            results = analyzer.analyze_match_directory(
                match_dir,
                depth=12,
                multipv=3,
                save_results=True
            )
            
            total_moves = sum(len(analyses) for analyses in results.values())
            print(f"✅ Analysis complete! Analyzed {total_moves} moves")
            
        except Exception as e:
            print(f"❌ Move analysis failed: {e}")
            print("💡 Make sure Stockfish is installed:")
            print("   - macOS: brew install stockfish")
            print("   - Ubuntu: sudo apt install stockfish")
            print("   - Or download from https://stockfishchess.org/")
            return False
    
    # Regenerate notebook with move analysis charts
    print("\n📓 Regenerating analysis notebook with move quality charts...")
    
    try:
        create_analysis_notebook(match_dir.name, str(match_dir.parent))
        notebook_path = match_dir / f"{match_dir.name}_analysis.ipynb"
        
        print(f"✅ Enhanced notebook created: {notebook_path}")
        print("\n🎯 New features in the notebook:")
        print("  📊 Move Quality vs Time Taken (both models)")
        print("  📈 Move Quality vs Turn Number (both models)")
        print("  📋 Move Quality Distribution")
        print("  📊 Move Quality Summary Statistics")
        print("  🎚️  Quality threshold reference lines:")
        print("    - Orange: Inaccuracy threshold (25 cp)")
        print("    - Red: Mistake threshold (50 cp)")
        print("    - Dark Red: Blunder threshold (100 cp)")
        
        print(f"\n🚀 To view the enhanced analysis:")
        print(f"   jupyter notebook {notebook_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook generation failed: {e}")
        return False


def demo_move_quality_scale():
    """Show the move quality scale used in the analysis."""
    print("\n\n📏 Move Quality Scale (Centipawn Loss)")
    print("=" * 40)
    
    examples = [
        (0, "Perfect move", "Engine's top choice"),
        (5, "Excellent", "Very strong move"),
        (15, "Good", "Reasonable choice"),
        (35, "Inaccuracy", "Suboptimal but playable"),
        (75, "Mistake", "Clearly worse move"),
        (150, "Blunder", "Significant error"),
        (300, "Major blunder", "Game-changing mistake"),
    ]
    
    for cp_loss, category, description in examples:
        print(f"  {cp_loss:3d} cp: {category:12} - {description}")
    
    print("\n💡 How to interpret the charts:")
    print("  • Lower centipawn loss = better move quality")
    print("  • Y-axis shows centipawn loss (higher = worse)")
    print("  • Horizontal reference lines mark quality thresholds")
    print("  • Compare patterns between models across the game")


def show_file_structure():
    """Show what files are generated by the move analysis."""
    print("\n\n📁 Generated Files Structure")
    print("=" * 40)
    
    structure = """
match_directory/
├── game_1_moves.csv              # Original move data
├── game_2_moves.csv              # (more games...)
├── metadata.json                 # Match metadata  
├── games_summary.csv             # Game results
├── complete_move_analysis.csv    # 🆕 Move quality analysis
├── move_analysis_summary.json    # 🆕 Quality statistics
├── game_1_move_analysis.csv      # 🆕 Per-game analysis
├── game_2_move_analysis.csv      # 🆕 (more per-game analysis...)
└── match_id_analysis.ipynb       # 🆕 Enhanced notebook
"""
    
    print(structure)
    print("🆕 = New files generated by move quality analysis")


def main():
    """Run the complete demonstration."""
    print("🎮 Game Arena - Enhanced Move Analysis Demo")
    print("This demo shows the integration of Stockfish move analysis with Jupyter notebooks.\n")
    
    # Run the main demo
    success = demo_move_analysis_integration()
    
    # Show additional information
    demo_move_quality_scale()
    show_file_structure()
    
    print("\n\n🏁 Demo Complete!")
    
    if success:
        print("\n📚 Next Steps:")
        print("1. Open the enhanced Jupyter notebook")
        print("2. Run all cells to see the move quality analysis")
        print("3. Compare move quality patterns between models")
        print("4. Look for correlations between time pressure and move quality")
        print("5. Identify critical moments where blunders occurred")
        
        print("\n🔧 Advanced Usage:")
        print("• Adjust analysis depth for accuracy vs speed trade-off")
        print("• Run analysis on tournament data for comprehensive comparison")
        print("• Use move analysis to identify model strengths and weaknesses")
    else:
        print("\n🛠️  Setup Required:")
        print("1. Install Stockfish chess engine")
        print("2. Run some blitz matches to generate move data")
        print("3. Re-run this demo script")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
