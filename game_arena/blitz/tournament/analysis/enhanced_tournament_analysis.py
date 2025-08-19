#!/usr/bin/env python3
"""Enhanced tournament analysis that includes move quality analysis using Stockfish.

This module extends the existing tournament analysis capabilities with move quality
metrics calculated using the Stockfish chess engine.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import datetime
import warnings

from game_arena.blitz import data_collector
from game_arena.blitz.move_analysis.move_analyzer import MoveQualityAnalyzer, MoveAnalysis


class EnhancedTournamentAnalyzer:
    """Enhanced tournament analyzer with move quality analysis."""
    
    def __init__(self, 
                 stockfish_path: Optional[str] = None,
                 analysis_depth: int = 15,
                 multipv: int = 3):
        """Initialize the enhanced analyzer.
        
        Args:
            stockfish_path: Path to Stockfish executable
            analysis_depth: Search depth for move analysis
            multipv: Number of principal variations to analyze
        """
        self.stockfish_path = stockfish_path
        self.analysis_depth = analysis_depth
        self.multipv = multipv
        
        # Initialize move analyzer (will fail gracefully if Stockfish not available)
        self.move_analyzer = None
        try:
            self.move_analyzer = MoveQualityAnalyzer(
                engine_path=stockfish_path,
                default_depth=analysis_depth,
                default_multipv=multipv
            )
            print(f"✅ Move quality analyzer initialized with Stockfish")
        except Exception as e:
            warnings.warn(f"Move quality analysis unavailable: {e}")
    
    def analyze_tournament_with_move_quality(self,
                                           tournament_dir: Union[str, Path],
                                           run_move_analysis: bool = True) -> Dict[str, Any]:
        """Analyze a complete tournament including move quality metrics.
        
        Args:
            tournament_dir: Path to tournament directory
            run_move_analysis: Whether to run Stockfish analysis (requires Stockfish)
            
        Returns:
            Comprehensive tournament analysis including move quality
        """
        tournament_dir = Path(tournament_dir)
        
        print(f"🔍 Analyzing tournament in: {tournament_dir}")
        
        # Find all match directories
        match_dirs = [d for d in tournament_dir.iterdir() if d.is_dir()]
        if not match_dirs:
            raise ValueError(f"No match directories found in {tournament_dir}")
        
        print(f"📁 Found {len(match_dirs)} matches to analyze")
        
        # Collect all tournament data
        tournament_data = {
            "metadata": {
                "tournament_dir": str(tournament_dir),
                "analysis_timestamp": datetime.datetime.now().isoformat(),
                "move_analysis_enabled": run_move_analysis and self.move_analyzer is not None,
                "stockfish_path": self.stockfish_path,
                "analysis_depth": self.analysis_depth,
            },
            "matches": {},
            "overall_statistics": {},
            "move_quality_statistics": {} if run_move_analysis else None
        }
        
        all_games_data = []
        all_moves_data = []
        all_move_analyses = []
        
        # Process each match
        for match_dir in match_dirs:
            print(f"\n📊 Processing match: {match_dir.name}")
            
            match_data = self._analyze_single_match(
                match_dir, 
                run_move_analysis=run_move_analysis
            )
            
            tournament_data["matches"][match_dir.name] = match_data
            
            # Collect data for tournament-wide analysis
            if "games_data" in match_data:
                all_games_data.extend(match_data["games_data"])
            if "moves_data" in match_data:
                all_moves_data.extend(match_data["moves_data"])
            if "move_analyses" in match_data:
                all_move_analyses.extend(match_data["move_analyses"])
        
        # Generate tournament-wide statistics
        if all_games_data:
            tournament_data["overall_statistics"] = self._generate_tournament_statistics(
                all_games_data, all_moves_data
            )
        
        # Generate move quality statistics
        if all_move_analyses and run_move_analysis:
            tournament_data["move_quality_statistics"] = self._generate_move_quality_statistics(
                all_move_analyses
            )
        
        # Save tournament analysis
        self._save_tournament_analysis(tournament_dir, tournament_data)
        
        return tournament_data
    
    def _analyze_single_match(self, 
                            match_dir: Path, 
                            run_move_analysis: bool = True) -> Dict[str, Any]:
        """Analyze a single match including move quality."""
        match_data = {
            "match_id": match_dir.name,
            "basic_stats": {},
            "games_data": [],
            "moves_data": [],
            "move_analyses": []
        }
        
        # Load basic match data
        try:
            # Load games summary
            games_file = match_dir / "games_summary.csv"
            if games_file.exists():
                games_df = pd.read_csv(games_file)
                match_data["games_data"] = games_df.to_dict('records')
                match_data["basic_stats"]["total_games"] = len(games_df)
                
                # Basic game statistics
                match_data["basic_stats"]["model_a_wins"] = len(games_df[games_df["winner"] == "model a"])
                match_data["basic_stats"]["model_b_wins"] = len(games_df[games_df["winner"] == "model b"])
                match_data["basic_stats"]["draws"] = len(games_df[games_df["winner"] == "draw"])
                
        except Exception as e:
            print(f"Warning: Could not load basic match data: {e}")
        
        # Load move data
        move_files = list(match_dir.glob("game_*_moves.csv"))
        for move_file in move_files:
            try:
                moves_df = pd.read_csv(move_file)
                game_num = int(move_file.stem.split('_')[1])
                
                # Add game number to moves data
                moves_data = moves_df.to_dict('records')
                for move in moves_data:
                    move['game_number'] = game_num
                    move['match_id'] = match_dir.name
                
                match_data["moves_data"].extend(moves_data)
                
            except Exception as e:
                print(f"Warning: Could not load moves from {move_file}: {e}")
        
        # Run move quality analysis if requested and available
        if run_move_analysis and self.move_analyzer and move_files:
            try:
                print(f"  🔬 Running move quality analysis...")
                
                move_analyses = self.move_analyzer.analyze_match_directory(
                    match_dir,
                    depth=self.analysis_depth,
                    multipv=self.multipv,
                    save_results=True
                )
                
                # Flatten move analyses
                for game_analyses in move_analyses.values():
                    for analysis in game_analyses:
                        match_data["move_analyses"].append(analysis.__dict__)
                
                print(f"  ✅ Analyzed {len(match_data['move_analyses'])} moves")
                
            except Exception as e:
                print(f"  ⚠️ Move quality analysis failed: {e}")
        
        return match_data
    
    def _generate_tournament_statistics(self, 
                                      games_data: List[Dict], 
                                      moves_data: List[Dict]) -> Dict[str, Any]:
        """Generate overall tournament statistics."""
        games_df = pd.DataFrame(games_data)
        moves_df = pd.DataFrame(moves_data) if moves_data else pd.DataFrame()
        
        stats = {
            "total_games": len(games_data),
            "total_matches": games_df['match_id'].nunique() if 'match_id' in games_df.columns else 0,
            "total_moves": len(moves_data),
        }
        
        if not games_df.empty:
            # Game outcome statistics
            stats["game_outcomes"] = {
                "model_a_wins": len(games_df[games_df["winner"] == "model a"]),
                "model_b_wins": len(games_df[games_df["winner"] == "model b"]),
                "draws": len(games_df[games_df["winner"] == "draw"]),
                "errors": len(games_df[games_df["winner"] == "error"])
            }
            
            # Time management
            stats["time_management"] = {
                "avg_game_duration": games_df["game_duration_seconds"].mean(),
                "avg_moves_per_game": games_df["total_moves"].mean(),
                "avg_model_a_time_used": games_df["model_a_time_used"].mean(),
                "avg_model_b_time_used": games_df["model_b_time_used"].mean(),
            }
            
            # Performance metrics
            stats["performance"] = {
                "avg_model_a_tokens": games_df["model_a_total_tokens"].mean(),
                "avg_model_b_tokens": games_df["model_b_total_tokens"].mean(),
                "avg_model_a_parsing_failures": games_df["model_a_parsing_failures"].mean(),
                "avg_model_b_parsing_failures": games_df["model_b_parsing_failures"].mean(),
            }
        
        return stats
    
    def _generate_move_quality_statistics(self, move_analyses: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive move quality statistics."""
        if not move_analyses:
            return {}
        
        analyses_df = pd.DataFrame(move_analyses)
        
        stats = {
            "overall_quality": {
                "total_moves_analyzed": len(move_analyses),
                "average_centipawn_loss": analyses_df["centipawn_loss"].mean(),
                "median_centipawn_loss": analyses_df["centipawn_loss"].median(),
                "total_centipawn_loss": analyses_df["centipawn_loss"].sum(),
            },
            "move_classification": {
                "perfect_moves": len(analyses_df[analyses_df["centipawn_loss"] == 0]),
                "excellent_moves": len(analyses_df[(analyses_df["centipawn_loss"] > 0) & (analyses_df["centipawn_loss"] <= 10)]),
                "good_moves": len(analyses_df[(analyses_df["centipawn_loss"] > 10) & (analyses_df["centipawn_loss"] < 25)]),
                "inaccuracies": len(analyses_df[(analyses_df["centipawn_loss"] >= 25) & (analyses_df["centipawn_loss"] < 50)]),
                "mistakes": len(analyses_df[(analyses_df["centipawn_loss"] >= 50) & (analyses_df["centipawn_loss"] < 100)]),
                "blunders": len(analyses_df[analyses_df["centipawn_loss"] >= 100]),
            }
        }
        
        # Calculate percentages
        total = len(move_analyses)
        for category, count in stats["move_classification"].items():
            stats["move_classification"][f"{category}_percent"] = (count / total * 100) if total > 0 else 0
        
        # Player-specific statistics
        stats["player_analysis"] = {}
        for player in analyses_df["player"].unique():
            player_df = analyses_df[analyses_df["player"] == player]
            
            stats["player_analysis"][player.lower().replace(' ', '_')] = {
                "total_moves": len(player_df),
                "average_centipawn_loss": player_df["centipawn_loss"].mean(),
                "median_centipawn_loss": player_df["centipawn_loss"].median(),
                "blunder_rate": (len(player_df[player_df["centipawn_loss"] >= 100]) / len(player_df) * 100) if len(player_df) > 0 else 0,
                "accuracy_rate": (len(player_df[player_df["centipawn_loss"] < 25]) / len(player_df) * 100) if len(player_df) > 0 else 0,
                "perfect_move_rate": (len(player_df[player_df["centipawn_loss"] == 0]) / len(player_df) * 100) if len(player_df) > 0 else 0,
                "avg_move_rank": player_df["played_move_rank_among_top"].mean(),
            }
        
        # Game phase analysis (if move numbers are available)
        if "move_number" in analyses_df.columns:
            stats["phase_analysis"] = {
                "opening": self._analyze_phase(analyses_df, 1, 15),
                "middlegame": self._analyze_phase(analyses_df, 16, 40),
                "endgame": self._analyze_phase(analyses_df, 41, 999),
            }
        
        return stats
    
    def _analyze_phase(self, analyses_df: pd.DataFrame, min_move: int, max_move: int) -> Dict[str, float]:
        """Analyze move quality for a specific game phase."""
        phase_df = analyses_df[(analyses_df["move_number"] >= min_move) & (analyses_df["move_number"] <= max_move)]
        
        if len(phase_df) == 0:
            return {"moves": 0, "avg_cp_loss": 0, "blunder_rate": 0}
        
        return {
            "moves": len(phase_df),
            "avg_cp_loss": phase_df["centipawn_loss"].mean(),
            "blunder_rate": (len(phase_df[phase_df["centipawn_loss"] >= 100]) / len(phase_df) * 100),
        }
    
    def _save_tournament_analysis(self, tournament_dir: Path, analysis_data: Dict[str, Any]) -> None:
        """Save tournament analysis results."""
        output_file = tournament_dir / "enhanced_tournament_analysis.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        cleaned_data = convert_numpy_types(analysis_data)
        
        with open(output_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2, default=str)
        
        print(f"\n📊 Enhanced tournament analysis saved to: {output_file}")
    
    def generate_move_quality_report(self, tournament_dir: Union[str, Path]) -> str:
        """Generate a human-readable move quality report."""
        tournament_dir = Path(tournament_dir)
        analysis_file = tournament_dir / "enhanced_tournament_analysis.json"
        
        if not analysis_file.exists():
            return "No enhanced tournament analysis found. Run analyze_tournament_with_move_quality first."
        
        with open(analysis_file) as f:
            data = json.load(f)
        
        if not data.get("move_quality_statistics"):
            return "No move quality data available in the analysis."
        
        stats = data["move_quality_statistics"]
        
        report = []
        report.append("🏆 CHESS MOVE QUALITY REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Overall statistics
        overall = stats["overall_quality"]
        report.append(f"📊 Overall Statistics:")
        report.append(f"  Total moves analyzed: {overall['total_moves_analyzed']:,}")
        report.append(f"  Average centipawn loss: {overall['average_centipawn_loss']:.1f}")
        report.append(f"  Median centipawn loss: {overall['median_centipawn_loss']:.1f}")
        report.append("")
        
        # Move classification
        classification = stats["move_classification"]
        report.append(f"🎯 Move Quality Distribution:")
        report.append(f"  Perfect moves (0 cp):      {classification['perfect_moves']:4d} ({classification['perfect_moves_percent']:.1f}%)")
        report.append(f"  Excellent moves (1-10 cp): {classification['excellent_moves']:4d} ({classification['excellent_moves_percent']:.1f}%)")
        report.append(f"  Good moves (11-24 cp):     {classification['good_moves']:4d} ({classification['good_moves_percent']:.1f}%)")
        report.append(f"  Inaccuracies (25-49 cp):   {classification['inaccuracies']:4d} ({classification['inaccuracies_percent']:.1f}%)")
        report.append(f"  Mistakes (50-99 cp):       {classification['mistakes']:4d} ({classification['mistakes_percent']:.1f}%)")
        report.append(f"  Blunders (100+ cp):        {classification['blunders']:4d} ({classification['blunders_percent']:.1f}%)")
        report.append("")
        
        # Player comparison
        if "player_analysis" in stats:
            report.append(f"👥 Player Comparison:")
            for player_key, player_stats in stats["player_analysis"].items():
                player_name = player_key.replace('_', ' ').title()
                report.append(f"  {player_name}:")
                report.append(f"    Average centipawn loss: {player_stats['average_centipawn_loss']:.1f}")
                report.append(f"    Accuracy rate (< 25 cp): {player_stats['accuracy_rate']:.1f}%")
                report.append(f"    Blunder rate (100+ cp): {player_stats['blunder_rate']:.1f}%")
                report.append(f"    Perfect move rate: {player_stats['perfect_move_rate']:.1f}%")
                report.append("")
        
        # Phase analysis
        if "phase_analysis" in stats:
            report.append(f"🏁 Game Phase Analysis:")
            for phase, phase_stats in stats["phase_analysis"].items():
                if phase_stats["moves"] > 0:
                    report.append(f"  {phase.title()}:")
                    report.append(f"    Moves: {phase_stats['moves']}")
                    report.append(f"    Avg centipawn loss: {phase_stats['avg_cp_loss']:.1f}")
                    report.append(f"    Blunder rate: {phase_stats['blunder_rate']:.1f}%")
            report.append("")
        
        return "\n".join(report)


def main():
    """CLI interface for enhanced tournament analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced tournament analysis with move quality")
    parser.add_argument("tournament_dir", help="Path to tournament directory")
    parser.add_argument("--stockfish-path", help="Path to Stockfish executable")
    parser.add_argument("--depth", type=int, default=15, help="Analysis depth (default: 15)")
    parser.add_argument("--multipv", type=int, default=3, help="Principal variations (default: 3)")
    parser.add_argument("--skip-move-analysis", action="store_true", help="Skip Stockfish move analysis")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing analysis")
    
    args = parser.parse_args()
    
    analyzer = EnhancedTournamentAnalyzer(
        stockfish_path=args.stockfish_path,
        analysis_depth=args.depth,
        multipv=args.multipv
    )
    
    if args.report_only:
        print(analyzer.generate_move_quality_report(args.tournament_dir))
    else:
        analyzer.analyze_tournament_with_move_quality(
            args.tournament_dir,
            run_move_analysis=not args.skip_move_analysis
        )
        print(analyzer.generate_move_quality_report(args.tournament_dir))


if __name__ == "__main__":
    main()
