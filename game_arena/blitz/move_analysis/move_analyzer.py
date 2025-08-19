#!/usr/bin/env python3
"""Move quality analysis using Stockfish engine for blitz chess games.

This module analyzes the quality of moves made by LLM agents during chess games,
calculating centipawn loss and handling mate scores correctly.
"""

import chess
import chess.engine
import csv
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import dataclasses
from dataclasses import asdict
import os

# Configuration
DEFAULT_ENGINE_PATH = "/opt/homebrew/bin/stockfish"  # Default for macOS with Homebrew
WINDOWS_ENGINE_PATH = "C:\\engines\\stockfish.exe"
LINUX_ENGINE_PATH = "/usr/local/bin/stockfish"

def get_default_engine_path() -> str:
    """Get the default Stockfish path based on the operating system."""
    import platform
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        # Try common paths
        possible_paths = [
            "/opt/homebrew/bin/stockfish",  # Homebrew on Apple Silicon
            "/usr/local/bin/stockfish",     # Homebrew on Intel
            "/usr/bin/stockfish"            # System installation
        ]
    elif system == "linux":
        possible_paths = [
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "/usr/games/stockfish"
        ]
    elif system == "windows":
        possible_paths = [
            "C:\\engines\\stockfish.exe",
            "C:\\Program Files\\stockfish\\stockfish.exe"
        ]
    else:
        possible_paths = ["/usr/local/bin/stockfish"]
    
    # Check if any of the paths exist
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Return the first path as default if none exist
    return possible_paths[0] if possible_paths else "/usr/local/bin/stockfish"


@dataclasses.dataclass
class MoveAnalysis:
    """Analysis result for a single chess move."""
    # Game identification
    match_id: str
    game_number: int
    move_number: int
    color: str  # "white" or "black"
    player: str  # "Model A" or "Model B"
    
    # Move information
    move_played: str
    board_fen_before: str
    
    # Engine analysis
    best_move_uci: Optional[str]
    best_move_san: Optional[str]
    best_eval_cp_from_player_pov: int
    played_eval_cp_from_player_pov: int
    centipawn_loss: int
    played_move_rank_among_top: Optional[int]
    
    # Win probability analysis
    best_win_probability: float  # Win probability after best move (0.0-1.0)
    played_win_probability: float  # Win probability after played move (0.0-1.0)
    win_probability_loss: float  # Difference in win probability (0.0-1.0)
    
    # Human-readable strings
    best_eval_str: str
    played_eval_str: str
    cp_loss_str: str
    best_win_prob_str: str  # e.g., "65.2%"
    played_win_prob_str: str  # e.g., "58.1%"
    win_prob_loss_str: str  # e.g., "-7.1%"
    
    # Engine parameters used
    engine_depth: int
    multipv: int


class MoveQualityAnalyzer:
    """Analyzes chess move quality using Stockfish engine."""
    
    def __init__(self, 
                 engine_path: Optional[str] = None,
                 default_depth: int = 18,
                 default_multipv: int = 5,
                 threads: int = 8,
                 hash_mb: int = 512,
                 enable_wdl: bool = True):
        """Initialize the move analyzer.
        
        Args:
            engine_path: Path to Stockfish executable. If None, tries to auto-detect.
            default_depth: Default search depth
            default_multipv: Default number of principal variations to analyze
            threads: Number of threads for Stockfish
            hash_mb: Hash table size in MB
            enable_wdl: Whether to enable WDL (Win-Draw-Loss) statistics if available
        """
        self.engine_path = engine_path or get_default_engine_path()
        self.default_depth = default_depth
        self.default_multipv = default_multipv
        self.threads = threads
        self.hash_mb = hash_mb
        self.enable_wdl = enable_wdl
        
        # Verify engine is accessible
        self._verify_engine()
    
    def _verify_engine(self) -> None:
        """Verify that the Stockfish engine is accessible."""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(
                f"Stockfish engine not found at {self.engine_path}. "
                f"Please install Stockfish and set the correct path."
            )
        
        # Test engine startup
        try:
            with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
                pass  # Engine started successfully
        except Exception as e:
            raise RuntimeError(f"Failed to start Stockfish engine: {e}")
    
    def score_to_cp(self, score: chess.engine.Score, pov_color: chess.Color, mate_cp: int = 100000) -> int:
        """Convert python-chess Score to a single CP number from pov_color's perspective.
        
        Args:
            score: The engine score
            pov_color: Color from whose perspective to evaluate
            mate_cp: Centipawn value to assign for mate positions
            
        Returns:
            Centipawn score from pov_color's perspective
        """
        s = score.pov(pov_color)
        if s.is_mate():
            m = s.mate()  # +N means mate in N for pov_color, -N means getting mated
            return mate_cp if m > 0 else -mate_cp
        return s.score()  # centipawns
    
    def cp_to_win_probability(self, cp_score: int) -> float:
        """Convert centipawn score to win probability using standard logistic formula.
        
        This uses the widely-adopted formula used by chess.com, lichess, and other platforms:
        win_probability = 1 / (1 + 10^(-cp/400))
        
        Args:
            cp_score: Centipawn score from player's perspective
            
        Returns:
            Win probability (0.0 to 1.0) for the player
        """
        if abs(cp_score) >= 99000:  # Mate scores
            return 1.0 if cp_score > 0 else 0.0
        
        # Standard logistic conversion validated across millions of games
        return 1.0 / (1.0 + 10**(-cp_score / 400.0))
    
    def extract_wdl_from_info(self, info: Dict) -> Optional[Tuple[float, float, float]]:
        """Extract Win-Draw-Loss probabilities from engine info if available.
        
        Args:
            info: Engine analysis info dictionary
            
        Returns:
            Tuple of (win_prob, draw_prob, loss_prob) or None if not available
        """
        # Check if WDL data is available in the analysis info
        # This depends on engine configuration and version
        if 'wdl' in info:
            wdl = info['wdl']
            if isinstance(wdl, (list, tuple)) and len(wdl) >= 3:
                # WDL values are typically in permille (out of 1000)
                win_permille, draw_permille, loss_permille = wdl[:3]
                total = win_permille + draw_permille + loss_permille
                if total > 0:
                    return (
                        win_permille / total,
                        draw_permille / total, 
                        loss_permille / total
                    )
        return None
    
    def evaluate_move(self, 
                     fen: str, 
                     move_str: str, 
                     depth: Optional[int] = None, 
                     multipv: Optional[int] = None) -> Dict[str, Any]:
        """Analyze how good a move is compared to the engine's best choice.
        
        Args:
            fen: Board position in FEN notation
            move_str: Move to analyze (UCI format like 'e2e4' or SAN like 'e4')
            depth: Search depth (uses default if None)
            multipv: Number of principal variations (uses default if None)
            
        Returns:
            Dictionary with analysis results
        """
        if depth is None:
            depth = self.default_depth
        if multipv is None:
            multipv = self.default_multipv
            
        board = chess.Board(fen)
        
        # Parse move (accepts UCI 'e2e4' or SAN 'e4')
        try:
            move = board.parse_uci(move_str)
        except ValueError:
            try:
                move = board.parse_san(move_str)
            except ValueError:
                raise ValueError(f"Unable to parse move '{move_str}' for position {fen}")
        
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move '{move_str}' for position {fen}")
        
        color = board.turn  # side making the move
        
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            # Configure engine
            config = {"Threads": self.threads, "Hash": self.hash_mb}
            
            # Try to enable WDL (Win-Draw-Loss) statistics if supported
            if self.enable_wdl:
                try:
                    config["UCI_ShowWDL"] = True
                except:
                    pass  # Not all engines support this
            
            engine.configure(config)
            
            # 1) Engine's evaluation of the current position (best line)
            infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
            # python-chess returns a list when multipv>1; normalize
            infos = infos if isinstance(infos, list) else [infos]
            best_info = infos[0]
            best_move = best_info["pv"][0] if "pv" in best_info and best_info["pv"] else None
            best_cp = self.score_to_cp(best_info["score"], color)
            
            # 2) Evaluation after *our* move (from same player's POV)
            b2 = board.copy()
            b2.push(move)
            played_info = engine.analyse(b2, chess.engine.Limit(depth=depth))
            played_cp = self.score_to_cp(played_info["score"], color)
            
            # 3) Centipawn loss (how much worse than the engine's top choice)
            cp_loss = best_cp - played_cp  # >0 means your move is worse than best
            
            # 4) Win probability analysis
            # Try to extract WDL probabilities first, fallback to centipawn conversion
            best_wdl = self.extract_wdl_from_info(best_info)
            played_wdl = self.extract_wdl_from_info(played_info)
            
            if best_wdl and played_wdl:
                # Use direct WDL probabilities (more accurate)
                best_win_prob = best_wdl[0]  # Win probability
                played_win_prob = played_wdl[0]  # Win probability
            else:
                # Fallback to centipawn conversion
                best_win_prob = self.cp_to_win_probability(best_cp)
                played_win_prob = self.cp_to_win_probability(played_cp)
            
            win_prob_loss = best_win_prob - played_win_prob  # >0 means move lost win probability
            
            # Optional: rank of your move among top N
            rank = None
            if best_move is not None:
                top_moves = [i["pv"][0] for i in infos if "pv" in i and i["pv"]]
                for i, m in enumerate(top_moves, 1):
                    if m == move:
                        rank = i
                        break
            
            # Human-friendly strings
            def fmt_score(score_cp):
                if abs(score_cp) >= 99999:
                    return "MATE (for you)" if score_cp > 0 else "MATE (against you)"
                return f"{score_cp/100:.2f}"
            
            def fmt_win_prob(prob):
                return f"{prob*100:.1f}%"
            
            def fmt_win_prob_loss(loss):
                return f"{loss*100:+.1f}%" if loss != 0 else "0.0%"
            
            return {
                "best_move_uci": best_move.uci() if best_move else None,
                "best_eval_cp_from_player_pov": best_cp,
                "played_eval_cp_from_player_pov": played_cp,
                "centipawn_loss": cp_loss,
                "played_move_rank_among_top": rank,  # 1 means it matched the engine's top choice
                "best_win_probability": best_win_prob,
                "played_win_probability": played_win_prob,
                "win_probability_loss": win_prob_loss,
                "pretty": {
                    "best_move_san": board.san(best_move) if best_move else None,
                    "best_eval_str": fmt_score(best_cp),
                    "played_eval_str": fmt_score(played_cp),
                    "cp_loss_str": f"{cp_loss} cp",
                    "best_win_prob_str": fmt_win_prob(best_win_prob),
                    "played_win_prob_str": fmt_win_prob(played_win_prob),
                    "win_prob_loss_str": fmt_win_prob_loss(win_prob_loss),
                },
            }
    
    def analyze_game_moves(self, 
                          moves_df: pd.DataFrame, 
                          match_id: str, 
                          game_number: int,
                          depth: Optional[int] = None,
                          multipv: Optional[int] = None) -> List[MoveAnalysis]:
        """Analyze all moves in a game.
        
        Args:
            moves_df: DataFrame with move data (from game_X_moves.csv)
            match_id: Match identifier
            game_number: Game number
            depth: Search depth (uses default if None)
            multipv: Number of principal variations (uses default if None)
            
        Returns:
            List of MoveAnalysis objects
        """
        if depth is None:
            depth = self.default_depth
        if multipv is None:
            multipv = self.default_multipv
            
        analyses = []
        
        for _, row in moves_df.iterrows():
            try:
                result = self.evaluate_move(
                    row['board_state_before_move'],
                    row['move_played'],
                    depth=depth,
                    multipv=multipv
                )
                
                analysis = MoveAnalysis(
                    match_id=match_id,
                    game_number=game_number,
                    move_number=row['move_number'],
                    color=row['color'],
                    player=row['who_played'],
                    move_played=row['move_played'],
                    board_fen_before=row['board_state_before_move'],
                    best_move_uci=result['best_move_uci'],
                    best_move_san=result['pretty']['best_move_san'],
                    best_eval_cp_from_player_pov=result['best_eval_cp_from_player_pov'],
                    played_eval_cp_from_player_pov=result['played_eval_cp_from_player_pov'],
                    centipawn_loss=result['centipawn_loss'],
                    played_move_rank_among_top=result['played_move_rank_among_top'],
                    best_win_probability=result['best_win_probability'],
                    played_win_probability=result['played_win_probability'],
                    win_probability_loss=result['win_probability_loss'],
                    best_eval_str=result['pretty']['best_eval_str'],
                    played_eval_str=result['pretty']['played_eval_str'],
                    cp_loss_str=result['pretty']['cp_loss_str'],
                    best_win_prob_str=result['pretty']['best_win_prob_str'],
                    played_win_prob_str=result['pretty']['played_win_prob_str'],
                    win_prob_loss_str=result['pretty']['win_prob_loss_str'],
                    engine_depth=depth,
                    multipv=multipv
                )
                
                analyses.append(analysis)
                
                # Progress feedback
                if len(analyses) % 10 == 0:
                    print(f"Analyzed {len(analyses)} moves...")
                
            except Exception as e:
                print(f"Warning: Failed to analyze move {row['move_number']} ({row['move_played']}): {e}")
                continue
        
        return analyses
    
    def analyze_match_directory(self,
                               match_dir: Union[str, Path],
                               depth: Optional[int] = None,
                               multipv: Optional[int] = None,
                               save_results: bool = True) -> Dict[int, List[MoveAnalysis]]:
        """Analyze all games in a match directory.
        
        Args:
            match_dir: Path to match directory containing game move files
            depth: Search depth (uses default if None)
            multipv: Number of principal variations (uses default if None)
            save_results: Whether to save results to CSV files
            
        Returns:
            Dictionary mapping game numbers to their move analyses
        """
        match_dir = Path(match_dir)
        match_id = match_dir.name
        
        # Find all game move files
        move_files = list(match_dir.glob("game_*_moves.csv"))
        if not move_files:
            raise ValueError(f"No game move files found in {match_dir}")
        
        print(f"Found {len(move_files)} games to analyze in {match_id}")
        
        all_analyses = {}
        
        for move_file in sorted(move_files):
            # Extract game number from filename
            game_num = int(move_file.stem.split('_')[1])
            
            print(f"\nAnalyzing Game {game_num}...")
            
            # Load move data
            moves_df = pd.read_csv(move_file)
            
            # Analyze moves
            analyses = self.analyze_game_moves(
                moves_df, match_id, game_num, depth=depth, multipv=multipv
            )
            
            all_analyses[game_num] = analyses
            
            print(f"Game {game_num}: Analyzed {len(analyses)} moves")
            
            # Save individual game analysis
            if save_results:
                self._save_game_analysis(match_dir, game_num, analyses)
        
        # Save combined analysis
        if save_results:
            self._save_match_analysis(match_dir, all_analyses, depth or self.default_depth, multipv or self.default_multipv)
        
        return all_analyses
    
    def _save_game_analysis(self, match_dir: Path, game_num: int, analyses: List[MoveAnalysis]) -> None:
        """Save analysis results for a single game."""
        if not analyses:
            return
            
        # Convert to DataFrame and save as CSV
        analysis_dicts = [asdict(analysis) for analysis in analyses]
        df = pd.DataFrame(analysis_dicts)
        
        output_file = match_dir / f"game_{game_num}_move_analysis.csv"
        df.to_csv(output_file, index=False)
        print(f"  Saved analysis to {output_file}")
    
    def _save_match_analysis(self, 
                           match_dir: Path, 
                           all_analyses: Dict[int, List[MoveAnalysis]], 
                           depth: int, 
                           multipv: int) -> None:
        """Save combined analysis results for the entire match."""
        # Flatten all analyses
        all_moves = []
        for game_analyses in all_analyses.values():
            all_moves.extend(game_analyses)
        
        if not all_moves:
            return
        
        # Convert to DataFrame
        analysis_dicts = [asdict(analysis) for analysis in all_moves]
        df = pd.DataFrame(analysis_dicts)
        
        # Save complete analysis
        output_file = match_dir / "complete_move_analysis.csv"
        df.to_csv(output_file, index=False)
        
        # Generate summary statistics
        summary = self._generate_analysis_summary(all_analyses)
        summary_file = match_dir / "move_analysis_summary.json"
        
        summary["analysis_parameters"] = {
            "engine_path": self.engine_path,
            "depth": depth,
            "multipv": multipv,
            "threads": self.threads,
            "hash_mb": self.hash_mb
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_summary = convert_numpy_types(summary)
        
        with open(summary_file, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
        print(f"\n📊 Complete analysis saved:")
        print(f"  - Move details: {output_file}")
        print(f"  - Summary: {summary_file}")
    
    def _generate_analysis_summary(self, all_analyses: Dict[int, List[MoveAnalysis]]) -> Dict[str, Any]:
        """Generate summary statistics from move analyses."""
        # Flatten all moves
        all_moves = []
        for game_analyses in all_analyses.values():
            all_moves.extend(game_analyses)
        
        if not all_moves:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([asdict(analysis) for analysis in all_moves])
        
        # Overall statistics
        summary = {
            "total_moves_analyzed": len(all_moves),
            "total_games": len(all_analyses),
            "overall_stats": {
                "average_centipawn_loss": df['centipawn_loss'].mean(),
                "median_centipawn_loss": df['centipawn_loss'].median(),
                "total_centipawn_loss": df['centipawn_loss'].sum(),
                "moves_with_zero_loss": len(df[df['centipawn_loss'] == 0]),
                "blunders_100cp_plus": len(df[df['centipawn_loss'] >= 100]),
                "mistakes_50_99cp": len(df[(df['centipawn_loss'] >= 50) & (df['centipawn_loss'] < 100)]),
                "inaccuracies_25_49cp": len(df[(df['centipawn_loss'] >= 25) & (df['centipawn_loss'] < 50)]),
            }
        }
        
        # Player-specific statistics
        for player in df['player'].unique():
            player_df = df[df['player'] == player]
            summary[f"{player.lower().replace(' ', '_')}_stats"] = {
                "total_moves": len(player_df),
                "average_centipawn_loss": player_df['centipawn_loss'].mean(),
                "median_centipawn_loss": player_df['centipawn_loss'].median(),
                "total_centipawn_loss": player_df['centipawn_loss'].sum(),
                "moves_with_zero_loss": len(player_df[player_df['centipawn_loss'] == 0]),
                "blunders_100cp_plus": len(player_df[player_df['centipawn_loss'] >= 100]),
                "mistakes_50_99cp": len(player_df[(player_df['centipawn_loss'] >= 50) & (player_df['centipawn_loss'] < 100)]),
                "inaccuracies_25_49cp": len(player_df[(player_df['centipawn_loss'] >= 25) & (player_df['centipawn_loss'] < 50)]),
                "perfect_moves_percent": (len(player_df[player_df['centipawn_loss'] == 0]) / len(player_df)) * 100,
                "top_3_moves_percent": (len(player_df[player_df['played_move_rank_among_top'].fillna(999) <= 3]) / len(player_df)) * 100,
            }
        
        # Game-by-game statistics
        summary["per_game_stats"] = {}
        for game_num, game_analyses in all_analyses.items():
            if game_analyses:
                game_df = pd.DataFrame([asdict(analysis) for analysis in game_analyses])
                summary["per_game_stats"][f"game_{game_num}"] = {
                    "total_moves": len(game_analyses),
                    "average_centipawn_loss": game_df['centipawn_loss'].mean(),
                    "total_centipawn_loss": game_df['centipawn_loss'].sum(),
                    "blunders": len(game_df[game_df['centipawn_loss'] >= 100]),
                    "mistakes": len(game_df[(game_df['centipawn_loss'] >= 50) & (game_df['centipawn_loss'] < 100)]),
                }
        
        return summary


def main():
    """Example usage and CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze chess move quality using Stockfish")
    parser.add_argument("match_dir", help="Path to match directory containing game files")
    parser.add_argument("--engine-path", help="Path to Stockfish executable")
    parser.add_argument("--depth", type=int, default=18, help="Search depth (default: 18)")
    parser.add_argument("--multipv", type=int, default=5, help="Number of principal variations (default: 5)")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (default: 8)")
    parser.add_argument("--hash", type=int, default=512, help="Hash table size in MB (default: 512)")
    
    args = parser.parse_args()
    
    try:
        analyzer = MoveQualityAnalyzer(
            engine_path=args.engine_path,
            default_depth=args.depth,
            default_multipv=args.multipv,
            threads=args.threads,
            hash_mb=args.hash
        )
        
        print(f"Using Stockfish at: {analyzer.engine_path}")
        print(f"Analysis parameters: depth={args.depth}, multipv={args.multipv}")
        
        results = analyzer.analyze_match_directory(args.match_dir, depth=args.depth, multipv=args.multipv)
        
        print(f"\n✅ Analysis complete! Analyzed {len(results)} games.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
