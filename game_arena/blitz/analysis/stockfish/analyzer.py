#!/usr/bin/env python3
"""Move quality analysis using Stockfish engine for blitz chess games."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import pandas as pd

from game_arena.blitz.analysis.stockfish.types import MoveAnalysis


def get_default_engine_path() -> str:
    """Get the default Stockfish path based on the operating system."""
    import platform
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
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
    
    return possible_paths[0] if possible_paths else "/usr/local/bin/stockfish"


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
        
        try:
            with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
                pass
        except Exception as e:
            raise RuntimeError(f"Failed to start Stockfish engine: {e}")
    
    def score_to_cp(self, score: chess.engine.Score, pov_color: chess.Color, mate_cp: int = 100000) -> int:
        """Convert python-chess Score to a single CP number from pov_color's perspective."""
        s = score.pov(pov_color)
        if s.is_mate():
            m = s.mate()
            return mate_cp if m > 0 else -mate_cp
        return s.score()
    
    def cp_to_win_probability(self, cp_score: int) -> float:
        """Convert centipawn score to win probability using standard logistic formula."""
        if abs(cp_score) >= 99000:
            return 1.0 if cp_score > 0 else 0.0
        return 1.0 / (1.0 + 10**(-cp_score / 400.0))
    
    def extract_wdl_from_info(self, info: Dict) -> Optional[Tuple[float, float, float]]:
        """Extract Win-Draw-Loss probabilities from engine info if available."""
        if 'wdl' in info:
            wdl = info['wdl']
            if isinstance(wdl, (list, tuple)) and len(wdl) >= 3:
                win_permille, draw_permille, loss_permille = wdl[:3]
                total = win_permille + draw_permille + loss_permille
                if total > 0:
                    return (win_permille / total, draw_permille / total, loss_permille / total)
        return None
    
    def evaluate_move(self, 
                     fen: str, 
                     move_str: str, 
                     depth: Optional[int] = None, 
                     multipv: Optional[int] = None) -> Dict[str, Any]:
        """Analyze how good a move is compared to the engine's best choice."""
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
        
        color = board.turn
        
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            config = {"Threads": self.threads, "Hash": self.hash_mb}
            
            if self.enable_wdl:
                try:
                    config["UCI_ShowWDL"] = True
                except:
                    pass
            
            engine.configure(config)
            
            # Engine's evaluation of the current position
            infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
            infos = infos if isinstance(infos, list) else [infos]
            best_info = infos[0]
            best_move = best_info["pv"][0] if "pv" in best_info and best_info["pv"] else None
            best_cp = self.score_to_cp(best_info["score"], color)
            
            # Evaluation after our move
            b2 = board.copy()
            b2.push(move)
            played_info = engine.analyse(b2, chess.engine.Limit(depth=depth))
            played_cp = self.score_to_cp(played_info["score"], color)
            
            # Centipawn loss
            cp_loss = best_cp - played_cp
            
            # Win probability analysis
            best_wdl = self.extract_wdl_from_info(best_info)
            played_wdl = self.extract_wdl_from_info(played_info)
            
            if best_wdl and played_wdl:
                best_win_prob = best_wdl[0]
                played_win_prob = played_wdl[0]
            else:
                best_win_prob = self.cp_to_win_probability(best_cp)
                played_win_prob = self.cp_to_win_probability(played_cp)
            
            win_prob_loss = best_win_prob - played_win_prob
            
            # Rank of your move among top N
            rank = None
            if best_move is not None:
                top_moves = [i["pv"][0] for i in infos if "pv" in i and i["pv"]]
                for i, m in enumerate(top_moves, 1):
                    if m == move:
                        rank = i
                        break
            
            # Position complexity metrics
            num_legal_moves = len(list(board.legal_moves))
            
            # Eval sharpness: difference between best and 2nd best move
            eval_sharpness = 0
            if len(infos) >= 2:
                second_cp = self.score_to_cp(infos[1]["score"], color)
                eval_sharpness = abs(best_cp - second_cp)
            
            # Absolute position evaluation (0 = equal, higher = more decisive)
            position_eval_abs = abs(best_cp)
            
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
                "played_move_rank_among_top": rank,
                "best_win_probability": best_win_prob,
                "played_win_probability": played_win_prob,
                "win_probability_loss": win_prob_loss,
                # Position complexity metrics
                "num_legal_moves": num_legal_moves,
                "eval_sharpness": eval_sharpness,
                "position_eval_abs": position_eval_abs,
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
        """Analyze all moves in a game."""
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
                    num_legal_moves=result['num_legal_moves'],
                    eval_sharpness=result['eval_sharpness'],
                    position_eval_abs=result['position_eval_abs'],
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
                
                if len(analyses) % 10 == 0:
                    print(f"Analyzed {len(analyses)} moves...")
                
            except Exception as e:
                print(f"Warning: Failed to analyze move {row['move_number']} ({row['move_played']}): {e}")
                continue
        
        return analyses
    
    def analyze_game_file(self,
                         game_file: Union[str, Path],
                         depth: Optional[int] = None,
                         multipv: Optional[int] = None) -> List[MoveAnalysis]:
        """Analyze all moves in a single game file.
        
        Args:
            game_file: Path to game_N_moves.csv file
            depth: Stockfish search depth
            multipv: Number of principal variations
            
        Returns:
            List of MoveAnalysis objects for each move in the game
        """
        game_file = Path(game_file)
        match_dir = game_file.parent
        match_id = match_dir.name
        
        # Extract game number from filename (e.g., "game_1_moves.csv" -> 1)
        game_num = int(game_file.stem.split('_')[1])
        
        moves_df = pd.read_csv(game_file)
        return self.analyze_game_moves(
            moves_df, match_id, game_num, depth=depth, multipv=multipv
        )
    
    def save_game_analysis(self, analyses: List[MoveAnalysis], output_file: Union[str, Path]) -> None:
        """Save analysis results for a single game to a CSV file.
        
        Args:
            analyses: List of MoveAnalysis objects
            output_file: Path where to save the CSV file
        """
        if not analyses:
            return
            
        analysis_dicts = [asdict(analysis) for analysis in analyses]
        df = pd.DataFrame(analysis_dicts)
        df.to_csv(output_file, index=False)

    def analyze_match_directory(self,
                               match_dir: Union[str, Path],
                               depth: Optional[int] = None,
                               multipv: Optional[int] = None,
                               save_results: bool = True) -> Dict[int, List[MoveAnalysis]]:
        """Analyze all games in a match directory."""
        match_dir = Path(match_dir)
        match_id = match_dir.name
        
        move_files = list(match_dir.glob("game_*_moves.csv"))
        if not move_files:
            raise ValueError(f"No game move files found in {match_dir}")
        
        print(f"Found {len(move_files)} games to analyze in {match_id}")
        
        all_analyses = {}
        
        for move_file in sorted(move_files):
            game_num = int(move_file.stem.split('_')[1])
            print(f"\nAnalyzing Game {game_num}...")
            
            moves_df = pd.read_csv(move_file)
            analyses = self.analyze_game_moves(
                moves_df, match_id, game_num, depth=depth, multipv=multipv
            )
            
            all_analyses[game_num] = analyses
            print(f"Game {game_num}: Analyzed {len(analyses)} moves")
            
            if save_results:
                self._save_game_analysis(match_dir, game_num, analyses)
        
        if save_results:
            self._save_match_analysis(match_dir, all_analyses, depth or self.default_depth, multipv or self.default_multipv)
        
        return all_analyses
    
    def _save_game_analysis(self, match_dir: Path, game_num: int, analyses: List[MoveAnalysis]) -> None:
        """Save analysis results for a single game."""
        if not analyses:
            return
            
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
        all_moves = []
        for game_analyses in all_analyses.values():
            all_moves.extend(game_analyses)
        
        if not all_moves:
            return
        
        analysis_dicts = [asdict(analysis) for analysis in all_moves]
        df = pd.DataFrame(analysis_dicts)
        
        output_file = match_dir / "complete_move_analysis.csv"
        df.to_csv(output_file, index=False)
        
        summary = self._generate_analysis_summary(all_analyses)
        summary["analysis_parameters"] = {
            "engine_path": self.engine_path,
            "depth": depth,
            "multipv": multipv,
            "threads": self.threads,
            "hash_mb": self.hash_mb
        }
        
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_summary = convert_numpy_types(summary)
        
        summary_file = match_dir / "move_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
        print(f"\n📊 Complete analysis saved:")
        print(f"  - Move details: {output_file}")
        print(f"  - Summary: {summary_file}")
    
    def _generate_analysis_summary(self, all_analyses: Dict[int, List[MoveAnalysis]]) -> Dict[str, Any]:
        """Generate summary statistics from move analyses."""
        all_moves = []
        for game_analyses in all_analyses.values():
            all_moves.extend(game_analyses)
        
        if not all_moves:
            return {}
        
        df = pd.DataFrame([asdict(analysis) for analysis in all_moves])
        
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
        
        for player in df['player'].unique():
            player_df = df[df['player'] == player]
            summary[f"{player.lower().replace(' ', '_')}_stats"] = {
                "total_moves": len(player_df),
                "average_centipawn_loss": player_df['centipawn_loss'].mean(),
                "median_centipawn_loss": player_df['centipawn_loss'].median(),
                "total_centipawn_loss": player_df['centipawn_loss'].sum(),
                "moves_with_zero_loss": len(player_df[player_df['centipawn_loss'] == 0]),
                "blunders_100cp_plus": len(player_df[player_df['centipawn_loss'] >= 100]),
                "perfect_moves_percent": (len(player_df[player_df['centipawn_loss'] == 0]) / len(player_df)) * 100,
            }
        
        return summary


def main():
    """CLI interface for move analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze chess move quality using Stockfish")
    parser.add_argument("match_dir", help="Path to match directory containing game files")
    parser.add_argument("--engine-path", help="Path to Stockfish executable")
    parser.add_argument("--depth", type=int, default=18, help="Search depth (default: 18)")
    parser.add_argument("--multipv", type=int, default=5, help="Number of principal variations (default: 5)")
    
    args = parser.parse_args()
    
    try:
        analyzer = MoveQualityAnalyzer(
            engine_path=args.engine_path,
            default_depth=args.depth,
            default_multipv=args.multipv,
        )
        
        print(f"Using Stockfish at: {analyzer.engine_path}")
        results = analyzer.analyze_match_directory(args.match_dir)
        print(f"\n✅ Analysis complete! Analyzed {len(results)} games.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

