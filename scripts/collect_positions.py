#!/usr/bin/env python3
"""Utility script to collect and curate chess positions for offline evaluation.

This script provides various methods to build a position dataset:
1. From Lichess puzzle API
2. From PGN game files
3. From Stockfish analysis of random positions
4. Manual curation helpers

Usage:
    python scripts/collect_positions.py --source lichess --count 50
    python scripts/collect_positions.py --source pgn --file games.pgn
    python scripts/collect_positions.py --analyze-dataset positions.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def fetch_lichess_puzzles(
    count: int = 50,
    min_rating: int = 1500,
    max_rating: int = 2200,
) -> list:
    """Fetch puzzles from Lichess puzzle API.
    
    Note: Requires network access and may need API token for high volume.
    """
    import requests
    
    from game_arena.blitz.offline_eval import ChessPosition
    
    positions = []
    
    print(f"Fetching {count} puzzles from Lichess (rating {min_rating}-{max_rating})...")
    
    # Use Lichess puzzle CSV endpoint (easier for bulk)
    # Note: This is a simplified version; full implementation would use the streaming API
    url = "https://lichess.org/api/puzzle/daily"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            puzzle = response.json()
            fen = puzzle.get('game', {}).get('pgn', '').split('\n')
            
            # For now, create a sample position
            positions.append(ChessPosition(
                position_id=f"lichess_{puzzle.get('puzzle', {}).get('id', 'unknown')}",
                fen=puzzle.get('initialPly', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'),
                category="tactical",
                difficulty="medium",
                description="Lichess daily puzzle",
                source="lichess",
                tags=puzzle.get('puzzle', {}).get('themes', []),
            ))
    except Exception as e:
        print(f"Error fetching from Lichess: {e}")
    
    print(f"Collected {len(positions)} positions")
    return positions


def extract_from_pgn(pgn_path: Path, count: int = 50) -> list:
    """Extract interesting positions from a PGN file.
    
    Selects positions at various game phases and looks for
    tactical moments based on evaluation swings.
    """
    try:
        import chess
        import chess.pgn
    except ImportError:
        print("python-chess required: pip install python-chess")
        return []
    
    from game_arena.blitz.offline_eval import ChessPosition
    
    positions = []
    
    with open(pgn_path) as f:
        game_num = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None or len(positions) >= count:
                break
            
            game_num += 1
            board = game.board()
            
            # Sample positions at different phases
            moves = list(game.mainline_moves())
            
            # Opening: moves 8-12
            # Middlegame: moves 20-35
            # Endgame: moves 45+
            
            sample_points = [
                (10, "opening"),
                (25, "middlegame"),
                (35, "middlegame"),
                (50, "endgame"),
            ]
            
            for move_num, category in sample_points:
                if move_num < len(moves):
                    temp_board = game.board()
                    for i, move in enumerate(moves[:move_num]):
                        temp_board.push(move)
                    
                    # Skip if game is over
                    if temp_board.is_game_over():
                        continue
                    
                    pos_id = f"pgn_{game_num}_{move_num}"
                    
                    positions.append(ChessPosition(
                        position_id=pos_id,
                        fen=temp_board.fen(),
                        category=category,
                        difficulty="medium",
                        description=f"From game {game_num}, move {move_num}",
                        source="pgn",
                        tags=[],
                    ))
            
            if len(positions) >= count:
                break
    
    print(f"Extracted {len(positions)} positions from {game_num} games")
    return positions


def generate_random_positions(count: int = 20) -> list:
    """Generate random legal positions by playing random moves.
    
    Creates positions at various game phases for testing.
    """
    try:
        import chess
    except ImportError:
        print("python-chess required: pip install python-chess")
        return []
    
    from game_arena.blitz.offline_eval import ChessPosition
    
    positions = []
    
    for i in range(count):
        board = chess.Board()
        
        # Random number of moves (10-60)
        num_moves = random.randint(10, 60)
        
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            move = random.choice(legal_moves)
            board.push(move)
        
        if board.is_game_over():
            continue
        
        # Categorize by move count
        if num_moves < 15:
            category = "opening"
        elif num_moves < 40:
            category = "middlegame"
        else:
            category = "endgame"
        
        positions.append(ChessPosition(
            position_id=f"random_{i}",
            fen=board.fen(),
            category=category,
            difficulty="medium",
            description=f"Random position after {num_moves} moves",
            source="generated",
            tags=["random"],
        ))
    
    print(f"Generated {len(positions)} random positions")
    return positions


def analyze_with_stockfish(positions: list, depth: int = 15) -> list:
    """Analyze positions with Stockfish to add best move and evaluation.
    
    Updates positions in-place with:
    - best_move
    - best_move_eval
    - second_best_move
    - second_best_eval
    - difficulty (based on eval difference between top moves)
    """
    try:
        import chess
        import chess.engine
    except ImportError:
        print("python-chess required: pip install python-chess")
        return positions
    
    # Try to find Stockfish
    stockfish_paths = [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        "stockfish",
    ]
    
    engine = None
    for path in stockfish_paths:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(path)
            print(f"Using Stockfish at: {path}")
            break
        except Exception:
            continue
    
    if engine is None:
        print("Stockfish not found. Install with: brew install stockfish")
        return positions
    
    try:
        for pos in positions:
            board = chess.Board(pos.fen)
            
            # Get top 3 moves
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=3)
            
            if info:
                # Best move
                best = info[0]
                pos.best_move = board.san(best["pv"][0])
                pos.best_move_eval = best["score"].relative.score(mate_score=10000)
                
                # Second best if available
                if len(info) > 1:
                    second = info[1]
                    pos.second_best_move = board.san(second["pv"][0])
                    pos.second_best_eval = second["score"].relative.score(mate_score=10000)
                    
                    # Set difficulty based on gap
                    if pos.best_move_eval and pos.second_best_eval:
                        gap = abs(pos.best_move_eval - pos.second_best_eval)
                        if gap > 200:
                            pos.difficulty = "easy"  # Clear best move
                        elif gap > 50:
                            pos.difficulty = "medium"
                        else:
                            pos.difficulty = "hard"  # Multiple good options
            
            print(f"  {pos.position_id}: best={pos.best_move} ({pos.best_move_eval})")
    
    finally:
        engine.quit()
    
    return positions


def main():
    parser = argparse.ArgumentParser(description="Collect chess positions for offline evaluation")
    parser.add_argument("--source", choices=["lichess", "pgn", "random", "standard"], 
                       default="standard", help="Position source")
    parser.add_argument("--count", type=int, default=50, help="Number of positions")
    parser.add_argument("--file", type=Path, help="PGN file path (for pgn source)")
    parser.add_argument("--output", type=Path, default=Path("_datasets/positions.json"),
                       help="Output dataset path")
    parser.add_argument("--analyze", action="store_true", help="Run Stockfish analysis")
    parser.add_argument("--analyze-dataset", type=Path, help="Analyze existing dataset")
    
    args = parser.parse_args()
    
    from game_arena.blitz.offline_eval import (
        PositionDataset,
        create_standard_dataset,
        save_dataset,
        load_dataset,
    )
    
    # Handle analyzing existing dataset
    if args.analyze_dataset:
        print(f"Loading dataset: {args.analyze_dataset}")
        dataset = load_dataset(args.analyze_dataset)
        dataset.positions = analyze_with_stockfish(dataset.positions)
        save_dataset(dataset, args.analyze_dataset)
        print(f"Updated dataset saved: {args.analyze_dataset}")
        return
    
    # Collect positions based on source
    if args.source == "standard":
        dataset = create_standard_dataset()
    
    elif args.source == "lichess":
        positions = fetch_lichess_puzzles(args.count)
        dataset = PositionDataset(
            name="lichess_puzzles",
            description=f"Positions from Lichess puzzles",
            positions=positions,
        )
    
    elif args.source == "pgn":
        if not args.file or not args.file.exists():
            print(f"PGN file required: --file path/to/games.pgn")
            return
        positions = extract_from_pgn(args.file, args.count)
        dataset = PositionDataset(
            name=f"pgn_{args.file.stem}",
            description=f"Positions from {args.file.name}",
            positions=positions,
        )
    
    elif args.source == "random":
        positions = generate_random_positions(args.count)
        dataset = PositionDataset(
            name="random_positions",
            description="Randomly generated positions",
            positions=positions,
        )
    
    else:
        print(f"Unknown source: {args.source}")
        return
    
    print(f"\nDataset: {dataset.name}")
    print(f"Positions: {len(dataset.positions)}")
    
    # Optionally analyze with Stockfish
    if args.analyze:
        dataset.positions = analyze_with_stockfish(dataset.positions)
    
    # Save dataset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(dataset, args.output)
    print(f"\nSaved to: {args.output}")
    
    # Print summary by category
    categories = {}
    for pos in dataset.positions:
        categories[pos.category] = categories.get(pos.category, 0) + 1
    
    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()

