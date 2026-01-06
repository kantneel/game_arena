#!/usr/bin/env python3
"""Chess position dataset for offline evaluation.

This module provides a curated collection of chess positions for controlled
experiments on how LLMs respond to time pressure.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ChessPosition:
    """A chess position for evaluation."""
    
    # Position identification
    position_id: str
    fen: str
    
    # Metadata
    category: str  # "opening", "middlegame", "endgame", "tactical", "positional"
    difficulty: str  # "easy", "medium", "hard"
    description: str
    
    # Ground truth from Stockfish (computed offline)
    best_move: Optional[str] = None
    best_move_eval: Optional[int] = None  # Centipawns from current player's perspective
    second_best_move: Optional[str] = None
    second_best_eval: Optional[int] = None
    
    # Additional context
    source: str = "curated"  # "curated", "lichess", "chessgames", etc.
    tags: list[str] = field(default_factory=list)


@dataclass
class PositionDataset:
    """A collection of chess positions for evaluation."""
    
    name: str
    description: str
    positions: list[ChessPosition]
    version: str = "1.0"
    
    def __len__(self):
        return len(self.positions)
    
    def __iter__(self):
        return iter(self.positions)
    
    def filter_by_category(self, category: str) -> "PositionDataset":
        """Filter positions by category."""
        filtered = [p for p in self.positions if p.category == category]
        return PositionDataset(
            name=f"{self.name}_{category}",
            description=f"{self.description} (filtered: {category})",
            positions=filtered,
            version=self.version,
        )
    
    def filter_by_difficulty(self, difficulty: str) -> "PositionDataset":
        """Filter positions by difficulty."""
        filtered = [p for p in self.positions if p.difficulty == difficulty]
        return PositionDataset(
            name=f"{self.name}_{difficulty}",
            description=f"{self.description} (filtered: {difficulty})",
            positions=filtered,
            version=self.version,
        )


def save_dataset(dataset: PositionDataset, path: Path) -> None:
    """Save dataset to JSON file."""
    data = {
        "name": dataset.name,
        "description": dataset.description,
        "version": dataset.version,
        "positions": [asdict(p) for p in dataset.positions],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset(path: Path) -> PositionDataset:
    """Load dataset from JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    positions = [ChessPosition(**p) for p in data["positions"]]
    return PositionDataset(
        name=data["name"],
        description=data["description"],
        positions=positions,
        version=data.get("version", "1.0"),
    )


def create_standard_dataset() -> PositionDataset:
    """Create a standard evaluation dataset with diverse positions.
    
    The dataset includes:
    - Opening positions (move 5-10)
    - Middlegame positions (tactical and positional)
    - Endgame positions (various piece configurations)
    - Positions with clear best moves vs ambiguous positions
    
    Returns:
        PositionDataset with curated positions
    """
    positions = [
        # ===== OPENING POSITIONS =====
        ChessPosition(
            position_id="open_001",
            fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            category="opening",
            difficulty="easy",
            description="Italian Game setup - develop bishop",
            best_move="Bb5",
            tags=["italian", "development"],
        ),
        ChessPosition(
            position_id="open_002",
            fen="rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            category="opening",
            difficulty="easy",
            description="After 1.e4 Nf6 - Alekhine Defense",
            best_move="e5",
            tags=["alekhine", "pawn_advance"],
        ),
        ChessPosition(
            position_id="open_003",
            fen="rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
            category="opening",
            difficulty="easy",
            description="Scandinavian Defense - capture or advance?",
            best_move="exd5",
            tags=["scandinavian", "pawn_capture"],
        ),
        
        # ===== MIDDLEGAME TACTICAL =====
        ChessPosition(
            position_id="mid_tac_001",
            fen="r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            category="tactical",
            difficulty="easy",
            description="Scholar's mate threat - Qxf7#",
            best_move="Qxf7",
            tags=["mate_in_1", "scholars_mate"],
        ),
        ChessPosition(
            position_id="mid_tac_002",
            fen="r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
            category="tactical",
            difficulty="medium",
            description="Pin on f7 - develop with tempo",
            best_move="Bg5",
            tags=["pin", "development"],
        ),
        ChessPosition(
            position_id="mid_tac_003",
            fen="r2qkb1r/ppp2ppp/2n1bn2/3pp3/4P3/1PN2N2/PBPP1PPP/R2QKB1R w KQkq - 0 6",
            category="tactical",
            difficulty="medium",
            description="Central tension - tactical options",
            best_move="exd5",
            tags=["center", "pawn_tension"],
        ),
        ChessPosition(
            position_id="mid_tac_004",
            fen="r1bqr1k1/ppp2ppp/2n2n2/3p4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 8",
            category="tactical",
            difficulty="hard",
            description="IQP position - dynamic play",
            best_move="O-O",
            tags=["isolated_pawn", "castling"],
        ),
        
        # ===== MIDDLEGAME POSITIONAL =====
        ChessPosition(
            position_id="mid_pos_001",
            fen="r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
            category="positional",
            difficulty="medium",
            description="Closed Italian - maneuvering",
            best_move="h3",
            tags=["prophylaxis", "italian"],
        ),
        ChessPosition(
            position_id="mid_pos_002",
            fen="r2q1rk1/pppb1ppp/2np1n2/2b1p3/4P3/2PP1N1P/PP1N1PP1/R1BQRBK1 w - - 0 10",
            category="positional",
            difficulty="hard",
            description="Complex middlegame - many plans",
            tags=["complex", "multiple_plans"],
        ),
        ChessPosition(
            position_id="mid_pos_003",
            fen="r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8",
            category="positional",
            difficulty="medium",
            description="Queen's Gambit structure",
            best_move="dxc5",
            tags=["qgd", "pawn_structure"],
        ),
        
        # ===== ENDGAME POSITIONS =====
        ChessPosition(
            position_id="end_001",
            fen="8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
            category="endgame",
            difficulty="easy",
            description="King and pawn vs king - basic",
            best_move="Kf4",
            tags=["kp_vs_k", "opposition"],
        ),
        ChessPosition(
            position_id="end_002",
            fen="8/5pk1/8/5PK1/8/8/8/8 w - - 0 1",
            category="endgame",
            difficulty="medium",
            description="Pawn race - calculation needed",
            best_move="f6+",
            tags=["pawn_race", "check"],
        ),
        ChessPosition(
            position_id="end_003",
            fen="8/8/8/3k4/8/3K4/3R4/8 w - - 0 1",
            category="endgame",
            difficulty="easy",
            description="Rook and king vs king - technique",
            best_move="Rd1",
            tags=["rook_endgame", "cutting_off"],
        ),
        ChessPosition(
            position_id="end_004",
            fen="8/2k5/3p4/3P4/8/3K4/8/8 w - - 0 1",
            category="endgame",
            difficulty="medium",
            description="King and pawn - opposition battle",
            best_move="Ke4",
            tags=["opposition", "pawn_endgame"],
        ),
        ChessPosition(
            position_id="end_005",
            fen="6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
            category="endgame",
            difficulty="medium",
            description="Rook vs pawns - activity",
            best_move="Re7",
            tags=["rook_activity", "seventh_rank"],
        ),
        
        # ===== COMPLEX/AMBIGUOUS POSITIONS =====
        ChessPosition(
            position_id="complex_001",
            fen="r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9",
            category="positional",
            difficulty="hard",
            description="Dragon Sicilian - many candidate moves",
            tags=["sicilian", "dragon", "complex"],
        ),
        ChessPosition(
            position_id="complex_002",
            fen="r2qr1k1/1b1nbppp/pp1ppn2/8/2PNP3/1PN1B3/PB1Q1PPP/R3R1K1 w - - 0 14",
            category="positional",
            difficulty="hard",
            description="Hedgehog structure - strategic depth",
            tags=["hedgehog", "strategic"],
        ),
        
        # ===== TIME SCRAMBLE TYPICAL =====
        ChessPosition(
            position_id="scramble_001",
            fen="r4rk1/pp2qppp/2p1bn2/4p3/4P3/1PN2N2/PBP2PPP/R2QR1K1 w - - 0 12",
            category="middlegame",
            difficulty="medium",
            description="Typical blitz position - quick decision needed",
            best_move="d4",
            tags=["blitz", "central_break"],
        ),
        ChessPosition(
            position_id="scramble_002",
            fen="r1bq1rk1/ppp1nppp/3p1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 8",
            category="middlegame",
            difficulty="medium",
            description="Active piece play - intuition test",
            best_move="Nd5",
            tags=["knight_outpost", "activity"],
        ),
    ]
    
    return PositionDataset(
        name="standard_eval_v1",
        description="Standard evaluation dataset for time pressure experiments",
        positions=positions,
        version="1.0",
    )


def create_stress_test_dataset() -> PositionDataset:
    """Create a stress test dataset with positions likely to cause blunders.
    
    This dataset is specifically designed to test LLM decision-making under
    time pressure with positions that have:
    - Tempting but losing moves (traps)
    - Counterintuitive best moves
    - High-stakes calculations (one move wins, others lose)
    - Complex tactics with multiple threats
    - Positions where quick/obvious choices are wrong
    
    Returns:
        PositionDataset with challenging positions
    """
    positions = [
        # ===== TACTICAL TRAPS (tempting but losing) =====
        ChessPosition(
            position_id="trap_001",
            fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            category="tactical",
            difficulty="hard",
            description="Italian trap: Ng5 looks aggressive but Bxf7+ Ke7 is better",
            best_move="Bxf7+",
            second_best_move="Ng5",
            tags=["trap", "sacrifice", "counterintuitive"],
        ),
        ChessPosition(
            position_id="trap_002",
            fen="r1bqk2r/ppp2ppp/2n2n2/2bpp3/4P3/2PP1N2/PP3PPP/RNBQKB1R w KQkq - 0 5",
            category="tactical",
            difficulty="hard",
            description="Tempting to capture on d5, but cxd5 loses to Nxe4",
            best_move="Be2",
            second_best_move="exd5",
            tags=["trap", "pawn_grab", "development"],
        ),
        ChessPosition(
            position_id="trap_003",
            fen="r1bqkbnr/pp1ppppp/2n5/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
            category="tactical", 
            difficulty="medium",
            description="Sicilian - cxd4 is obvious but e6 is more solid",
            best_move="cxd4",
            second_best_move="e6",
            tags=["sicilian", "capture", "theory"],
        ),
        
        # ===== COUNTERINTUITIVE MOVES =====
        ChessPosition(
            position_id="counter_001",
            fen="r1bq1rk1/ppp2ppp/2n5/3np3/1bB5/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 8",
            category="tactical",
            difficulty="hard",
            description="Bishop looks trapped - but Bxd5! sacrifices to win it back with tempo",
            best_move="Bxd5",
            tags=["sacrifice", "counterintuitive", "tactics"],
        ),
        ChessPosition(
            position_id="counter_002",
            fen="r2qkb1r/ppp1pppp/2n2n2/3p4/3P1Bb1/2N2N2/PPP1PPPP/R2QKB1R w KQkq - 4 5",
            category="tactical",
            difficulty="hard",
            description="Natural developing moves lose - Ne5! exploits the pin",
            best_move="Ne5",
            second_best_move="e3",
            tags=["pin", "counterintuitive", "knight_tactics"],
        ),
        ChessPosition(
            position_id="counter_003",
            fen="r1bq1rk1/pp3ppp/2n1pn2/2pp4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 8",
            category="tactical",
            difficulty="hard",
            description="Don't recapture the pawn! a3 wins the bishop",
            best_move="a3",
            second_best_move="cxd5",
            tags=["trap", "bishop_trap", "tactics"],
        ),
        
        # ===== HIGH-STAKES CALCULATION (one right answer) =====
        ChessPosition(
            position_id="calc_001",
            fen="r1b1k2r/ppppqppp/2n2n2/2b1p1N1/2B1P3/3P4/PPP2PPP/RNBQK2R w KQkq - 0 6",
            category="tactical",
            difficulty="hard",
            description="Fried Liver setup - Bxf7+ or Nxf7? Only one works",
            best_move="Nxf7",
            second_best_move="Bxf7+",
            tags=["sacrifice", "calculation", "fried_liver"],
        ),
        ChessPosition(
            position_id="calc_002",
            fen="r2qr1k1/ppp2ppp/2nb4/3np1B1/8/2NB4/PPP2PPP/R2Q1RK1 w - - 0 12",
            category="tactical",
            difficulty="hard",
            description="Multiple captures possible - only Bxh7+! wins material",
            best_move="Bxh7+",
            tags=["greek_gift", "sacrifice", "calculation"],
        ),
        ChessPosition(
            position_id="calc_003",
            fen="r1bqk2r/ppp2ppp/2n2n2/3Pp3/1b6/2N2N2/PPP1BPPP/R1BQK2R w KQkq - 0 7",
            category="tactical",
            difficulty="hard",
            description="The d-pawn looks weak but d6! creates problems",
            best_move="d6",
            second_best_move="O-O",
            tags=["pawn_push", "calculation", "disruption"],
        ),
        
        # ===== COMPLEX MULTI-PIECE TACTICS =====
        ChessPosition(
            position_id="complex_tac_001",
            fen="r2qkb1r/1b1n1ppp/p2ppn2/1p6/3NP1P1/2N1BP2/PPPQ3P/R3KB1R w KQkq - 0 10",
            category="tactical",
            difficulty="hard",
            description="Multiple pieces are loose - find the forcing sequence",
            best_move="g5",
            tags=["attack", "loose_pieces", "calculation"],
        ),
        ChessPosition(
            position_id="complex_tac_002",
            fen="r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 9",
            category="tactical",
            difficulty="hard",
            description="Tension in the center - timing of captures is critical",
            best_move="dxc5",
            second_best_move="e4",
            tags=["center", "tension", "timing"],
        ),
        ChessPosition(
            position_id="complex_tac_003",
            fen="r2q1rk1/pb1nbppp/1p2pn2/2p5/3P4/1PN1PN2/PB2BPPP/R2Q1RK1 w - - 0 11",
            category="tactical",
            difficulty="hard",
            description="Hedgehog - d5 break requires precise calculation",
            best_move="d5",
            tags=["pawn_break", "calculation", "hedgehog"],
        ),
        
        # ===== QUEEN TRAP POSITIONS =====
        ChessPosition(
            position_id="queen_trap_001",
            fen="rnbqkbnr/ppp2ppp/8/3pp3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 3",
            category="tactical",
            difficulty="medium",
            description="Qh5 looks tempting (threatens mate) but can be trapped",
            best_move="Qh5",
            tags=["queen_activity", "aggression", "blunder_prone"],
        ),
        ChessPosition(
            position_id="queen_trap_002",
            fen="r1bqkb1r/pppp1ppp/2n5/4p2n/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 1 4",
            category="tactical",
            difficulty="medium",
            description="d5 gains space but Ne4 is coming - timing matters",
            best_move="Nxe5",
            second_best_move="d5",
            tags=["central_control", "timing", "tactics"],
        ),
        
        # ===== DEFENSIVE POSITIONS (only move holds) =====
        ChessPosition(
            position_id="defense_001",
            fen="r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 5",
            category="tactical",
            difficulty="hard",
            description="White threatens Ng5 - only d6 defends properly",
            best_move="d6",
            second_best_move="O-O",
            tags=["defense", "prophylaxis", "f7_weakness"],
        ),
        ChessPosition(
            position_id="defense_002",
            fen="r1bqr1k1/ppp2ppp/2n2n2/3p4/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 8",
            category="tactical",
            difficulty="hard",
            description="Under attack - only a3 holds the position",
            best_move="a3",
            tags=["defense", "bishop_attack", "tempo"],
        ),
        
        # ===== BACK RANK VULNERABILITIES =====
        ChessPosition(
            position_id="backrank_001",
            fen="3r2k1/pp3ppp/8/8/8/1P6/P4PPP/3R2K1 w - - 0 1",
            category="tactical",
            difficulty="medium",
            description="Looks equal but Rd8+ leads to back rank issues",
            best_move="Rd8+",
            tags=["back_rank", "endgame", "tactics"],
        ),
        ChessPosition(
            position_id="backrank_002",
            fen="r2q1rk1/ppp2ppp/2n5/3np3/8/2N2Q2/PPP2PPP/R1B2RK1 w - - 0 10",
            category="tactical",
            difficulty="hard",
            description="Multiple threats - but watch your own back rank",
            best_move="Qg3",
            second_best_move="Qd3",
            tags=["queen_placement", "back_rank", "defense"],
        ),
        
        # ===== PAWN ENDGAME TRAPS =====
        ChessPosition(
            position_id="pawn_end_001",
            fen="8/8/4k3/8/3K4/8/4P3/8 w - - 0 1",
            category="endgame",
            difficulty="hard",
            description="Opposition is key - Ke4 wins, e4 draws",
            best_move="Ke4",
            second_best_move="e4",
            tags=["opposition", "pawn_endgame", "key_square"],
        ),
        ChessPosition(
            position_id="pawn_end_002",
            fen="8/5k2/8/5P2/5K2/8/8/8 w - - 0 1",
            category="endgame",
            difficulty="medium",
            description="Should the king advance or the pawn?",
            best_move="Ke5",
            second_best_move="f6+",
            tags=["pawn_promotion", "king_activity", "endgame"],
        ),
    ]
    
    return PositionDataset(
        name="stress_test_v1",
        description="Stress test dataset - positions designed to cause blunders under time pressure",
        positions=positions,
        version="1.0",
    )


def create_combined_dataset() -> PositionDataset:
    """Create a combined dataset with both standard and stress test positions.
    
    Returns:
        PositionDataset with all positions
    """
    standard = create_standard_dataset()
    stress = create_stress_test_dataset()
    
    return PositionDataset(
        name="combined_eval_v1",
        description="Combined standard + stress test positions",
        positions=standard.positions + stress.positions,
        version="1.0",
    )


# ===== SUGGESTIONS FOR EXPANDING THE DATASET =====
"""
To expand this dataset, consider:

1. **Lichess Puzzles API**
   - Filter by theme (fork, pin, mate, etc.)
   - Filter by rating range
   - Get positions with known best moves

2. **Famous Games Database**
   - Critical moments from GM games
   - Known theoretical positions
   - Positions where strong players blundered

3. **Tactical Pattern Collections**
   - Classic combinations (Greek gift, Bxh7+)
   - Deflection, decoy, discovered attack
   - Mate patterns (back rank, smothered)

4. **Positional Themes**
   - Pawn structures (IQP, hanging pawns, etc.)
   - Piece activity patterns
   - King safety positions

5. **Blitz-Specific Positions**
   - Positions from actual blitz games where players flagged
   - Positions requiring quick calculation vs deep thought
   - "Trick" positions that look obvious but aren't

6. **Model Failure Analysis**
   - Positions where LLMs commonly err
   - Positions testing specific weaknesses

Usage example to add positions from Lichess:
```python
import requests

def fetch_lichess_puzzles(count=20, min_rating=1500, max_rating=2000):
    positions = []
    # Use Lichess puzzle API
    # Filter and convert to ChessPosition format
    return positions
```
"""

