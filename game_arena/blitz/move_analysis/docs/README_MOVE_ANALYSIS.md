# Chess Move Quality Analysis

This module provides comprehensive chess move quality analysis using the Stockfish engine to evaluate moves made by LLM agents during blitz chess games.

## 🆕 **Automatic Integration Available!**

**NEW: Move analysis now runs automatically with every blitz match!** 

- 📊 **Instant move quality charts** in every notebook
- 🚀 **Zero configuration** - analysis runs automatically after matches  
- 🎯 **Immediate insights** - correlate thinking time with move quality
- 🏆 **Engine-grade evaluation** - Stockfish analysis of every move

See `README_MOVE_QUALITY.md` for the new automatic integration features!

## 🚀 Quick Start

### 1. Install Dependencies

The required dependencies are already included in the project:
- `chess` (python-chess library) - already in pyproject.toml
- `pandas` - added to pyproject.toml

Install or update dependencies:
```bash
pip install -e .
```

### 2. Install Stockfish

Use the setup script to install and configure Stockfish:
```bash
python game_arena/blitz/setup_stockfish.py
```

Or install manually:
- **macOS**: `brew install stockfish`
- **Linux**: `sudo apt-get install stockfish` (Ubuntu/Debian)
- **Windows**: Download from [stockfishchess.org](https://stockfishchess.org/download/)

### 3. Analyze Moves

#### Single Match Analysis
```bash
python -m game_arena.blitz.move_analysis.move_analyzer <match_directory>
```

#### Enhanced Tournament Analysis
```bash
python -m game_arena.blitz.tournament.analysis.enhanced_tournament_analysis <tournament_directory>
```

#### Demo and Examples
```bash
python game_arena/blitz/move_analysis_demo.py
```

## 📊 Understanding Move Quality

### Move Quality Scale
- **0-10 cp loss**: Excellent moves (engine-level play)
- **11-24 cp loss**: Good moves (reasonable choices)
- **25-49 cp loss**: Inaccuracies (suboptimal but not terrible)
- **50-99 cp loss**: Mistakes (clearly worse moves)
- **100+ cp loss**: Blunders (significant errors)

### Example Analysis Output
```
Move Quality Breakdown:
  Perfect moves (0 cp):      15 (12.5%)
  Excellent moves (1-10 cp): 45 (37.5%)
  Good moves (11-24 cp):     35 (29.2%)
  Inaccuracies (25-49 cp):   15 (12.5%)
  Mistakes (50-99 cp):        8 (6.7%)
  Blunders (100+ cp):         2 (1.7%)
```

## 🔧 API Usage

### Basic Move Analysis
```python
from game_arena.blitz.move_analysis.move_analyzer import MoveQualityAnalyzer

analyzer = MoveQualityAnalyzer()

# Analyze a single move
result = analyzer.evaluate_move(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    move_str="e4",
    depth=18
)

print(f"Centipawn loss: {result['centipawn_loss']}")
print(f"Best move: {result['pretty']['best_move_san']}")
```

### Batch Analysis
```python
# Analyze all games in a match
results = analyzer.analyze_match_directory(
    "path/to/match/directory",
    depth=15,
    save_results=True
)
```

### Enhanced Tournament Analysis
```python
from game_arena.blitz.tournament.analysis.enhanced_tournament_analysis import EnhancedTournamentAnalyzer

analyzer = EnhancedTournamentAnalyzer()
analysis = analyzer.analyze_tournament_with_move_quality("tournament_dir")

# Generate human-readable report
report = analyzer.generate_move_quality_report("tournament_dir")
print(report)
```

## 📁 Output Files

The analysis generates several output files:

### Per-Game Files
- `game_X_move_analysis.csv` - Detailed analysis for each move in game X
- `complete_move_analysis.csv` - All moves from all games combined

### Summary Files
- `move_analysis_summary.json` - Statistical summary of move quality
- `enhanced_tournament_analysis.json` - Complete tournament analysis with move quality

### Analysis Fields

Each analyzed move includes:
- **Basic Info**: Game, move number, player, color
- **Move Data**: Move played, board position (FEN)
- **Engine Analysis**: Best move, evaluations, centipawn loss
- **Quality Metrics**: Move rank among top choices
- **Human-Readable**: Formatted strings for easy reading

## ⚙️ Configuration

### Engine Settings
```python
analyzer = MoveQualityAnalyzer(
    engine_path="/custom/path/to/stockfish",
    default_depth=20,        # Search depth (higher = more accurate, slower)
    default_multipv=5,       # Number of top moves to consider
    threads=8,               # CPU threads for analysis
    hash_mb=1024            # Memory allocation (MB)
)
```

### Analysis Parameters
- **Depth 10-15**: Fast analysis for large datasets
- **Depth 16-20**: Standard analysis (recommended)
- **Depth 21+**: Deep analysis for critical games

## 🎯 Use Cases

### 1. Model Evaluation
Compare LLM chess playing strength:
```bash
# Analyze tournament
python -m game_arena.blitz.enhanced_tournament_analysis tournament_data/

# View results
cat tournament_data/enhanced_tournament_analysis.json | jq '.move_quality_statistics.player_analysis'
```

### 2. Training Data Quality
Evaluate training data quality:
- Low centipawn loss → High-quality training examples
- High blunder rate → Need for training improvements

### 3. Tactical Analysis
Identify tactical weaknesses:
- Compare opening vs. middlegame vs. endgame performance
- Analyze time pressure effects on move quality

### 4. Model Comparison
Direct model comparison:
```python
analyzer = EnhancedTournamentAnalyzer()
report = analyzer.generate_move_quality_report("claude_vs_gpt_tournament/")
print(report)  # Shows detailed player comparison
```

## 🛠️ Troubleshooting

### Stockfish Issues
```bash
# Verify Stockfish installation
python game_arena/blitz/setup_stockfish.py

# Test engine directly
stockfish
# Type: uci
# Should respond with: uciok
```

### Common Errors
- **"Stockfish not found"**: Run setup script or check PATH
- **"JSON serialization error"**: Update move_analyzer.py (fixed in current version)
- **"Illegal move"**: Check FEN notation and move format (UCI vs SAN)

### Performance Tips
- Use depth 12-15 for initial analysis
- Reduce multipv for faster analysis
- Process games in parallel for large datasets

## 📈 Integration with Existing Analysis

The move quality analysis integrates seamlessly with existing tournament analysis:

1. **Data Collection**: Uses existing `game_X_moves.csv` format
2. **Tournament Analysis**: Extends `tournament_analysis.py` capabilities
3. **Visualization**: Compatible with existing Jupyter notebooks
4. **Reporting**: Adds move quality metrics to existing reports

## 🤝 Contributing

To extend the move analysis functionality:

1. Add new quality metrics in `MoveAnalysis` dataclass
2. Extend `_generate_analysis_summary()` for new statistics
3. Update visualization in `enhanced_tournament_analysis.py`
4. Add tests for new functionality

## 📚 References

- [Stockfish Chess Engine](https://stockfishchess.org/)
- [python-chess Documentation](https://python-chess.readthedocs.io/)
- [Centipawn Evaluation](https://en.wikipedia.org/wiki/Chess_engine#Evaluation)
- [Chess Move Quality](https://en.wikipedia.org/wiki/Computer_chess#Evaluation_function)
