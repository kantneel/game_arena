# Move Quality Analysis for Blitz Chess

This module provides Stockfish-powered move quality analysis for blitz chess games, including **win probability estimation**, integrated directly into Jupyter notebook visualizations.

## 🎯 Features

### 🆕 **Automatic Integration**
- **Zero-config setup**: Move analysis runs automatically after every blitz match
- **Instant notebooks**: Enhanced analysis notebooks are ready immediately 
- **Seamless workflow**: No separate commands needed - just run your match!

### Enhanced Notebook Charts
When move analysis data is available, notebooks automatically include:

**Move Quality Analysis:**
1. **Move Quality vs Time Taken** - Scatter plot showing correlation between thinking time and move quality
2. **Move Quality vs Turn Number** - Line plot showing move quality evolution throughout the game  
3. **Move Quality Distribution** - Histogram comparing quality distributions between models
4. **Move Quality Summary Statistics** - Detailed breakdown of perfect moves, mistakes, and blunders

**🆕 Win Probability Analysis:**
5. **Win Probability vs Time Taken** - Correlation between thinking time and resulting win chances
6. **Win Probability Evolution** - How win likelihood changes throughout the game
7. **Win Probability Loss Distribution** - Frequency of different-sized probability losses
8. **CP Loss vs Win Probability Correlation** - Relationship between centipawn and probability metrics
9. **Best vs Played Win Probability** - Accuracy visualization with perfect play line
10. **Win Probability Summary Statistics** - Detailed breakdown of probability accuracy

### Quality Thresholds
- **Perfect (0 cp)**: Engine's top choice
- **Excellent (1-10 cp)**: Very strong moves
- **Good (11-24 cp)**: Reasonable choices
- **Inaccuracy (25-49 cp)**: Suboptimal but playable
- **Mistake (50-99 cp)**: Clearly worse moves  
- **Blunder (100+ cp)**: Significant errors

### 🎯 Win Probability Metrics
- **Win Probability**: Likelihood of winning from the resulting position (0-100%)
- **Win Probability Loss**: How much win chance was lost by the move (percentage points)
- **Best Win Probability**: Optimal win chance after engine's best move
- **Played Win Probability**: Actual win chance after the played move

#### **Win Probability Scale:**
- **90-100%**: Winning position
- **70-90%**: Large advantage  
- **55-70%**: Clear advantage
- **45-55%**: Roughly equal
- **30-45%**: Clear disadvantage
- **10-30%**: Large disadvantage
- **0-10%**: Lost position

#### **Win Probability Loss Scale:**
- **0-2%**: Excellent move accuracy
- **2-5%**: Good move accuracy
- **5-10%**: Noticeable but acceptable loss
- **10-20%**: Significant mistake
- **20%+**: Major blunder

## 🚀 Quick Start

### 1. Prerequisites
Install Stockfish chess engine:
```bash
# macOS
brew install stockfish

# Ubuntu/Debian
sudo apt install stockfish

# Or download from https://stockfishchess.org/
```

### 2. Run Blitz Match with Automatic Analysis
**🆕 NEW: Move analysis now runs automatically!**
```bash
# Run a blitz match - move analysis happens automatically at the end
python -m game_arena.blitz.blitz_match --model_a=claude-sonnet-4 --model_b=claude-opus-4

# The notebook will be ready immediately with move quality charts included!
jupyter notebook game_arena/blitz/_results/your_match_id/your_match_id_analysis.ipynb
```

### 3. Manual Analysis (Optional)
For existing match data or custom analysis:
```bash
# Analyze a specific match directory manually
python -m game_arena.blitz.move_analysis.move_analyzer game_arena/blitz/_results/your_match_id/

# Or run the demo
python game_arena/blitz/setup_move_analysis_demo.py
```

### 4. Configuration Options
```bash
# Disable automatic analysis
python -m game_arena.blitz.blitz_match --run_move_analysis=false

# Adjust analysis quality (higher = more accurate, slower)
python -m game_arena.blitz.blitz_match --move_analysis_depth=20 --move_analysis_multipv=5

# Quick analysis for testing
python -m game_arena.blitz.blitz_match --move_analysis_depth=10 --move_analysis_multipv=3
```

## 📊 Generated Files

After running move analysis, these files are created:

```
match_directory/
├── complete_move_analysis.csv    # Complete move quality data
├── move_analysis_summary.json    # Quality statistics summary
├── game_1_move_analysis.csv      # Per-game analysis details
└── match_id_analysis.ipynb       # Enhanced notebook with quality charts
```

## 🔬 Analysis Details

### Engine Configuration
- **Default Depth**: 18 (adjustable for speed vs accuracy)
- **MultiPV**: 5 (analyzes top 5 moves)
- **Hash**: 512MB (configurable)
- **Threads**: 8 (configurable)

### Quality Metrics
- **Centipawn Loss**: Primary quality measure (lower = better)
- **Move Rank**: Where the played move ranks among engine's top choices
- **Best Move**: Engine's recommended move in each position
- **Evaluation**: Position evaluation before and after each move

## 📈 Interpreting Results

### Move Quality vs Time Charts
- **X-axis**: Time taken for move (seconds)
- **Y-axis**: Centipawn loss (quality - lower is better)
- **Colors**: Blue (Model A), Red (Model B)
- **Reference Lines**: Quality thresholds (25cp, 50cp, 100cp)

**Look for:**
- Do models make better moves when they think longer?
- Are there blunders under time pressure?
- Which model maintains quality under pressure?

### Move Quality vs Turn Number
- **X-axis**: Move number in the game  
- **Y-axis**: Centipawn loss (quality - lower is better)
- **Lines**: Connected points showing quality progression

**Look for:**
- Quality changes in opening, middlegame, endgame
- Critical moments where quality drops significantly
- Consistency patterns between models

### Quality Statistics
Detailed breakdown includes:
- Perfect moves percentage
- Blunder rates  
- Average centipawn loss
- Quality distribution by game phase

### 🆕 Win Probability Chart Interpretation

#### **Win Probability vs Time Taken**
- **X-axis**: Time taken for move (seconds)
- **Y-axis**: Win probability after the move (0-100%)
- **Interpretation**: Shows if longer thinking leads to better positions

**Look for:**
- Correlation between thinking time and resulting win chances
- Time pressure effects on position evaluation
- Differences in time utilization efficiency between models

#### **Win Probability Evolution**
- **X-axis**: Move number
- **Y-axis**: Win probability (0-100%)
- **Interpretation**: Shows the ebb and flow of the game

**Look for:**
- Critical turning points where advantage shifts
- Consistency in maintaining/converting advantages
- Opening, middlegame, and endgame performance patterns

#### **Win Probability Loss Distribution**
- **X-axis**: Win probability loss (percentage points)
- **Y-axis**: Frequency of moves
- **Interpretation**: Distribution of move accuracy

**Look for:**
- How often models make accurate moves (low loss)
- Frequency of major mistakes (high loss)
- Overall accuracy profile differences

#### **Best vs Played Win Probability**
- **X-axis**: Best possible win probability
- **Y-axis**: Actual win probability achieved
- **Diagonal line**: Perfect play
- **Interpretation**: Accuracy relative to optimal play

**Look for:**
- Points close to diagonal = accurate moves
- Points below diagonal = missed opportunities
- Systematic patterns in accuracy

## 🛠️ Advanced Usage

### Custom Analysis Parameters
```python
from game_arena.blitz.move_analysis.move_analyzer import MoveQualityAnalyzer

analyzer = MoveQualityAnalyzer(
    default_depth=20,      # Higher = more accurate, slower
    default_multipv=5,     # Number of variations to consider
    threads=16,            # CPU threads to use
    hash_mb=1024          # Memory for engine
)

results = analyzer.analyze_match_directory(
    "path/to/match",
    depth=20,
    multipv=5
)
```

### Integration with Tournament Analysis
```python
from game_arena.blitz.tournament.analysis.enhanced_tournament_analysis import EnhancedTournamentAnalyzer

analyzer = EnhancedTournamentAnalyzer()
tournament_data = analyzer.analyze_tournament_with_move_quality(
    "tournament_directory",
    run_move_analysis=True
)
```

## 🎮 Demo and Testing

### Run Complete Demo
```bash
python game_arena/blitz/setup_move_analysis_demo.py
```

This demo will:
1. Find available match data
2. Run move quality analysis  
3. Generate enhanced notebooks
4. Show example results and usage

### Analyze Single Position
```python
from game_arena.blitz.move_analysis.move_analyzer import MoveQualityAnalyzer

analyzer = MoveQualityAnalyzer()
result = analyzer.evaluate_move(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    move_str="e4",
    depth=15
)

print(f"Centipawn loss: {result['centipawn_loss']}")
print(f"Best move: {result['pretty']['best_move_san']}")
```

## 📚 Tips for Analysis

### Identifying Strong Play
- Low average centipawn loss (< 25)
- High percentage of perfect/excellent moves
- Consistent quality across game phases
- Good moves under time pressure

### Spotting Weaknesses  
- High blunder rate (> 5%)
- Quality drops significantly under time pressure
- Inconsistent play (high variance in centipawn loss)
- Specific opening/endgame weaknesses

### Comparing Models
- Average centipawn loss comparison
- Blunder rates under different time pressures
- Quality consistency patterns
- Strength in different game phases

## 🔧 Troubleshooting

### Common Issues

**Stockfish not found**
```
FileNotFoundError: Stockfish engine not found
```
Solution: Install Stockfish and ensure it's in your PATH

**Analysis takes too long**  
```
# Use lower depth for faster analysis
analyzer = MoveQualityAnalyzer(default_depth=12)
```

**No timing correlation**
```
"No timing data available for quality/time correlation"
```
Solution: Ensure original move CSV files contain timing data

### Performance Tips
- Use depth 12-15 for quick analysis
- Use depth 18+ for tournament-quality analysis  
- Adjust threads based on your CPU cores
- Increase hash size for complex positions

## 🚀 Next Steps

1. **Run analysis on your matches** to see move quality patterns
2. **Compare different models** across multiple games
3. **Identify improvement areas** based on blunder patterns
4. **Correlate quality with time pressure** to optimize thinking time
5. **Use in tournaments** to track model performance over time

Happy analyzing! 🏆
