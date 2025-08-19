# Blitz Chess Best of 7 Match Script

This script implements a real-time blitz chess match between OpenAI and Gemini models, featuring comprehensive time management, network latency calibration, and detailed reasoning efficiency analysis.

## 🏆 Key Features

### ⏰ Real-Time Clock Management
- **Wall Clock Time**: Uses actual wall clock time that directly corresponds to reasoning trace length
- **Time Pressure**: Models must balance move quality vs. time consumption
- **Increment System**: Configurable time increment added after each move
- **Time Forfeit**: Players lose immediately if they run out of time

### 🌐 Network Latency Calibration
- **Pre-Match Calibration**: Measures network roundtrip time for each model
- **Latency Adjustment**: Subtracts network latency from thinking time calculation
- **Minimal Token Requests**: Uses `max_output_tokens=1` for accurate latency measurement

### 📊 Comprehensive Analytics
- **Move-by-Move Statistics**: Tracks thinking time, reasoning tokens, and time remaining
- **Reasoning Efficiency**: Measures tokens per second (reasoning throughput)
- **Time Pressure Analysis**: Analyzes performance under different time constraints
- **Match-Level Insights**: Aggregated statistics across all games

### 🏆 Tournament System (NEW)
- **Single Elimination Brackets**: 8-model tournaments with automatic bracket management
- **Cross-Match Analysis**: Compare performance across different opponents
- **Tournament Analytics**: Advanced time pressure and tactical behavior analysis
- **Automated Data Collection**: Comprehensive tournament data tracking and visualization

### 🎯 Research Focus
Tests how LLMs handle the **latency-quality tradeoff** in real-time scenarios:
- Do models adapt their reasoning depth based on available time?
- How does time pressure affect move quality?
- Which model better manages the speed vs. accuracy balance?
- Do models use time tactically to pressure opponents?

## 🚀 Usage

### Single Best-of-7 Match
```bash
python blitz_match.py
```

Default settings:
- **Time Control**: 5 minutes + 3 seconds increment per move
- **Gemini**: `gemini-2.5-flash`
- **OpenAI**: `gpt-4.1`
- **Calibration**: 3 rounds of latency measurement

### Tournament Mode (8 Models)
```bash
python run_tournament.py
```

Default tournament with 8 models:
- `gemini-2.5-flash`, `gpt-4.1`, `claude-3.5-sonnet`, `llama-3.1-70b`
- `gemini-2.5-pro`, `gpt-4.0`, `claude-3-opus`, `mistral-large`

Custom model selection:
```bash
python run_tournament.py \
    --models="model1,model2,model3,model4,model5,model6,model7,model8"
```

### Custom Time Controls
```bash
# Bullet chess (1 minute + 1 second)
python blitz_match.py \
    --initial_time_seconds=60 \
    --increment_seconds=1

# Rapid chess (10 minutes + 5 seconds)
python blitz_match.py \
    --initial_time_seconds=600 \
    --increment_seconds=5
```

### Advanced Options
```bash
# Enable rethinking with custom settings
python blitz_match.py \
    --use_rethinking=true \
    --max_rethinks=3 \
    --reasoning_budget=10000 \
    --show_reasoning_traces=true

# Custom parsing strategy
python blitz_match.py \
    --parser_choice=llm_only \
    --max_parsing_failures=5
```

## 📊 Data Analysis

### Individual Match Analysis
After each match, the system automatically generates:
- **CSV Files**: `games.csv` and `moves.csv` with detailed statistics
- **Jupyter Notebook**: Interactive analysis with visualizations
- **Summary Report**: Key insights and performance metrics

### Tournament Analysis
For tournaments, additional analysis includes:
- **Tournament Bracket**: Visual bracket with results
- **Cross-Match Performance**: How models perform against different opponents  
- **Time Pressure Behavior**: Detailed analysis of responses to time constraints
- **Tactical Time Usage**: Detection of strategic time management patterns
- **Token Efficiency**: Analysis of reasoning efficiency under pressure

Key questions answered:
1. **End scores in each round**: Tracked in tournament metadata and match results
2. **Time remaining after every move**: Recorded in `MoveRecord.time_remaining_after`
3. **Tokens spent on each response**: Tracked in `MoveRecord.total_tokens` and `reasoning_tokens`
4. **Response to dwindling time**: Analyzed via `time_pressure_level` (LOW/MEDIUM/HIGH)
5. **Tactical time usage**: Correlation analysis between opponent pressure and thinking time

### Data Location
- Single matches: `_data/{match_id}/`
- Tournaments: `tournament_data/{tournament_id}/`

## 🧠 Rethinking System

## ⚙️ Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--initial_time_seconds` | 300 | Starting time per player (seconds) |
| `--increment_seconds` | 3 | Time increment per move (seconds) |
| `--calibration_rounds` | 3 | Number of latency calibration rounds |
| `--gemini_model` | `gemini-2.5-flash` | Gemini model to use |
| `--openai_model` | `gpt-4.1` | OpenAI model to use |
| `--parser_choice` | `RULE_THEN_SOFT` | Move parser type |
| `--max_moves_per_game` | 200 | Maximum moves to prevent infinite games |

## 📋 Output & Analysis

### Live Game Display
```
=== BLITZ GAME 1 ===
Gemini (White) vs OpenAI (Black)
⏰ Starting time: 05:00.0 each
⏰ Increment: +3s per move

🌐 Calibrating network latencies...
  Round 1: 0.245s
  Round 2: 0.238s
  Round 3: 0.251s
Average network latency: 0.245s

Move 1: Gemini's turn
⏰ Gemini: 05:00.0 | Opponent: 05:00.0
Gemini response: I'll start with a solid opening move...
Parsed move: e4
⏰ Thinking time: 2.34s

Move 2: OpenAI's turn
⏰ OpenAI: 05:00.0 | Opponent: 05:00.7
Current time pressure level: 🟢 LOW
```

### Detailed Game Analysis
```
📊 DETAILED ANALYSIS - GAME 1
Duration: 847.3s, Moves: 42
Final times - Gemini: 01:23.4, OpenAI: 02:15.7

Gemini stats:
  Moves played: 21
  Avg thinking time: 3.45s
  Total thinking time: 72.4s
  Avg reasoning tokens: 1,247
  Slowest move: Qxf7+ (8.23s)
  Fastest move: O-O (1.12s)

OpenAI stats:
  Moves played: 21
  Avg thinking time: 2.89s
  Total thinking time: 60.7s
  Avg reasoning tokens: 892
  Slowest move: Rxd4 (6.78s)
  Fastest move: exd4 (0.98s)
```

### Final Match Statistics
```
⏰ TIME MANAGEMENT ANALYSIS:
Average final time - Gemini: 01:45.2
Average final time - OpenAI: 02:03.8

🧠 REASONING EFFICIENCY ANALYSIS:
Gemini overall performance:
  Total moves: 84
  Avg thinking time: 3.21s
  Avg reasoning tokens: 1,156
  Reasoning efficiency: 360.1 tokens/second
  Under time pressure (<60s): 12 moves, avg 1.89s thinking

OpenAI overall performance:
  Total moves: 82
  Avg thinking time: 2.94s
  Avg reasoning tokens: 934
  Reasoning efficiency: 317.7 tokens/second
  Under time pressure (<60s): 8 moves, avg 1.45s thinking
```

## 🔬 Research Applications

### Time Pressure Studies
- **Adaptive Reasoning**: Does reasoning depth decrease under time pressure?
- **Quality vs. Speed**: How does move quality correlate with thinking time?
- **Pressure Response**: Do models make more errors when time is low?

### Model Comparison
- **Time Management**: Which model better manages its clock?
- **Efficiency**: Which model generates more reasoning per second?
- **Adaptability**: Which model better adapts to time constraints?

### Performance Metrics
- **Reasoning Tokens/Second**: Measures reasoning throughput
- **Time Utilization**: How effectively models use available time
- **Pressure Performance**: Quality under different time constraints

## 🎮 Time Control Variants

### Bullet Chess (Ultra-Fast)
```bash
python blitz_match.py \
    --initial_time_seconds=60 \
    --increment_seconds=1
```

### Blitz Chess (Fast)
```bash
python blitz_match.py \
    --initial_time_seconds=300 \
    --increment_seconds=3
```

### Rapid Chess (Standard)
```bash
python blitz_match.py \
    --initial_time_seconds=900 \
    --increment_seconds=10
```

## 🧠 Prompt Engineering for Time Awareness

The script automatically includes time-aware information in prompts:

```
BLITZ CHESS TIME INFORMATION:
⏰ Your remaining time: 02:34.5
⏰ Opponent's remaining time: 03:12.8
⏰ Time increment per move: +3 seconds

⚠️ CRITICAL TIME RULES:
- This is REAL WALL CLOCK TIME - your thinking/reasoning time directly consumes your clock
- You lose immediately if your time runs out (time forfeit)
- Longer reasoning traces = more time consumed = higher risk of time forfeit
- You must balance move quality vs. time management
- Each move adds 3 seconds to your clock after you play it
- Consider quick, good moves over perfect moves that consume too much time

Current time pressure level: 🟡 MEDIUM
```

## 🔧 Technical Implementation

### Clock Management
- **PlayerClock Class**: Tracks time remaining, move timing, and statistics
- **Real-Time Measurement**: Uses `time.time()` for wall clock precision
- **Latency Compensation**: Subtracts measured network latency from thinking time

### Statistics Collection
- **MoveStats**: Per-move data including tokens, timing, and context
- **GameStats**: Complete game analysis with aggregated metrics
- **Match-Level Analytics**: Cross-game performance comparisons

### Network Latency Calibration
```python
def calibrate_network_latency(model) -> float:
    """Calibrate by making minimal token requests."""
    # Set max_output_tokens=1 for fastest response
    # Measure roundtrip time
    # Return average latency across multiple rounds
```

## 📈 Expected Research Insights

1. **Time Adaptation**: Do models reduce reasoning depth when time is scarce?
2. **Quality Degradation**: How does move quality change under time pressure?
3. **Model Differences**: Which models are better at time management?
4. **Optimal Balance**: What's the sweet spot between speed and accuracy?
5. **Pressure Patterns**: Are there consistent patterns in time pressure responses?

## 🎯 Use Cases

- **LLM Research**: Study reasoning efficiency and time management
- **Competitive Analysis**: Compare model performance under time constraints
- **Algorithm Development**: Test real-time decision-making capabilities
- **Benchmarking**: Establish time-aware performance metrics

## 🚨 Requirements

- OpenAI API access with `OPENAI_API_KEY` environment variable
- Google AI Studio API access for Gemini models
- All game_arena project dependencies
- Stable internet connection for accurate latency measurement

## 💡 Tips for Best Results

1. **Stable Network**: Run on a stable connection for consistent latency
2. **Multiple Calibrations**: Use more calibration rounds for better accuracy
3. **Time Control Selection**: Choose appropriate time controls for your research goals
4. **Model Selection**: Consider using latest models for optimal reasoning capabilities
5. **Analysis Focus**: Focus on the metrics most relevant to your research questions 