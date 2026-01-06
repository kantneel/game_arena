# LLM Behavior Under Time Pressure: A Chess-Based Analysis

## Abstract

This study investigates how Large Language Models (LLMs) adapt their behavior when facing time constraints in a competitive chess setting. Using blitz chess as an experimental paradigm, we analyze whether LLMs exhibit time-awareness similar to human players—specifically, whether they speed up decision-making, reduce reasoning depth, and maintain or sacrifice move quality when running low on time. We compare multiple Gemini model variants across generations (2.5 and 3) and capability tiers (Pro and Flash) to understand how model architecture and training affect time management strategies.

---

## 1. Introduction

### 1.1 Motivation

Human chess players exhibit sophisticated time management under pressure. When the clock runs low, experienced players:
- Make moves more quickly, often relying on intuition
- Reduce calculation depth, focusing on tactical threats
- Accept slightly suboptimal moves to preserve time
- Shift from strategic planning to tactical survival

**Do LLMs exhibit similar adaptive behavior?**

Unlike humans, LLMs don't have an innate sense of urgency. They generate tokens at a consistent rate and don't "feel" time pressure. However, through prompting and context, we can inform LLMs about their remaining time and observe whether they adapt their reasoning patterns accordingly.

### 1.2 Research Questions

This study addresses the following questions:

> **RQ1: Speed Adaptation**  
> Do LLMs make faster moves when time is running low? If so, how dramatic is the speedup?

> **RQ2: Thinking Depth**  
> Do LLMs reduce their reasoning token output under time pressure? Is this reduction deliberate or emergent?

> **RQ3: Quality Preservation**  
> How does move quality (measured by centipawn loss) change under pressure? Do models make more blunders?

> **RQ4: Reasoning Content**  
> What patterns appear in the model's reasoning traces under pressure? Do they mention time? Do they truncate analysis?

> **RQ5: Model Differences**  
> How do different model variants compare? Do "Pro" models handle pressure differently than "Flash" models? Has time management improved across generations?

> **RQ6: Prompt Engineering Effect**  
> Does including time pressure information in prompts actually make a difference? What is the effect of different prompt styles (standard vs dramatic)?

> **RQ7: Recurrent Awareness**  
> Can models adapt better when given feedback about their previous response (time taken, tokens used, tokens/second rate)?

> **RQ8: Controlled Position Analysis** *(New)*  
> When given the exact same position with different time remaining values, how do models adjust their behavior? What is the variance in responses across repeated samples?

### 1.3 Hypotheses

Based on our understanding of LLM behavior, we hypothesize:

1. **H1:** Models will show some speed adaptation, but less dramatically than humans
2. **H2:** Pro models will maintain quality better under pressure than Flash models
3. **H3:** Newer generations (Gemini 3) will show better time management than older (Gemini 2.5)
4. **H4:** Self-play will reveal inherent variance and white/black asymmetries
5. **H5:** Removing time pressure prompts will eliminate or significantly reduce adaptation behavior
6. **H6:** Dramatic prompts will increase adaptation but may hurt move quality
7. **H7:** Response feedback will enable models to calibrate their thinking depth more effectively
8. **H8:** Offline evaluation will show consistent adaptation patterns across samples, with tactical positions being most affected by time pressure

---

## 2. Methodology

### 2.1 Experimental Setup

**Game Format:**
- Chess variant: Standard chess
- Time control: 5 minutes + 3 second increment per move
- Games per match: [TODO: 6-10 games]
- Color assignment: Alternating white/black each game

**Models Under Test:**

| Model ID | Generation | Tier | API Identifier |
|----------|------------|------|----------------|
| G3P | Gemini 3 | Pro | `gemini-3-pro-preview` |
| G3F | Gemini 3 | Flash | `gemini-3-flash-preview` |
| G25P | Gemini 2.5 | Pro | `gemini-2.5-pro-preview-05-06` |
| G25F | Gemini 2.5 | Flash | `gemini-2.5-flash-preview-05-20` |

**Data Collection:**

For each move, we record:
- `time_available_at_turn_start` — Clock reading when turn began
- `time_taken_seconds` — Duration of model response
- `thinking_tokens` — Number of reasoning tokens generated
- `response_with_thoughts` — Full reasoning trace
- `move_played` — The chess move in algebraic notation
- `board_state_before_move` — FEN position

Post-game, we run Stockfish analysis to compute:
- `centipawn_loss` — Quality difference from best move
- `is_blunder` — Whether move lost >100 centipawns

### 2.2 Pressure Categories

We categorize time remaining into four pressure levels:

| Level | Time Remaining | Expected Behavior |
|-------|---------------|-------------------|
| **Comfortable** | > 2 minutes | Full analysis, strategic planning |
| **Medium** | 1-2 minutes | Balanced, some urgency |
| **High** | 30s - 1 minute | Accelerated decisions |
| **Critical** | < 30 seconds | Survival mode, rapid moves |

### 2.3 Metrics

**Primary Metrics:**
- **Speed Adaptation Ratio** = (avg move time under high/critical pressure) / (avg move time at comfortable)
  - < 1.0 indicates speedup under pressure
- **Quality Degradation Ratio** = (avg centipawn loss under pressure) / (avg centipawn loss comfortable)
  - > 1.0 indicates quality decline
- **Thinking Reduction Ratio** = (avg thinking tokens under pressure) / (avg thinking tokens comfortable)
  - < 1.0 indicates reduced reasoning depth

**Secondary Metrics:**
- Blunder rate by pressure level
- Time-to-move variance
- Win rate correlation with time management

---

## 3. Experiments

### 3.1 Baseline Experiments (Self-Play)

Self-play matches establish baseline variance and reveal any inherent asymmetries.

#### 3.1.1 Gemini 3 Flash vs Itself

**Match ID:** `[TODO]`

**Purpose:** Establish baseline behavior for the primary model under study.

**Key Questions:**
- What is the natural variance in move time and quality?
- Does white vs black show different time management?
- Is behavior symmetric when facing an identical opponent?

**Results:**

| Metric | White | Black | Δ |
|--------|-------|-------|---|
| Win Rate | [TODO] | [TODO] | |
| Avg Move Time | [TODO]s | [TODO]s | |
| Speed Adaptation Ratio | [TODO] | [TODO] | |
| Avg Thinking Tokens | [TODO] | [TODO] | |

**Pressure Response Curve:**

> [TODO: Insert scatter plot - time_remaining vs move_time]

**Observations:**
- [TODO: Note any white/black asymmetries]
- [TODO: Note variance patterns]
- [TODO: Note adaptation behavior]

---

#### 3.1.2 Gemini 3 Pro vs Itself

**Match ID:** `[TODO]`

**Purpose:** Baseline for Pro-tier model.

**Results:**

| Metric | White | Black | Δ |
|--------|-------|-------|---|
| Win Rate | [TODO] | [TODO] | |
| Speed Adaptation Ratio | [TODO] | [TODO] | |
| Quality Degradation Ratio | [TODO] | [TODO] | |

**Observations:**
- [TODO]

---

#### 3.1.3 Gemini 2.5 Flash vs Itself

**Match ID:** `[TODO]`

**Purpose:** Prior-generation baseline for Flash tier.

**Results:**

| Metric | White | Black | Δ |
|--------|-------|-------|---|
| Win Rate | [TODO] | [TODO] | |
| Speed Adaptation Ratio | [TODO] | [TODO] | |

**Observations:**
- [TODO]

---

### 3.2 Within-Generation Comparisons (Pro vs Flash)

#### 3.2.1 Gemini 3 Pro vs Gemini 3 Flash

**Match ID:** `[TODO]`

**Purpose:** Compare how Pro and Flash handle pressure within the same generation.

**Hypotheses:**
- Pro will take more time overall
- Pro will maintain quality better under pressure
- Flash will adapt speed more aggressively

**Results:**

| Metric | G3 Pro | G3 Flash |
|--------|--------|----------|
| Win Rate | [TODO] | [TODO] |
| Avg Move Time (Comfortable) | [TODO]s | [TODO]s |
| Avg Move Time (Critical) | [TODO]s | [TODO]s |
| Speed Adaptation Ratio | [TODO] | [TODO] |
| Quality Degradation Ratio | [TODO] | [TODO] |
| Thinking Reduction Ratio | [TODO] | [TODO] |
| Blunder Rate (Comfortable) | [TODO]% | [TODO]% |
| Blunder Rate (Critical) | [TODO]% | [TODO]% |

**Time Pressure Response Comparison:**

> [TODO: Insert dual scatter plot - G3P vs G3F pressure response]

**Thinking Tokens by Pressure Level:**

> [TODO: Insert bar chart comparing thinking tokens at each pressure level]

**Key Insights:**
- [TODO: Which model adapted better?]
- [TODO: Quality vs speed tradeoff comparison]
- [TODO: Did tier difference predict behavior?]

---

#### 3.2.2 Gemini 2.5 Pro vs Gemini 2.5 Flash

**Match ID:** `[TODO]`

**Purpose:** Same comparison for prior generation.

**Results:**

| Metric | G25 Pro | G25 Flash |
|--------|---------|-----------|
| Speed Adaptation Ratio | [TODO] | [TODO] |
| Quality Degradation Ratio | [TODO] | [TODO] |

**Observations:**
- [TODO]

---

### 3.3 Cross-Generation Comparisons (Same Tier)

#### 3.3.1 Gemini 3 Flash vs Gemini 2.5 Flash

**Match ID:** `[TODO]`

**Purpose:** Has time management improved in the Flash tier across generations?

**Key Questions:**
- Does G3F show better awareness of time pressure?
- Has the speed/quality tradeoff improved?
- Are there qualitative differences in reasoning traces?

**Results:**

| Metric | G3 Flash | G2.5 Flash |
|--------|----------|------------|
| Win Rate | [TODO] | [TODO] |
| Speed Adaptation Ratio | [TODO] | [TODO] |
| Quality Degradation Ratio | [TODO] | [TODO] |

**Generational Progress:**

> [TODO: Insert comparison visualization]

**Observations:**
- [TODO: Evidence of improvement or regression]

---

#### 3.3.2 Gemini 3 Pro vs Gemini 2.5 Pro

**Match ID:** `[TODO]`

**Purpose:** Pro-tier evolution across generations.

**Results:**

| Metric | G3 Pro | G2.5 Pro |
|--------|--------|----------|
| Speed Adaptation Ratio | [TODO] | [TODO] |
| Quality Degradation Ratio | [TODO] | [TODO] |

**Observations:**
- [TODO]

---

### 3.4 Cross-Tier Cross-Generation

#### 3.4.1 Gemini 3 Flash vs Gemini 2.5 Pro

**Match ID:** `[TODO]`

**Purpose:** Is the new Flash model competitive with or better than the old Pro?

**Results:**

| Metric | G3 Flash | G2.5 Pro |
|--------|----------|----------|
| Win Rate | [TODO] | [TODO] |
| Speed Adaptation Ratio | [TODO] | [TODO] |
| Quality Degradation Ratio | [TODO] | [TODO] |

**Implications:**
- [TODO: What does this say about model evolution?]

---

### 3.5 Prompt Ablation Experiments

These experiments test whether time pressure prompts actually affect model behavior.

#### 3.5.1 G3 Flash: Standard Prompts vs No Time Prompts

**Experiment IDs:** 1 (baseline) vs 10 (no time prompts)

**Purpose:** Does including time pressure information in prompts make a difference, or do models adapt based on other cues?

**Conditions:**
- **Standard:** Full time pressure information in prompt
- **Ablation:** Time pressure prompts completely removed

**Results:**

| Metric | With Time Prompts | Without Time Prompts | Δ |
|--------|-------------------|----------------------|---|
| Speed Adaptation Ratio | [TODO] | [TODO] | |
| Quality Degradation | [TODO] | [TODO] | |
| Thinking Reduction | [TODO] | [TODO] | |
| Explicit Time Mentions | [TODO]% | [TODO]% | |

**Key Questions:**
- Does adaptation disappear without prompts?
- Does the model mention time even when not told about it?
- Is there any residual adaptation behavior?

**Observations:**
- [TODO]

---

#### 3.5.2 G3 Flash: Standard Prompts vs Dramatic Prompts

**Experiment IDs:** 1 (baseline) vs 11 (dramatic prompts)

**Purpose:** Do ALL-CAPS dramatic prompts improve adaptation or hurt quality?

**Conditions:**
- **Standard:** Neutral time information
- **Dramatic:** ALL-CAPS urgency ("🚨 CRITICAL TIME EMERGENCY!!!")

**Results:**

| Metric | Standard Prompts | Dramatic Prompts | Δ |
|--------|------------------|------------------|---|
| Speed Adaptation Ratio | [TODO] | [TODO] | |
| Quality Degradation | [TODO] | [TODO] | |
| Blunder Rate (Critical) | [TODO]% | [TODO]% | |

**Key Questions:**
- Do dramatic prompts cause faster moves?
- Do they also cause more blunders?
- Is there a speed/quality tradeoff?

**Observations:**
- [TODO]

---

### 3.6 Response Feedback Experiments

These experiments test whether models can adapt better when given feedback about their previous response.

#### 3.6.1 G3 Flash: Standard vs Response Feedback

**Experiment IDs:** 1 (baseline) vs 20 (with response feedback)

**Purpose:** Can models calibrate thinking depth when told their token/second rate?

**Feedback Information Provided:**
```
📊 YOUR PREVIOUS RESPONSE ANALYSIS:
• Your last move took 8.5 seconds
• You used 12,000 thinking tokens
• Your thinking speed: ~1,400 tokens/second
```

**Results:**

| Metric | No Feedback | With Feedback | Δ |
|--------|-------------|---------------|---|
| Speed Adaptation Ratio | [TODO] | [TODO] | |
| Move Time Variance | [TODO] | [TODO] | |
| Time Forfeit Rate | [TODO] | [TODO] | |

**Key Questions:**
- Does feedback enable more consistent timing?
- Can models use token rate to estimate time consumption?
- Is there evidence of deliberate adaptation?

**Observations:**
- [TODO]

---

#### 3.6.2 G3 Flash: Response Feedback + Efficiency Guidance

**Experiment IDs:** 20 vs 21 (with efficiency guidance)

**Purpose:** Does explicit guidance about affordable tokens improve adaptation?

**Additional Guidance Provided:**
```
⚡ EFFICIENCY GUIDANCE:
• At your current speed, generating 10,000 tokens would take ~7s
• You have 45s remaining
• ⚠️ Consider shorter reasoning to preserve time!
```

**Results:**

| Metric | Feedback Only | Feedback + Guidance | Δ |
|--------|---------------|---------------------|---|
| Thinking Tokens (Critical) | [TODO] | [TODO] | |
| Move Time (Critical) | [TODO]s | [TODO]s | |
| Time Forfeit Rate | [TODO] | [TODO] | |

**Observations:**
- [TODO]

---

#### 3.6.3 G3 Flash: Full Awareness (All Features)

**Experiment ID:** 22

**Purpose:** Maximum prompt engineering: dramatic prompts + response feedback + efficiency guidance

**Results:**

| Metric | Baseline (1) | Full Awareness (22) | Δ |
|--------|--------------|---------------------|---|
| Speed Adaptation Ratio | [TODO] | [TODO] | |
| Quality Degradation | [TODO] | [TODO] | |
| Thinking Reduction | [TODO] | [TODO] | |
| Time Forfeits | [TODO] | [TODO] | |

**Key Finding:**
- [TODO: Is full awareness better or does it overwhelm the model?]

---

### 3.7 Offline Position Evaluation (Controlled Experiments)

Unlike live games where board positions evolve naturally, offline evaluation provides **controlled, reproducible experiments** where we test the same position under different time constraints.

#### 3.7.1 Methodology

**Approach:**
- Present identical chess positions to models with varying "time remaining" values
- Sample each (position, time_level) condition multiple times to measure variance
- Compute centipawn loss via Stockfish to assess move quality
- No opponent moves—pure decision analysis

**Time Levels Tested:**
| Level | Time Remaining | Pressure Category |
|-------|---------------|-------------------|
| T1 | 300s (5 min) | Comfortable |
| T2 | 120s (2 min) | Comfortable |
| T3 | 60s (1 min) | Medium |
| T4 | 30s | High |
| T5 | 15s | Critical |

**Position Dataset:**
- **Opening positions:** Standard opening theory positions (moves 5-12)
- **Middlegame tactical:** Positions with clear tactical motifs
- **Middlegame positional:** Positions requiring strategic judgment
- **Endgame:** Various piece configurations
- **Ambiguous positions:** Multiple reasonable candidate moves

**Metrics:**
- Response time (seconds)
- Thinking tokens used
- Move played (SAN notation)
- Centipawn loss from best move
- Move consistency (unique moves across samples)

---

#### 3.7.2 G3 Flash Position Evaluation

**Session ID:** `[TODO]`

**Results Table:**

| Time Remaining | Avg Response Time | Avg Thinking Tokens | Avg CPL | Best Move Rate |
|----------------|-------------------|---------------------|---------|----------------|
| 300s | [TODO]s | [TODO] | [TODO] | [TODO]% |
| 120s | [TODO]s | [TODO] | [TODO] | [TODO]% |
| 60s | [TODO]s | [TODO] | [TODO] | [TODO]% |
| 30s | [TODO]s | [TODO] | [TODO] | [TODO]% |
| 15s | [TODO]s | [TODO] | [TODO] | [TODO]% |

**Variance Analysis:**

| Time Level | Response Time CV | Unique Moves/3 Samples | Token CV |
|------------|------------------|------------------------|----------|
| 300s | [TODO] | [TODO] | [TODO] |
| 60s | [TODO] | [TODO] | [TODO] |
| 15s | [TODO] | [TODO] | [TODO] |

> [TODO: Insert response time vs time remaining curve with error bars]

**Key Observations:**
- [TODO: Does the model speed up with less time?]
- [TODO: Does quality degrade?]
- [TODO: How consistent are moves across samples?]

---

#### 3.7.3 G3 Pro Position Evaluation

**Session ID:** `[TODO]`

**Results Table:**

| Time Remaining | Avg Response Time | Avg Thinking Tokens | Avg CPL |
|----------------|-------------------|---------------------|---------|
| 300s | [TODO]s | [TODO] | [TODO] |
| 60s | [TODO]s | [TODO] | [TODO] |
| 15s | [TODO]s | [TODO] | [TODO] |

---

#### 3.7.4 Model Comparison (Same Positions)

**Cross-Model Comparison at Each Time Level:**

> [TODO: Insert heatmap comparing models × time levels]

| Metric | G3 Flash (300s) | G3 Flash (15s) | G3 Pro (300s) | G3 Pro (15s) |
|--------|-----------------|----------------|---------------|--------------|
| Avg Response Time | [TODO]s | [TODO]s | [TODO]s | [TODO]s |
| Avg Thinking Tokens | [TODO] | [TODO] | [TODO] | [TODO] |
| Best Move Rate | [TODO]% | [TODO]% | [TODO]% | [TODO]% |

**Key Findings:**
- [TODO: Which model adapts better while preserving quality?]
- [TODO: Is Flash's speed advantage offset by lower quality?]

---

#### 3.7.5 Position Category Analysis

**Performance by Position Type:**

| Category | Model | Time Level | Avg CPL | Best Move Rate |
|----------|-------|------------|---------|----------------|
| Opening | G3 Flash | 300s | [TODO] | [TODO]% |
| Opening | G3 Flash | 15s | [TODO] | [TODO]% |
| Tactical | G3 Flash | 300s | [TODO] | [TODO]% |
| Tactical | G3 Flash | 15s | [TODO] | [TODO]% |
| Endgame | G3 Flash | 300s | [TODO] | [TODO]% |
| Endgame | G3 Flash | 15s | [TODO] | [TODO]% |

**Key Questions:**
- [TODO: Are tactical positions more affected by time pressure?]
- [TODO: Are endgames more robust to pressure?]

---

#### 3.7.6 Offline vs Live Comparison

Comparing offline evaluation results with live game behavior:

| Metric | Offline (Controlled) | Live Games | Notes |
|--------|----------------------|------------|-------|
| Speed Adaptation Ratio | [TODO] | [TODO] | |
| Quality Degradation | [TODO] | [TODO] | |
| Thinking Reduction | [TODO] | [TODO] | |

**Key Question:** Does the artificial "time remaining" prompt produce similar adaptation to real game pressure?

---

## 4. Analysis

### 4.1 Speed Adaptation Across Models

**Summary Table:**

| Model | Avg Time (Comfortable) | Avg Time (Critical) | Speed Adaptation Ratio |
|-------|------------------------|---------------------|------------------------|
| G3 Pro | [TODO]s | [TODO]s | [TODO] |
| G3 Flash | [TODO]s | [TODO]s | [TODO] |
| G2.5 Pro | [TODO]s | [TODO]s | [TODO] |
| G2.5 Flash | [TODO]s | [TODO]s | [TODO] |

**Aggregate Visualization:**

> [TODO: Insert grouped bar chart or radar chart comparing all models]

**Key Finding:**
- [TODO: Which models adapted most? Did any fail to adapt?]

---

### 4.2 Quality Under Pressure

**Centipawn Loss by Pressure Level:**

| Model | CPL (Comfortable) | CPL (Medium) | CPL (High) | CPL (Critical) |
|-------|-------------------|--------------|------------|----------------|
| G3 Pro | [TODO] | [TODO] | [TODO] | [TODO] |
| G3 Flash | [TODO] | [TODO] | [TODO] | [TODO] |
| G2.5 Pro | [TODO] | [TODO] | [TODO] | [TODO] |
| G2.5 Flash | [TODO] | [TODO] | [TODO] | [TODO] |

**Blunder Rate by Pressure Level:**

> [TODO: Insert heatmap showing blunder rates]

**Key Finding:**
- [TODO: How does quality degrade? Linear or threshold effect?]

---

### 4.3 Thinking Depth Analysis

**Thinking Tokens by Pressure Level:**

| Model | Tokens (Comfortable) | Tokens (Critical) | Reduction Ratio |
|-------|----------------------|-------------------|-----------------|
| G3 Pro | [TODO] | [TODO] | [TODO] |
| G3 Flash | [TODO] | [TODO] | [TODO] |
| G2.5 Pro | [TODO] | [TODO] | [TODO] |
| G2.5 Flash | [TODO] | [TODO] | [TODO] |

**Reasoning Efficiency (Tokens per Second):**

> [TODO: Insert line chart showing efficiency over time remaining]

---

### 4.4 Reasoning Content Analysis

**Sample Reasoning Traces Under Pressure:**

**Comfortable (>2 min remaining):**
```
[TODO: Insert example reasoning trace]
```

**Critical (<30s remaining):**
```
[TODO: Insert example reasoning trace]
```

**Qualitative Patterns Observed:**
- [ ] Model explicitly mentions time remaining
- [ ] Model truncates candidate move analysis
- [ ] Model prioritizes tactical over strategic considerations
- [ ] Model shows urgency language ("must", "quickly", "immediately")
- [ ] Model explicitly trades quality for speed

---

### 4.5 White vs Black Asymmetry

From self-play experiments:

| Model | White Win Rate | Black Win Rate | White Speed Adapt | Black Speed Adapt |
|-------|----------------|----------------|-------------------|-------------------|
| G3F vs G3F | [TODO]% | [TODO]% | [TODO] | [TODO] |
| G3P vs G3P | [TODO]% | [TODO]% | [TODO] | [TODO] |

**Observations:**
- [TODO: Does opening as white vs black affect time management?]

---

## 5. Discussion

### 5.1 Do LLMs Really Adapt?

[TODO: Synthesize findings across experiments]

Evidence FOR adaptation:
- [TODO]

Evidence AGAINST adaptation:
- [TODO]

### 5.2 Model Tier Differences

[TODO: Compare Pro vs Flash behavior patterns]

### 5.3 Generational Improvements

[TODO: Has Gemini 3 improved over 2.5?]

### 5.4 The Role of Prompting

[TODO: Synthesize findings from ablation experiments]

**Key Questions Answered:**
- Does adaptation require explicit time prompts? [TODO]
- Do dramatic prompts help or hurt? [TODO]
- What is the optimal prompt strategy? [TODO]

### 5.5 Recurrent Awareness: Can Models Learn During a Game?

[TODO: Synthesize findings from response feedback experiments]

**Key Questions Answered:**
- Can models use feedback about their token rate? [TODO]
- Does efficiency guidance improve time management? [TODO]
- Is there evidence of within-game learning? [TODO]

### 5.6 Comparison to Human Behavior

[TODO: How does LLM time management compare to human blitz players?]

### 5.7 Limitations

- Limited to Gemini model family
- Single time control (5+3)
- Sample size considerations
- Stockfish evaluation limitations for unusual positions
- Prompt sensitivity

---

## 6. Conclusions

### 6.1 Summary of Findings

1. **RQ1 (Speed):** [TODO: One-sentence summary]
2. **RQ2 (Thinking):** [TODO: One-sentence summary]
3. **RQ3 (Quality):** [TODO: One-sentence summary]
4. **RQ4 (Reasoning):** [TODO: One-sentence summary]
5. **RQ5 (Model Differences):** [TODO: One-sentence summary]
6. **RQ6 (Prompt Engineering):** [TODO: One-sentence summary]
7. **RQ7 (Recurrent Awareness):** [TODO: One-sentence summary]
8. **RQ8 (Controlled Analysis):** [TODO: One-sentence summary]

### 6.2 Implications

For LLM Development:
- [TODO]

For Using LLMs in Time-Sensitive Applications:
- [TODO]

### 6.3 Future Work

- Test with additional model families (Claude, GPT)
- Vary time controls (bullet, rapid, classical)
- Analyze opening vs middlegame vs endgame pressure handling
- Investigate prompt engineering for better time awareness
- Study rethinking/self-correction under pressure

---

## Appendix A: Experiment Log

### Phase 1-4: Model Comparisons

| # | Match ID | Models | Games | Date | Status |
|---|----------|--------|-------|------|--------|
| 1 | | G3F vs G3F (baseline) | 8 | | Pending |
| 2 | | G3P vs G3P (baseline) | 6 | | Pending |
| 3 | | G25F vs G25F (baseline) | 6 | | Pending |
| 4 | | G3P vs G3F | 8 | | Pending |
| 5 | | G25P vs G25F | 6 | | Pending |
| 6 | | G3F vs G25F | 8 | | Pending |
| 7 | | G3P vs G25P | 6 | | Pending |
| 8 | | G3F vs G25P | 6 | | Pending |

### Phase 5: Prompt Ablation

| # | Match ID | Models | Config | Games | Date | Status |
|---|----------|--------|--------|-------|------|--------|
| 10 | | G3F vs G3F | No time prompts | 8 | | Pending |
| 11 | | G3F vs G3F | Dramatic prompts | 8 | | Pending |

### Phase 6: Response Feedback

| # | Match ID | Models | Config | Games | Date | Status |
|---|----------|--------|--------|-------|------|--------|
| 20 | | G3F vs G3F | Response feedback | 8 | | Pending |
| 21 | | G3F vs G3F | Feedback + guidance | 8 | | Pending |
| 22 | | G3F vs G3F | Full awareness | 8 | | Pending |

### Phase 7: Comparison with Features

| # | Match ID | Models | Config | Games | Date | Status |
|---|----------|--------|--------|-------|------|--------|
| 30 | | G3P vs G3F | With feedback | 8 | | Pending |

### Phase 8: Offline Position Evaluation

| # | Session ID | Model | Positions | Samples/Cond | Time Levels | Date | Status |
|---|------------|-------|-----------|--------------|-------------|------|--------|
| 40 | | G3 Flash | 20 (standard) | 3 | 300,120,60,30,15 | | Pending |
| 41 | | G3 Pro | 20 (standard) | 3 | 300,120,60,30,15 | | Pending |
| 42 | | G25 Flash | 20 (standard) | 3 | 300,120,60,30,15 | | Pending |
| 43 | | G3F (no time) | 20 | 3 | 300,60,15 | | Pending |
| 44 | | G3F (dramatic) | 20 | 3 | 300,60,15 | | Pending |
| 45 | | G3F (tactical) | tactical | 3 | 300,60,15 | | Pending |
| 46 | | G3F (endgame) | endgame | 3 | 300,60,15 | | Pending |

## Appendix B: Raw Data Location

**Match data stored in:** `game_arena/_results/`

Each match folder contains:
- `metadata.json` — Match configuration and results
- `game_N_moves.csv` — Move-by-move data for each game
- `games_summary.csv` — Per-game outcomes
- `complete_move_analysis.csv` — Stockfish evaluation (if available)

**Offline evaluation data stored in:** `game_arena/_results/offline_eval/`

Each session file contains:
- `session_id` — Unique identifier
- `model_id` — Model evaluated
- `config` — Evaluation configuration
- `results` — Array of individual evaluation results

## Appendix C: Code References

- Analysis Service: `web/backend/services/analysis_service.py`
- Data Collector: `game_arena/blitz/data/collector.py`
- Match Runner: `game_arena/blitz/match.py`
- Offline Evaluator: `game_arena/blitz/offline_eval/evaluator.py`
- Position Dataset: `game_arena/blitz/offline_eval/position_dataset.py`
- Offline Analysis: `game_arena/blitz/offline_eval/analysis.py`
- Dashboard: `http://localhost:3000`

## Appendix D: Running Offline Evaluation

**Standard evaluation:**
```bash
python scripts/run_offline_eval.py --model gemini-3-flash --samples 3
```

**Compare multiple models:**
```bash
python scripts/run_offline_eval.py --compare gemini-3-pro gemini-3-flash --samples 3
```

**Filter by position category:**
```bash
python scripts/run_offline_eval.py --model gemini-3-flash --category tactical
```

**Without time pressure prompts (ablation):**
```bash
python scripts/run_offline_eval.py --model gemini-3-flash --noenable_time_pressure_prompt
```

**Generate analysis report:**
```bash
python scripts/run_offline_eval.py --analyze --generate_notebook
```

---

*Report generated for Game Arena LLM Chess Analysis*
*Last updated: [TODO: Date]*

