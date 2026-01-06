"""Time-Constrained Reasoning Insights for LLM Chess Agents."""
from typing import Optional
from fastapi import APIRouter, Request
from pydantic import BaseModel
import numpy as np
from collections import defaultdict

router = APIRouter(tags=["insights"])


# === DATA MODELS ===

class DegradationPoint(BaseModel):
    """A point on the degradation curve."""
    time_bucket_start: int  # seconds
    time_bucket_end: int
    bucket_label: str
    move_count: int
    avg_cp_loss: float
    median_cp_loss: float
    blunder_rate: float
    p90_cp_loss: float  # 90th percentile


class ModelDegradationCurve(BaseModel):
    """Degradation curve for a single model."""
    model_name: str
    total_moves: int
    curve: list[DegradationPoint]
    degradation_ratio: float  # low_time_quality / high_time_quality
    critical_threshold: Optional[int]  # time where quality collapses


class EfficiencyMetrics(BaseModel):
    """Efficiency analysis for a model."""
    model_name: str
    total_moves: int
    avg_time_per_move: float
    avg_cp_loss: float
    avg_tokens_per_move: Optional[float]
    
    # Quality per resource
    quality_per_second: float  # (100 - avg_cp_loss) / avg_time
    quality_per_token: Optional[float]  # (100 - avg_cp_loss) / avg_tokens
    
    # Diminishing returns analysis
    efficiency_by_time_spent: list[dict]  # [{time_range, quality, efficiency}]
    optimal_time_range: Optional[str]  # Best quality/time tradeoff


class TimeAllocationStats(BaseModel):
    """How a model allocates time across game phases."""
    model_name: str
    opening_avg_time: float  # moves 1-15
    middlegame_avg_time: float  # moves 16-35
    endgame_avg_time: float  # moves 36+
    opening_avg_quality: float
    middlegame_avg_quality: float
    endgame_avg_quality: float
    # Correlation with complexity
    complexity_time_correlation: float  # Should be positive if smart
    complexity_quality_correlation: float


class PositionPerformance(BaseModel):
    """Performance by position characteristics."""
    model_name: str
    # By complexity (number of legal moves)
    simple_pos_quality: float  # <20 legal moves
    complex_pos_quality: float  # 30+ legal moves
    complexity_penalty: float  # How much worse in complex positions
    
    # By criticality (eval sharpness)
    routine_pos_quality: float  # Low sharpness
    critical_pos_quality: float  # High sharpness
    criticality_penalty: float
    
    # Time spent on critical vs routine
    critical_time_ratio: float  # time_on_critical / time_on_routine


class OutcomeCorrelate(BaseModel):
    """What predicts winning."""
    factor: str
    correlation_with_win: float
    sample_size: int
    description: str


class MatchupSummary(BaseModel):
    """Summary of a specific model matchup."""
    model_a: str
    model_b: str
    is_same_model: bool
    total_games: int
    model_a_wins: int
    model_b_wins: int
    draws: int
    model_a_win_rate: float
    avg_game_length: float
    notes: list[str]  # Experiment notes for games in this matchup


class TournamentSummary(BaseModel):
    """Overall tournament/matchup summary."""
    total_unique_matchups: int
    same_model_matchups: int
    cross_model_matchups: int
    matchups: list[MatchupSummary]
    # Derived rankings
    model_rankings: list[dict]  # [{model, wins, losses, win_rate}]


class InsightsResponse(BaseModel):
    """Complete time-constrained reasoning insights."""
    total_matches: int
    total_games: int
    total_moves: int
    total_analyzed_moves: int
    outliers_capped: int  # Number of moves with CP loss capped at threshold
    cp_loss_cap: int  # The cap threshold used
    same_model_games_excluded: int  # Games where same model played itself
    
    # 0. Tournament overview
    tournament: Optional[TournamentSummary] = None
    
    # 1. Degradation curves per model
    degradation_curves: list[ModelDegradationCurve]
    
    # 2. Efficiency metrics per model
    efficiency_metrics: list[EfficiencyMetrics]
    
    # 3. Time allocation per model
    time_allocation: list[TimeAllocationStats]
    
    # 4. Position-dependent performance per model
    position_performance: list[PositionPerformance]
    
    # 5. Outcome correlates
    outcome_correlates: list[OutcomeCorrelate]
    
    # Raw data for custom charts
    scatter_data: list[dict]


def calculate_correlation(x: list, y: list) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) < 3 or len(y) < 3:
        return 0.0
    x_arr = np.array(x)
    y_arr = np.array(y)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def percentile(data: list, p: float) -> float:
    """Calculate percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def median(data: list) -> float:
    """Calculate median of a list."""
    return percentile(data, 50)


def remove_outliers_iqr(data: list, multiplier: float = 1.5) -> list:
    """Remove outliers using IQR method."""
    if len(data) < 4:
        return data
    q1 = percentile(data, 25)
    q3 = percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return [x for x in data if lower <= x <= upper]


def cap_outliers(data: list, cap_percentile: float = 95) -> list:
    """Cap outliers at a percentile threshold."""
    if not data:
        return data
    cap = percentile(data, cap_percentile)
    return [min(x, cap) for x in data]


# Outlier thresholds for chess analysis
CP_LOSS_CAP = 200  # CP loss above this is just "blunder" - no meaningful difference
TIME_TAKEN_CAP = 120  # Moves taking >2 min are likely connection issues


@router.get("", response_model=InsightsResponse)
async def get_insights(request: Request, model_filter: Optional[str] = None):
    """Get time-constrained reasoning insights."""
    match_service = request.app.state.match_service
    matches = match_service.get_all_matches(limit=1000)
    
    # Collect all data
    all_moves = []
    game_outcomes = []  # For outcome analysis
    total_matches = 0
    total_games = 0
    
    for match in matches:
        if match.status != "completed":
            continue
        
        total_matches += 1
        match_detail = match_service.get_match(match.match_id)
        if not match_detail:
            continue
        
        for game in match_detail.games:
            total_games += 1
            game_detail = match_service.get_game(match.match_id, game.game_number)
            if not game_detail or not game_detail.moves:
                continue
            
            # Track game-level stats for outcome analysis
            white_model = game_detail.white_model
            black_model = game_detail.black_model
            winner = game.winner  # "model_a", "model_b", "draw"
            
            white_moves = []
            black_moves = []
            
            for move in game_detail.moves:
                move_model = white_model if move.color == "white" else black_model
                
                if model_filter and model_filter not in move_model:
                    continue
                
                move_data = {
                    "match_id": match.match_id,
                    "game_number": game.game_number,
                    "move_number": move.move_number,
                    "model": move_model,
                    "color": move.color,
                    "time_taken": move.time_taken,
                    "time_remaining": move.time_remaining,
                    "thinking_tokens": move.thinking_tokens,
                    "cp_loss": move.centipawn_loss,
                    "is_blunder": move.is_blunder,
                    "num_legal_moves": move.num_legal_moves,
                    "eval_sharpness": move.eval_sharpness,
                }
                all_moves.append(move_data)
                
                if move.color == "white":
                    white_moves.append(move_data)
                else:
                    black_moves.append(move_data)
            
            # Game outcome data
            if white_moves or black_moves:
                white_analyzed = [m for m in white_moves if m["cp_loss"] is not None]
                black_analyzed = [m for m in black_moves if m["cp_loss"] is not None]
                
                game_outcomes.append({
                    "white_model": white_model,
                    "black_model": black_model,
                    "winner": winner,
                    "white_won": winner == "model_a" if game_detail.white_model == match.model_a else winner == "model_b",
                    "white_avg_cp": sum(m["cp_loss"] for m in white_analyzed) / len(white_analyzed) if white_analyzed else None,
                    "black_avg_cp": sum(m["cp_loss"] for m in black_analyzed) / len(black_analyzed) if black_analyzed else None,
                    "white_blunders": sum(1 for m in white_analyzed if m["is_blunder"]),
                    "black_blunders": sum(1 for m in black_analyzed if m["is_blunder"]),
                    "white_avg_time": sum(m["time_taken"] or 0 for m in white_moves) / len(white_moves) if white_moves else 0,
                    "black_avg_time": sum(m["time_taken"] or 0 for m in black_moves) / len(black_moves) if black_moves else 0,
                    "white_end_time": white_moves[-1]["time_remaining"] if white_moves else None,
                    "black_end_time": black_moves[-1]["time_remaining"] if black_moves else None,
                })
    
    # === BUILD TOURNAMENT SUMMARY ===
    matchup_stats = defaultdict(lambda: {
        "games": 0, "model_a_wins": 0, "model_b_wins": 0, "draws": 0,
        "game_lengths": [], "notes": set()
    })
    same_model_game_count = 0
    
    for match in matches:
        if match.status != "completed":
            continue
        
        model_a = match.model_a
        model_b = match.model_b
        is_same_model = model_a == model_b
        
        # Normalize matchup key (alphabetical order)
        if model_a <= model_b:
            key = (model_a, model_b)
            swap = False
        else:
            key = (model_b, model_a)
            swap = True
        
        match_detail = match_service.get_match(match.match_id)
        if not match_detail:
            continue
        
        if match.notes:
            matchup_stats[key]["notes"].add(match.notes)
        
        for game in match_detail.games:
            matchup_stats[key]["games"] += 1
            matchup_stats[key]["game_lengths"].append(game.total_moves)
            
            if is_same_model:
                same_model_game_count += 1
                matchup_stats[key]["draws"] += 1  # Same model = no meaningful winner
            else:
                # Determine winner relative to normalized key
                if game.winner == "model_a":
                    winner_model = match.model_a
                elif game.winner == "model_b":
                    winner_model = match.model_b
                else:
                    winner_model = None
                
                if winner_model is None:
                    matchup_stats[key]["draws"] += 1
                elif winner_model == key[0]:
                    matchup_stats[key]["model_a_wins"] += 1
                else:
                    matchup_stats[key]["model_b_wins"] += 1
    
    # Build matchup summaries
    matchup_summaries = []
    model_records = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})
    
    for (model_a, model_b), stats in matchup_stats.items():
        is_same = model_a == model_b
        
        matchup_summaries.append(MatchupSummary(
            model_a=model_a,
            model_b=model_b,
            is_same_model=is_same,
            total_games=stats["games"],
            model_a_wins=stats["model_a_wins"],
            model_b_wins=stats["model_b_wins"],
            draws=stats["draws"],
            model_a_win_rate=stats["model_a_wins"] / stats["games"] if stats["games"] > 0 else 0.5,
            avg_game_length=sum(stats["game_lengths"]) / len(stats["game_lengths"]) if stats["game_lengths"] else 0,
            notes=list(stats["notes"]),
        ))
        
        if not is_same:
            model_records[model_a]["wins"] += stats["model_a_wins"]
            model_records[model_a]["losses"] += stats["model_b_wins"]
            model_records[model_a]["games"] += stats["games"]
            model_records[model_b]["wins"] += stats["model_b_wins"]
            model_records[model_b]["losses"] += stats["model_a_wins"]
            model_records[model_b]["games"] += stats["games"]
    
    # Sort matchups by games played
    matchup_summaries.sort(key=lambda x: x.total_games, reverse=True)
    
    # Build model rankings
    model_rankings = []
    for model, record in model_records.items():
        total = record["wins"] + record["losses"]
        model_rankings.append({
            "model": model,
            "wins": record["wins"],
            "losses": record["losses"],
            "games": record["games"],
            "win_rate": record["wins"] / total if total > 0 else 0.5,
        })
    model_rankings.sort(key=lambda x: x["win_rate"], reverse=True)
    
    same_model_matchups = sum(1 for m in matchup_summaries if m.is_same_model)
    cross_model_matchups = len(matchup_summaries) - same_model_matchups
    
    tournament_summary = TournamentSummary(
        total_unique_matchups=len(matchup_summaries),
        same_model_matchups=same_model_matchups,
        cross_model_matchups=cross_model_matchups,
        matchups=matchup_summaries,
        model_rankings=model_rankings,
    )
    
    analyzed_moves = [m for m in all_moves if m["cp_loss"] is not None]
    
    # === OUTLIER REMOVAL ===
    # Cap extreme CP losses - values above 200 are all just "blunders" with no meaningful difference
    for m in analyzed_moves:
        if m["cp_loss"] > CP_LOSS_CAP:
            m["cp_loss_raw"] = m["cp_loss"]  # Keep original for reference
            m["cp_loss"] = CP_LOSS_CAP
        if m["time_taken"] and m["time_taken"] > TIME_TAKEN_CAP:
            m["time_taken_raw"] = m["time_taken"]
            m["time_taken"] = TIME_TAKEN_CAP
    
    # Count outliers for reporting
    outlier_count = sum(1 for m in analyzed_moves if m.get("cp_loss_raw") is not None)
    
    # Group by model
    model_moves = defaultdict(list)
    for m in analyzed_moves:
        model_moves[m["model"]].append(m)
    
    # === 1. DEGRADATION CURVES ===
    time_buckets = [
        (0, 30, "0-30s (Critical)"),
        (30, 60, "30-60s (Low)"),
        (60, 120, "1-2min (Medium)"),
        (120, 180, "2-3min (Comfortable)"),
        (180, 300, "3-5min (High)"),
        (300, 9999, "5min+ (Abundant)"),
    ]
    
    degradation_curves = []
    for model, moves in model_moves.items():
        curve_points = []
        high_time_cp = []
        low_time_cp = []
        
        for start, end, label in time_buckets:
            bucket_moves = [m for m in moves 
                          if m["time_remaining"] is not None 
                          and start <= m["time_remaining"] < end]
            
            if not bucket_moves:
                continue
            
            cp_losses = [m["cp_loss"] for m in bucket_moves]
            
            # Track for degradation ratio
            if start >= 180:
                high_time_cp.extend(cp_losses)
            elif start < 60:
                low_time_cp.extend(cp_losses)
            
            curve_points.append(DegradationPoint(
                time_bucket_start=start,
                time_bucket_end=end,
                bucket_label=label,
                move_count=len(bucket_moves),
                avg_cp_loss=sum(cp_losses) / len(cp_losses),
                median_cp_loss=median(cp_losses),
                blunder_rate=sum(1 for m in bucket_moves if m["is_blunder"]) / len(bucket_moves),
                p90_cp_loss=percentile(cp_losses, 90),
            ))
        
        # Calculate degradation ratio
        high_avg = sum(high_time_cp) / len(high_time_cp) if high_time_cp else 1
        low_avg = sum(low_time_cp) / len(low_time_cp) if low_time_cp else high_avg
        degradation_ratio = low_avg / high_avg if high_avg > 0 else 1.0
        
        # Find critical threshold (where blunder rate jumps)
        critical_threshold = None
        for i, point in enumerate(curve_points):
            if point.blunder_rate > 0.15:  # 15% blunder rate threshold
                critical_threshold = point.time_bucket_end
                break
        
        degradation_curves.append(ModelDegradationCurve(
            model_name=model,
            total_moves=len(moves),
            curve=curve_points,
            degradation_ratio=degradation_ratio,
            critical_threshold=critical_threshold,
        ))
    
    # Sort by degradation ratio (best first)
    degradation_curves.sort(key=lambda x: x.degradation_ratio)
    
    # === 2. EFFICIENCY METRICS ===
    time_spent_buckets = [
        (0, 10, "0-10s"),
        (10, 20, "10-20s"),
        (20, 30, "20-30s"),
        (30, 45, "30-45s"),
        (45, 60, "45-60s"),
        (60, 9999, "60s+"),
    ]
    
    efficiency_metrics = []
    for model, moves in model_moves.items():
        timed_moves = [m for m in moves if m["time_taken"] is not None and m["time_taken"] > 0]
        token_moves = [m for m in moves if m["thinking_tokens"] is not None and m["thinking_tokens"] > 0]
        
        avg_time = sum(m["time_taken"] for m in timed_moves) / len(timed_moves) if timed_moves else 0
        avg_cp = sum(m["cp_loss"] for m in moves) / len(moves) if moves else 0
        avg_tokens = sum(m["thinking_tokens"] for m in token_moves) / len(token_moves) if token_moves else None
        
        # Quality = 100 - cp_loss (so higher is better)
        quality_per_sec = (100 - avg_cp) / avg_time if avg_time > 0 else 0
        quality_per_token = (100 - avg_cp) / avg_tokens if avg_tokens and avg_tokens > 0 else None
        
        # Efficiency by time spent
        efficiency_by_time = []
        best_efficiency = 0
        optimal_range = None
        
        for start, end, label in time_spent_buckets:
            bucket_moves = [m for m in timed_moves if start <= m["time_taken"] < end]
            if not bucket_moves:
                continue
            
            bucket_cp = sum(m["cp_loss"] for m in bucket_moves) / len(bucket_moves)
            bucket_time = sum(m["time_taken"] for m in bucket_moves) / len(bucket_moves)
            bucket_quality = 100 - bucket_cp
            bucket_efficiency = bucket_quality / bucket_time if bucket_time > 0 else 0
            
            efficiency_by_time.append({
                "time_range": label,
                "move_count": len(bucket_moves),
                "avg_quality": bucket_quality,
                "avg_time": bucket_time,
                "efficiency": bucket_efficiency,
            })
            
            if bucket_efficiency > best_efficiency and len(bucket_moves) >= 5:
                best_efficiency = bucket_efficiency
                optimal_range = label
        
        efficiency_metrics.append(EfficiencyMetrics(
            model_name=model,
            total_moves=len(moves),
            avg_time_per_move=avg_time,
            avg_cp_loss=avg_cp,
            avg_tokens_per_move=avg_tokens,
            quality_per_second=quality_per_sec,
            quality_per_token=quality_per_token,
            efficiency_by_time_spent=efficiency_by_time,
            optimal_time_range=optimal_range,
        ))
    
    efficiency_metrics.sort(key=lambda x: x.quality_per_second, reverse=True)
    
    # === 3. TIME ALLOCATION ===
    time_allocation = []
    for model, moves in model_moves.items():
        opening = [m for m in moves if m["move_number"] <= 15]
        middlegame = [m for m in moves if 16 <= m["move_number"] <= 35]
        endgame = [m for m in moves if m["move_number"] > 35]
        
        def phase_stats(phase_moves):
            timed = [m for m in phase_moves if m["time_taken"] is not None]
            return (
                sum(m["time_taken"] for m in timed) / len(timed) if timed else 0,
                sum(m["cp_loss"] for m in phase_moves) / len(phase_moves) if phase_moves else 0,
            )
        
        op_time, op_qual = phase_stats(opening)
        mid_time, mid_qual = phase_stats(middlegame)
        end_time, end_qual = phase_stats(endgame)
        
        # Complexity-time correlation
        complexity_moves = [m for m in moves if m["num_legal_moves"] is not None and m["time_taken"] is not None]
        comp_time_corr = calculate_correlation(
            [m["num_legal_moves"] for m in complexity_moves],
            [m["time_taken"] for m in complexity_moves]
        ) if complexity_moves else 0
        
        comp_qual_corr = calculate_correlation(
            [m["num_legal_moves"] for m in complexity_moves],
            [m["cp_loss"] for m in complexity_moves]
        ) if complexity_moves else 0
        
        time_allocation.append(TimeAllocationStats(
            model_name=model,
            opening_avg_time=op_time,
            middlegame_avg_time=mid_time,
            endgame_avg_time=end_time,
            opening_avg_quality=op_qual,
            middlegame_avg_quality=mid_qual,
            endgame_avg_quality=end_qual,
            complexity_time_correlation=comp_time_corr,
            complexity_quality_correlation=comp_qual_corr,
        ))
    
    # === 4. POSITION-DEPENDENT PERFORMANCE ===
    position_performance = []
    for model, moves in model_moves.items():
        complexity_moves = [m for m in moves if m["num_legal_moves"] is not None]
        sharpness_moves = [m for m in moves if m["eval_sharpness"] is not None]
        
        # By complexity
        simple = [m for m in complexity_moves if m["num_legal_moves"] < 20]
        complex_pos = [m for m in complexity_moves if m["num_legal_moves"] >= 30]
        
        simple_qual = sum(m["cp_loss"] for m in simple) / len(simple) if simple else 0
        complex_qual = sum(m["cp_loss"] for m in complex_pos) / len(complex_pos) if complex_pos else 0
        
        # By criticality (eval sharpness)
        if sharpness_moves:
            sharpness_median = median([m["eval_sharpness"] for m in sharpness_moves])
            routine = [m for m in sharpness_moves if m["eval_sharpness"] < sharpness_median]
            critical = [m for m in sharpness_moves if m["eval_sharpness"] >= sharpness_median]
        else:
            routine = []
            critical = []
        
        routine_qual = sum(m["cp_loss"] for m in routine) / len(routine) if routine else 0
        critical_qual = sum(m["cp_loss"] for m in critical) / len(critical) if critical else 0
        
        # Time on critical vs routine
        routine_time = sum(m["time_taken"] or 0 for m in routine) / len(routine) if routine else 0
        critical_time = sum(m["time_taken"] or 0 for m in critical) / len(critical) if critical else 0
        
        position_performance.append(PositionPerformance(
            model_name=model,
            simple_pos_quality=simple_qual,
            complex_pos_quality=complex_qual,
            complexity_penalty=complex_qual - simple_qual if simple else 0,
            routine_pos_quality=routine_qual,
            critical_pos_quality=critical_qual,
            criticality_penalty=critical_qual - routine_qual if routine else 0,
            critical_time_ratio=critical_time / routine_time if routine_time > 0 else 1.0,
        ))
    
    # === 5. OUTCOME CORRELATES ===
    outcome_correlates = []
    
    # Prepare outcome data
    valid_outcomes = [g for g in game_outcomes if g["white_avg_cp"] is not None]
    
    if len(valid_outcomes) >= 5:
        # Create per-side outcome data
        side_data = []
        for g in valid_outcomes:
            if g["white_avg_cp"] is not None:
                side_data.append({
                    "won": 1 if g["white_won"] else 0,
                    "avg_cp": g["white_avg_cp"],
                    "blunders": g["white_blunders"],
                    "avg_time": g["white_avg_time"],
                    "end_time": g["white_end_time"] or 0,
                })
            if g["black_avg_cp"] is not None:
                side_data.append({
                    "won": 0 if g["white_won"] else 1,
                    "avg_cp": g["black_avg_cp"],
                    "blunders": g["black_blunders"],
                    "avg_time": g["black_avg_time"],
                    "end_time": g["black_end_time"] or 0,
                })
        
        if len(side_data) >= 10:
            wins = [d["won"] for d in side_data]
            
            # Avg CP loss vs winning
            cp_corr = calculate_correlation([d["avg_cp"] for d in side_data], wins)
            outcome_correlates.append(OutcomeCorrelate(
                factor="Average CP Loss",
                correlation_with_win=cp_corr,
                sample_size=len(side_data),
                description="Lower CP loss → more wins (should be negative)",
            ))
            
            # Blunder count vs winning
            blunder_corr = calculate_correlation([d["blunders"] for d in side_data], wins)
            outcome_correlates.append(OutcomeCorrelate(
                factor="Blunder Count",
                correlation_with_win=blunder_corr,
                sample_size=len(side_data),
                description="More blunders → fewer wins (should be negative)",
            ))
            
            # Time remaining at end vs winning
            time_corr = calculate_correlation([d["end_time"] for d in side_data], wins)
            outcome_correlates.append(OutcomeCorrelate(
                factor="Time Remaining at End",
                correlation_with_win=time_corr,
                sample_size=len(side_data),
                description="More time left → more wins (if time management matters)",
            ))
            
            # Avg time per move vs winning
            avg_time_corr = calculate_correlation([d["avg_time"] for d in side_data], wins)
            outcome_correlates.append(OutcomeCorrelate(
                factor="Avg Time Per Move",
                correlation_with_win=avg_time_corr,
                sample_size=len(side_data),
                description="Spending more time → wins? (efficiency question)",
            ))
    
    outcome_correlates.sort(key=lambda x: abs(x.correlation_with_win), reverse=True)
    
    # === SCATTER DATA ===
    scatter_data = [
        {
            "model": m["model"],
            "move_number": m["move_number"],
            "time_taken": m["time_taken"],
            "time_remaining": m["time_remaining"],
            "thinking_tokens": m["thinking_tokens"],
            "cp_loss": m["cp_loss"],
            "num_legal_moves": m["num_legal_moves"],
            "eval_sharpness": m["eval_sharpness"],
        }
        for m in analyzed_moves[:1000]
    ]
    
    return InsightsResponse(
        total_matches=total_matches,
        total_games=total_games,
        total_moves=len(all_moves),
        total_analyzed_moves=len(analyzed_moves),
        outliers_capped=outlier_count,
        cp_loss_cap=CP_LOSS_CAP,
        same_model_games_excluded=same_model_game_count,
        tournament=tournament_summary,
        degradation_curves=degradation_curves,
        efficiency_metrics=efficiency_metrics,
        time_allocation=time_allocation,
        position_performance=position_performance,
        outcome_correlates=outcome_correlates,
        scatter_data=scatter_data,
    )
