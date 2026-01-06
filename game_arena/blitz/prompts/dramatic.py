#!/usr/bin/env python3
"""Dramatic time pressure prompt generation for blitz chess matches."""

from game_arena.blitz.display.formatting import format_time


def create_dramatic_time_pressure_text(
    player_time_remaining: float,
    opponent_time_remaining: float,
    increment_seconds: int
) -> str:
    """Generate dramatic time pressure text with ALL CAPS emphasis.
    
    Args:
        player_time_remaining: Current player's remaining time in seconds
        opponent_time_remaining: Opponent's remaining time in seconds
        increment_seconds: Time increment per move
        
    Returns:
        Dramatic time pressure prompt text
    """
    time_remaining = player_time_remaining
    
    if time_remaining < 30:
        urgency_level = "EXTREME"
        pressure_color = "🔴🔴🔴"
        dramatic_text = """
🚨🚨🚨 CRITICAL TIME EMERGENCY!!! 🚨🚨🚨

⚠️⚠️⚠️ YOU HAVE LESS THAN 30 SECONDS LEFT!!! ⚠️⚠️⚠️

🔥🔥🔥 EVERY SECOND COUNTS - MOVE IMMEDIATELY OR LOSE!!! 🔥🔥🔥

💥 DO NOT OVERTHINK! ANY REASONABLE MOVE IS BETTER THAN TIMING OUT! 💥
💥 EVEN A RANDOM LEGAL MOVE BEATS RUNNING OUT OF TIME! 💥
💥 SPEED IS MORE IMPORTANT THAN PERFECTION RIGHT NOW! 💥

⏰ THE CLOCK IS YOUR BIGGEST ENEMY - NOT YOUR OPPONENT! ⏰"""
    elif time_remaining < 60:
        urgency_level = "HIGH"
        pressure_color = "🔴🔴"
        dramatic_text = """
🚨🚨 TIME PRESSURE ALERT!!! 🚨🚨

⚠️ LESS THAN 1 MINUTE REMAINING! ⚠️

🔥 THINK FAST - EVERY SECOND MATTERS! 🔥
🔥 QUICK GOOD MOVES BEAT SLOW PERFECT MOVES! 🔥
🔥 TIME FORFEIT = INSTANT LOSS! 🔥

⏰ PRIORITIZE SPEED OVER DEEP ANALYSIS! ⏰"""
    elif time_remaining < 120:
        urgency_level = "MEDIUM"
        pressure_color = "🟡🟡"
        dramatic_text = """
⚠️ TIME PRESSURE BUILDING! ⚠️

🔥 UNDER 2 MINUTES - START MOVING FASTER! 🔥
⏰ BALANCE QUALITY WITH SPEED! ⏰
💭 LIMIT YOUR THINKING TIME PER MOVE! 💭"""
    else:
        urgency_level = "LOW"
        pressure_color = "🟢"
        dramatic_text = """
✅ COMFORTABLE TIME CUSHION ✅
💭 You can afford some analysis, but don't waste time! 💭"""

    return f"""
{dramatic_text}

{pressure_color} TIME PRESSURE LEVEL: {urgency_level} {pressure_color}
⏰ YOUR TIME: {format_time(time_remaining)}
⏰ OPPONENT TIME: {format_time(opponent_time_remaining)}
⏰ INCREMENT: +{increment_seconds}s per move

🎯 REMEMBER: Running out of time = AUTOMATIC LOSS!!!
🎯 A mediocre move in 5 seconds beats a brilliant move in 65 seconds when you only have 60 seconds left!
"""


def create_dramatic_instruction_text(time_remaining: float) -> str:
    """Generate dramatic instruction text based on time remaining.
    
    Args:
        time_remaining: Current player's remaining time in seconds
        
    Returns:
        Instruction text for the model
    """
    if time_remaining < 30:
        return "MOVE NOW!!! Minimal or no reasoning - just output your move immediately! Time is critical!!!"
    elif time_remaining < 60:
        return "Very brief reasoning only! Decide within a few seconds!"
    elif time_remaining < 120:
        return "Be efficient - reason only as much as needed for this position, then move!"
    else:
        return "Reason as much as you think is necessary for this position (could be extensive or brief), then output your final answer in the format \"Final Answer: X\" where X is your chosen move in algebraic notation."

