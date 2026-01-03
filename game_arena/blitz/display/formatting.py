#!/usr/bin/env python3
"""Formatting utilities for blitz chess matches."""


def format_time(seconds: float) -> str:
    """Format time as MM:SS.s"""
    if seconds < 0:
        return "00:00.0"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"


def abbreviate_model_name(model_name: str) -> str:
    """Create a short abbreviation for model names to keep folder names reasonable."""
    abbreviations = {
        # Anthropic
        "claude-sonnet-4": "cs4",
        "claude-sonnet-4.5": "cs45",
        "claude-opus-4": "co4",
        "claude-opus-4.5": "co45",
        # Google
        "gemini-2.5-flash": "g25f",
        "gemini-2.5-pro": "g25p",
        "gemini-3-flash": "g3f",
        "gemini-3-pro": "g3p",
        # OpenAI
        "gpt-4.1": "gpt41",
        "gpt-5.2": "gpt52",
        "o3": "o3",
        "o4-mini": "o4m",
        # xAI
        "grok-4": "grok4",
        "grok-4.1": "grok41",
        # Other
        "deepseek-r1": "dsr1",
        "kimi-k2": "kimik2",
        "qwen3": "qw3",
    }
    
    # Try exact match first
    if model_name.lower() in abbreviations:
        return abbreviations[model_name.lower()]
    
    # Try partial matches for more complex model names
    lower_name = model_name.lower()
    for full_name, abbrev in abbreviations.items():
        if full_name in lower_name:
            return abbrev
    
    # Fallback: take first letters of words, max 6 chars
    words = model_name.replace("-", " ").replace("_", " ").split()
    if len(words) > 1:
        abbrev = "".join(word[0] for word in words if word)[:6]
        return abbrev.lower()
    else:
        return model_name[:6].lower()

