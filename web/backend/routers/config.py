#!/usr/bin/env python3
"""Configuration API endpoints - available models and match options."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


# Available models for matches
AVAILABLE_MODELS = [
    # Anthropic
    {"id": "claude-sonnet-4", "name": "Claude Sonnet 4", "provider": "Anthropic"},
    {"id": "claude-sonnet-4.5", "name": "Claude Sonnet 4.5", "provider": "Anthropic"},
    {"id": "claude-opus-4", "name": "Claude Opus 4", "provider": "Anthropic"},
    {"id": "claude-opus-4.5", "name": "Claude Opus 4.5", "provider": "Anthropic"},
    # Google
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "Google"},
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "Google"},
    {"id": "gemini-3-flash", "name": "Gemini 3 Flash", "provider": "Google"},
    {"id": "gemini-3-pro", "name": "Gemini 3 Pro", "provider": "Google"},
    # OpenAI
    {"id": "gpt-4.1", "name": "GPT-4.1", "provider": "OpenAI"},
    {"id": "gpt-5.2", "name": "GPT-5.2", "provider": "OpenAI"},
    {"id": "o3", "name": "o3", "provider": "OpenAI"},
    {"id": "o4-mini", "name": "o4-mini", "provider": "OpenAI"},
    # xAI
    {"id": "grok-4", "name": "Grok 4", "provider": "xAI"},
    {"id": "grok-4.1", "name": "Grok 4.1", "provider": "xAI"},
    # Others
    {"id": "deepseek-r1", "name": "DeepSeek R1", "provider": "DeepSeek"},
    {"id": "kimi-k2", "name": "Kimi K2", "provider": "Moonshot"},
    {"id": "qwen3", "name": "Qwen 3", "provider": "Alibaba"},
]

# Time control presets
TIME_CONTROL_PRESETS = [
    {"id": "bullet-1", "name": "Bullet 1+0", "initial_time": 60, "increment": 0},
    {"id": "bullet-2", "name": "Bullet 2+1", "initial_time": 120, "increment": 1},
    {"id": "blitz-3", "name": "Blitz 3+2", "initial_time": 180, "increment": 2},
    {"id": "blitz-5", "name": "Blitz 5+3", "initial_time": 300, "increment": 3},
    {"id": "rapid-10", "name": "Rapid 10+5", "initial_time": 600, "increment": 5},
    {"id": "rapid-15", "name": "Rapid 15+10", "initial_time": 900, "increment": 10},
]


class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str


class TimeControlPreset(BaseModel):
    id: str
    name: str
    initial_time: int
    increment: int


class ConfigResponse(BaseModel):
    models: list[ModelInfo]
    time_control_presets: list[TimeControlPreset]


@router.get("", response_model=ConfigResponse)
async def get_config():
    """Get available configuration options for starting a match."""
    return ConfigResponse(
        models=[ModelInfo(**m) for m in AVAILABLE_MODELS],
        time_control_presets=[TimeControlPreset(**t) for t in TIME_CONTROL_PRESETS],
    )


@router.get("/models", response_model=list[ModelInfo])
async def get_models():
    """Get list of available models."""
    return [ModelInfo(**m) for m in AVAILABLE_MODELS]

