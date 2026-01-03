#!/usr/bin/env python3
"""Model registry utilities for blitz chess matches."""

import os

from game_arena.harness import model_registry


def get_api_key_for_model(registry_entry) -> str:
    """Get the appropriate API key for a model based on its provider.
    
    Args:
        registry_entry: ModelRegistry enum entry
        
    Returns:
        API key string from environment variables
    """
    if 'ANTHROPIC' in registry_entry.name:
        return os.getenv('ANTHROPIC_API_KEY', '')
    elif 'OPENAI' in registry_entry.name:
        return os.getenv('OPENAI_API_KEY', '')
    elif 'GEMINI' in registry_entry.name:
        return os.getenv('GOOGLE_API_KEY', '')
    elif 'XAI' in registry_entry.name:
        return os.getenv('XAI_API_KEY', '')
    elif 'DEEPSEEK' in registry_entry.name or 'KIMI' in registry_entry.name or 'QWEN' in registry_entry.name:
        return os.getenv('TOGETHER_API_KEY', '')
    else:
        return ''


def get_model_from_registry(model_name: str):
    """Get the appropriate model from the registry based on model name.
    
    Args:
        model_name: String name of the model (e.g., "claude-sonnet-4", "gemini-2.5-flash")
        
    Returns:
        Instantiated model object
        
    Raises:
        ValueError: If model name cannot be mapped to a registry entry
    """
    # Try to find exact match first
    for registry_entry in model_registry.ModelRegistry:
        if registry_entry.value == model_name:
            api_key = get_api_key_for_model(registry_entry)
            return registry_entry.build(api_key=api_key)
    
    # Try to find partial matches for common model names
    model_name_lower = model_name.lower()
    
    # Map common model names to registry entries
    name_mappings = {
        "claude-sonnet-4": model_registry.ModelRegistry.ANTHROPIC_CLAUDE_SONNET_4,
        "claude-sonnet-4.5": model_registry.ModelRegistry.ANTHROPIC_CLAUDE_SONNET_4_5,
        "claude-opus-4": model_registry.ModelRegistry.ANTHROPIC_CLAUDE_OPUS_4,
        "claude-opus-4.5": model_registry.ModelRegistry.ANTHROPIC_CLAUDE_OPUS_4_5,
        "gemini-2.5-flash": model_registry.ModelRegistry.GEMINI_2_5_FLASH,
        "gemini-2.5-pro": model_registry.ModelRegistry.GEMINI_2_5_PRO,
        "gemini-3-flash": model_registry.ModelRegistry.GEMINI_3_FLASH,
        "gemini-3-pro": model_registry.ModelRegistry.GEMINI_3_PRO,
        "gpt-4.1": model_registry.ModelRegistry.OPENAI_GPT_4_1,
        "gpt-5.2": model_registry.ModelRegistry.OPENAI_GPT_5_2,
        "o3": model_registry.ModelRegistry.OPENAI_O3,
        "o4-mini": model_registry.ModelRegistry.OPENAI_O4_MINI,
        "grok-4": model_registry.ModelRegistry.XAI_GROK_4,
        "grok-4.1": model_registry.ModelRegistry.XAI_GROK_4_1,
        "deepseek-r1": model_registry.ModelRegistry.DEEPSEEK_R1,
        "kimi-k2": model_registry.ModelRegistry.KIMI_K2,
        "qwen3": model_registry.ModelRegistry.QWEN_3,
    }
    
    if model_name_lower in name_mappings:
        registry_entry = name_mappings[model_name_lower]
        api_key = get_api_key_for_model(registry_entry)
        return registry_entry.build(api_key=api_key)
    
    # Fallback: try to infer from model name patterns
    if "claude" in model_name_lower:
        if "4.5" in model_name_lower or "4-5" in model_name_lower:
            if "sonnet" in model_name_lower:
                registry_entry = model_registry.ModelRegistry.ANTHROPIC_CLAUDE_SONNET_4_5
            else:
                registry_entry = model_registry.ModelRegistry.ANTHROPIC_CLAUDE_OPUS_4_5
        elif "sonnet" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.ANTHROPIC_CLAUDE_SONNET_4
        elif "opus" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.ANTHROPIC_CLAUDE_OPUS_4
        else:
            registry_entry = model_registry.ModelRegistry.ANTHROPIC_CLAUDE_SONNET_4_5
        api_key = get_api_key_for_model(registry_entry)
        return registry_entry.build(api_key=api_key)
    elif "gemini" in model_name_lower:
        if "3" in model_name_lower:
            if "pro" in model_name_lower:
                registry_entry = model_registry.ModelRegistry.GEMINI_3_PRO
            else:
                registry_entry = model_registry.ModelRegistry.GEMINI_3_FLASH
        elif "pro" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.GEMINI_2_5_PRO
        else:
            registry_entry = model_registry.ModelRegistry.GEMINI_2_5_FLASH
        api_key = get_api_key_for_model(registry_entry)
        return registry_entry.build(api_key=api_key)
    elif any(x in model_name_lower for x in ["gpt", "o3", "o4"]):
        if "o3" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.OPENAI_O3
        elif "o4" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.OPENAI_O4_MINI
        elif "5.2" in model_name_lower or "5-2" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.OPENAI_GPT_5_2
        else:
            registry_entry = model_registry.ModelRegistry.OPENAI_GPT_4_1
        api_key = get_api_key_for_model(registry_entry)
        return registry_entry.build(api_key=api_key)
    elif "grok" in model_name_lower:
        if "4.1" in model_name_lower or "4-1" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.XAI_GROK_4_1
        else:
            registry_entry = model_registry.ModelRegistry.XAI_GROK_4
        api_key = get_api_key_for_model(registry_entry)
        return registry_entry.build(api_key=api_key)
    
    # If we can't determine the model type, raise an error
    raise ValueError(f"Cannot determine API for model: {model_name}. "
                     f"Please use one of the supported models or update the mapping.")

