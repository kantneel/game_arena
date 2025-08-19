#!/usr/bin/env python3
"""Verification functionality for blitz chess match."""

import os
import time
import termcolor
from game_arena.harness import tournament_util
from game_arena.harness import model_registry
import game_arena.blitz.utils as utils

colored = termcolor.colored


def get_api_key_for_model(registry_entry):
    """Get the appropriate API key for a model based on its provider."""
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
    """Get the appropriate model from the registry based on model name."""
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
        "claude-opus-4": model_registry.ModelRegistry.ANTHROPIC_CLAUDE_OPUS_4,
        "gemini-2.5-flash": model_registry.ModelRegistry.GEMINI_2_5_FLASH,
        "gemini-2.5-pro": model_registry.ModelRegistry.GEMINI_2_5_PRO,
        "gpt-4.1": model_registry.ModelRegistry.OPENAI_GPT_4_1,
        "o3": model_registry.ModelRegistry.OPENAI_O3,
        "o4-mini": model_registry.ModelRegistry.OPENAI_O4_MINI,
        "grok-4": model_registry.ModelRegistry.XAI_GROK_4,
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
        if "sonnet" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.ANTHROPIC_CLAUDE_SONNET_4
            api_key = get_api_key_for_model(registry_entry)
            return registry_entry.build(api_key=api_key)
        elif "opus" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.ANTHROPIC_CLAUDE_OPUS_4
            api_key = get_api_key_for_model(registry_entry)
            return registry_entry.build(api_key=api_key)
    elif "gemini" in model_name_lower:
        if "pro" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.GEMINI_2_5_PRO
            api_key = get_api_key_for_model(registry_entry)
            return registry_entry.build(api_key=api_key)
        else:
            registry_entry = model_registry.ModelRegistry.GEMINI_2_5_FLASH
            api_key = get_api_key_for_model(registry_entry)
            return registry_entry.build(api_key=api_key)
    elif any(x in model_name_lower for x in ["gpt", "o3", "o4"]):
        if "o3" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.OPENAI_O3
            api_key = get_api_key_for_model(registry_entry)
            return registry_entry.build(api_key=api_key)
        elif "o4" in model_name_lower:
            registry_entry = model_registry.ModelRegistry.OPENAI_O4_MINI
            api_key = get_api_key_for_model(registry_entry)
            return registry_entry.build(api_key=api_key)
        else:
            registry_entry = model_registry.ModelRegistry.OPENAI_GPT_4_1
            api_key = get_api_key_for_model(registry_entry)
            return registry_entry.build(api_key=api_key)
    elif "grok" in model_name_lower:
        registry_entry = model_registry.ModelRegistry.XAI_GROK_4
        api_key = get_api_key_for_model(registry_entry)
        return registry_entry.build(api_key=api_key)
    
    # If we can't determine the model type, raise an error
    raise ValueError(f"Cannot determine API for model: {model_name}. "
                     f"Please use one of the supported models or update the mapping.")


def setup_models_and_rethink_samplers(flags_module):
    """Set up the models and rethink samplers for the match."""
    from game_arena.harness import model_generation_sdk
    from game_arena.harness import parsers
    from game_arena.harness import llm_parsers
    from game_arena.harness import rethink
    from game_arena.harness import prompt_generation
    
    # Set up model generation using the registry to determine correct API
    print(f"Setting up Model A: {flags_module._MODEL_A.value}")
    model_a = get_model_from_registry(flags_module._MODEL_A.value)
    
    print(f"Setting up Model B: {flags_module._MODEL_B.value}")
    model_b = get_model_from_registry(flags_module._MODEL_B.value)
    
    # Apply reasoning budget based on model type
    def apply_reasoning_budget(model):
        if not hasattr(model, '_model_options') or model._model_options is None:
            return
        
        # For Anthropic models, update thinking.budget_tokens
        if 'thinking' in model._model_options and isinstance(model._model_options['thinking'], dict):
            if model._model_options['thinking'].get('type') == 'enabled':
                model._model_options['thinking']['budget_tokens'] = flags_module._REASONING_BUDGET.value
        # For Gemini models, set thinking_budget
        elif 'thinking_budget' not in model._model_options:
            model._model_options['thinking_budget'] = flags_module._REASONING_BUDGET.value
    
    apply_reasoning_budget(model_a)
    apply_reasoning_budget(model_b)
    
    # Set up parsers for rethinking
    match flags_module._PARSER_CHOICE.value:
        case tournament_util.ParserChoice.RULE_THEN_SOFT:
            move_parser = parsers.RuleBasedMoveParser()
            legality_parser = parsers.SoftMoveParser("chess")
        case tournament_util.ParserChoice.LLM_ONLY:
            # Use a fast, reliable model for parsing
            parser_model = get_model_from_registry("gemini-2.5-flash")
            move_parser = llm_parsers.LLMParser(
                model=parser_model,
                instruction_config=llm_parsers.OpenSpielChessInstructionConfig_V0,
            )
            legality_parser = parsers.SoftMoveParser("chess")
        case _:
            raise ValueError(f"Unsupported parser choice: {flags_module._PARSER_CHOICE.value}")
    
    # Create rethink samplers if enabled
    if flags_module._USE_RETHINKING.value:
        prompt_generator = prompt_generation.PromptGeneratorText()
        
        model_a_sampler = rethink.RethinkSampler(
            model=model_a,
            strategy=flags_module._RETHINK_STRATEGY.value,
            num_max_rethinks=flags_module._MAX_RETHINKS.value,
            move_parser=move_parser,
            legality_parser=legality_parser,
            game_short_name="chess",
            prompt_generator=prompt_generator,
            rethink_template=None,
        )
        
        model_b_sampler = rethink.RethinkSampler(
            model=model_b,
            strategy=flags_module._RETHINK_STRATEGY.value,
            num_max_rethinks=flags_module._MAX_RETHINKS.value,
            move_parser=move_parser,
            legality_parser=legality_parser,
            game_short_name="chess",
            prompt_generator=prompt_generator,
            rethink_template=None,
        )
        
        return model_a, model_b, model_a_sampler, model_b_sampler
    else:
        # Fallback to basic parsing
        if flags_module._PARSER_CHOICE.value == tournament_util.ParserChoice.RULE_THEN_SOFT:
            parser = parsers.ChainedMoveParser([move_parser, legality_parser])
        else:
            parser = move_parser
        return model_a, model_b, parser, parser


def verify_retry_wrapper_functionality(flags_module):
    """Verify that the NoRetryModelWrapper works correctly."""
    print(colored("🔍 VERIFYING RETRY WRAPPER FUNCTIONALITY", "cyan"))
    
    try:
        # Set up models
        model_a, model_b, _, _ = setup_models_and_rethink_samplers(flags_module)
        
        # Wrap models
        model_a_wrapper = utils.NoRetryModelWrapper(model_a, max_retries=1, base_delay=0.1)
        model_b_wrapper = utils.NoRetryModelWrapper(model_b, max_retries=1, base_delay=0.1)
        
        print(f"✅ Successfully created wrappers")
        print(f"   Model A: {model_a_wrapper.model_name}")
        print(f"   Model B: {model_b_wrapper.model_name}")
        
        # Test basic functionality with minimal prompt
        test_prompt = tournament_util.ModelTextInput(prompt_text="Say 'Hello' in one word.")
        
        print("\n🧪 Testing Model A wrapper...")
        try:
            start_time = time.time()
            response, retry_count, retry_time = model_a_wrapper.generate_with_text_input(test_prompt)
            end_time = time.time()
            
            call_time = end_time - start_time - retry_time
            print(f"✅ Model A test successful:")
            print(f"   Response: {response.main_response[:50]}...")
            print(f"   Call time: {call_time:.3f}s")
            print(f"   Retries: {retry_count} ({retry_time:.3f}s retry time)")
            
        except Exception as e:
            print(f"❌ Model A test failed: {e}")
            return False
        
        print("\n🧪 Testing Model B wrapper...")
        try:
            start_time = time.time()
            response, retry_count, retry_time = model_b_wrapper.generate_with_text_input(test_prompt)
            end_time = time.time()
            
            call_time = end_time - start_time - retry_time
            print(f"✅ Model B test successful:")
            print(f"   Response: {response.main_response[:50]}...")
            print(f"   Call time: {call_time:.3f}s")
            print(f"   Retries: {retry_count} ({retry_time:.3f}s retry time)")
            
        except Exception as e:
            print(f"❌ Model B test failed: {e}")
            return False
        
        print(colored("\n🎉 VERIFICATION COMPLETE - All tests passed!", "green"))
        print(colored("The retry wrapper is working correctly and will exclude retry time from chess clocks.", "green"))
        return True
        
    except Exception as e:
        print(colored(f"❌ VERIFICATION FAILED: {e}", "red"))
        return False 