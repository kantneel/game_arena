#!/usr/bin/env python3
"""Verification functionality for blitz chess match.

This module provides model setup and verification utilities.
"""

import os
import time
import termcolor

from game_arena.harness import tournament_util
from game_arena.harness import parsers
from game_arena.harness import prompt_generation
from game_arena.harness import rethink

# Import from new modular structure
from game_arena.blitz.models.registry import get_model_from_registry, get_api_key_for_model
from game_arena.blitz.models.wrappers import NoRetryModelWrapper

colored = termcolor.colored


def setup_models_and_rethink_samplers(flags_module):
    """Set up the models and rethink samplers for the match."""
    from game_arena.harness import llm_parsers
    
    print(f"Setting up Model A: {flags_module._MODEL_A.value}")
    model_a = get_model_from_registry(flags_module._MODEL_A.value)
    
    print(f"Setting up Model B: {flags_module._MODEL_B.value}")
    model_b = get_model_from_registry(flags_module._MODEL_B.value)
    
    # Apply reasoning budget based on model type
    def apply_reasoning_budget(model):
        if not hasattr(model, '_model_options') or model._model_options is None:
            return
        
        if 'thinking' in model._model_options and isinstance(model._model_options['thinking'], dict):
            if model._model_options['thinking'].get('type') == 'enabled':
                model._model_options['thinking']['budget_tokens'] = flags_module._REASONING_BUDGET.value
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
        if flags_module._PARSER_CHOICE.value == tournament_util.ParserChoice.RULE_THEN_SOFT:
            parser = parsers.ChainedMoveParser([move_parser, legality_parser])
        else:
            parser = move_parser
        return model_a, model_b, parser, parser


def verify_retry_wrapper_functionality(flags_module):
    """Verify that the NoRetryModelWrapper works correctly."""
    print(colored("🔍 VERIFYING RETRY WRAPPER FUNCTIONALITY", "cyan"))
    
    try:
        model_a, model_b, _, _ = setup_models_and_rethink_samplers(flags_module)
        
        model_a_wrapper = NoRetryModelWrapper(model_a, max_retries=1, base_delay=0.1)
        model_b_wrapper = NoRetryModelWrapper(model_b, max_retries=1, base_delay=0.1)
        
        print(f"✅ Successfully created wrappers")
        print(f"   Model A: {model_a_wrapper.model_name}")
        print(f"   Model B: {model_b_wrapper.model_name}")
        
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
        return True
        
    except Exception as e:
        print(colored(f"❌ VERIFICATION FAILED: {e}", "red"))
        return False
