#!/usr/bin/env python3
"""Model wrappers for blitz chess matches with accurate timing."""

import time
from typing import Tuple

import termcolor

from game_arena.harness import model_generation
from game_arena.harness import model_generation_sdk
from game_arena.harness import tournament_util

colored = termcolor.colored


class NoRetryModelWrapper:
    """Wrapper that disables automatic retries and handles them manually for accurate timing."""
    
    def __init__(self, wrapped_model, max_retries: int = 3, base_delay: float = 1.0):
        self.wrapped_model = wrapped_model
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # We'll call the underlying _generate method directly to bypass the retry decorator
        # This avoids the automatic retry logic entirely
        self._direct_generate = wrapped_model._generate
    
    @property
    def model_name(self) -> str:
        return self.wrapped_model.model_name
        
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should be retried."""
        return not isinstance(exception, model_generation.DoNotRetryError)
    
    def generate_with_text_input(self, model_input: tournament_util.ModelTextInput) -> Tuple[tournament_util.GenerateReturn, int, float]:
        """
        Generate with retry logic that doesn't count retry time.
        
        Returns:
            Tuple of (GenerateReturn, retry_count, total_retry_time)
        """
        retry_count = 0
        total_retry_time = 0.0
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Time only the successful call - call _generate directly to bypass retry decorator
                call_start = time.time()
                
                # Convert ModelTextInput to the format expected by _generate
                if hasattr(self.wrapped_model, '_generate'):
                    if isinstance(self.wrapped_model, model_generation_sdk.OpenAIChatCompletionsModel):
                        # OpenAI format
                        content = [{"type": "text", "text": model_input.prompt_text}]
                        result = self._direct_generate(content, model_input.system_instruction)
                    elif isinstance(self.wrapped_model, model_generation_sdk.AIStudioModel):
                        # AI Studio format (e.g., Gemini models)
                        contents = [model_input.prompt_text]
                        result = self._direct_generate(contents, model_input.system_instruction)
                    else:
                        # Fallback - try the original method but catch exceptions ourselves
                        result = self.wrapped_model.generate_with_text_input(model_input)
                else:
                    # Fallback - use the wrapped method
                    result = self.wrapped_model.generate_with_text_input(model_input)
                
                call_end = time.time()
                
                # Only count the time of the successful call
                actual_call_time = call_end - call_start
                
                return result, retry_count, total_retry_time
                
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e) or attempt >= self.max_retries:
                    # Don't retry or max attempts reached
                    break
                
                # Calculate retry delay
                retry_delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                retry_delay = min(retry_delay, 60.0)  # Cap at 60 seconds
                
                print(colored(f"API call failed (attempt {attempt + 1}), retrying in {retry_delay:.1f}s: {e}", "yellow"))
                
                retry_start = time.time()
                time.sleep(retry_delay)
                retry_end = time.time()
                
                retry_count += 1
                total_retry_time += (retry_end - retry_start)
        
        # If we get here, all retries failed
        raise last_exception


class BlitzModelWrapper:
    """Wrapper for models used with rethink samplers to track retry info."""
    
    def __init__(self, wrapped_model: NoRetryModelWrapper):
        self.wrapped_model = wrapped_model
        self.total_retry_time = 0
        self.retry_count = 0
    
    def generate_with_text_input(self, model_input):
        response, retry_count, retry_time = self.wrapped_model.generate_with_text_input(model_input)
        self.retry_count += retry_count
        self.total_retry_time += retry_time
        return response

