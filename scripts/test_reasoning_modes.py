#!/usr/bin/env python3
"""Test script to verify reasoning mode and budget options work across model families.

This script tests whether reasoning/thinking mode produces different outputs
and properly reports reasoning tokens for each supported model family.

Usage:
    python scripts/test_reasoning_modes.py
    
    # Test specific families only:
    python scripts/test_reasoning_modes.py --families anthropic gemini
    
    # Custom reasoning budgets:
    python scripts/test_reasoning_modes.py --budgets 1000 8000 32000
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

# Add the game_arena package to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_arena.harness import model_generation_sdk, model_generation_http, tournament_util


# =============================================================================
# Configuration
# =============================================================================

# Test prompt - complex enough to benefit from reasoning
TEST_PROMPT = """You are playing chess as White. The current position after 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 is:

Position: Sicilian Najdorf, one of the most complex openings in chess.

What is your next move? Explain your reasoning briefly, then provide your move in standard algebraic notation.

Format: First explain your thinking, then on a new line write "MOVE: " followed by your move (e.g., "MOVE: Be3")"""


@dataclass
class TestResult:
    """Result of a single test."""
    model_family: str
    model_name: str
    reasoning_enabled: bool
    reasoning_budget: Optional[int]
    success: bool
    response_text: str = ""
    reasoning_tokens: Optional[int] = None
    generation_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    has_thinking_content: bool = False
    thinking_content_preview: str = ""
    wall_time_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class FamilyReport:
    """Report for a model family."""
    family: str
    tests: list[TestResult] = field(default_factory=list)
    reasoning_works: bool = False
    budget_affects_output: bool = False
    notes: list[str] = field(default_factory=list)


# =============================================================================
# Model Builders
# =============================================================================

def get_anthropic_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return key


def get_google_api_key() -> str:
    # Try GEMINI_API_KEY first (preferred), then GOOGLE_API_KEY (fallback)
    key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
    return key


def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return key


def get_xai_api_key() -> str:
    key = os.getenv("XAI_API_KEY", "")
    if not key:
        raise ValueError("XAI_API_KEY environment variable not set")
    return key


def build_anthropic_model(reasoning_enabled: bool, budget: int = 16000):
    """Build an Anthropic model with or without reasoning."""
    api_key = get_anthropic_api_key()
    
    if reasoning_enabled:
        model_options = {
            "max_tokens": 64000,
            "thinking": {"type": "enabled", "budget_tokens": budget}
        }
    else:
        model_options = {
            "max_tokens": 4096,
            # No thinking config = reasoning disabled
        }
    
    return model_generation_sdk.AnthropicModel(
        model_name="claude-sonnet-4-20250514",
        api_key=api_key,
        api_options={"stream": True},
        model_options=model_options,
    )


def build_gemini_model(include_thoughts: bool, thinking_budget: int = 8000):
    """Build a Gemini model with or without thought inclusion."""
    api_key = get_google_api_key()
    
    return model_generation_sdk.AIStudioModel(
        model_name="gemini-2.5-flash",
        api_key=api_key,
        api_options={"include_thoughts": include_thoughts},
        model_options={"thinking_budget": thinking_budget} if include_thoughts else {},
    )


def build_openai_model():
    """Build an OpenAI model (reasoning mode controlled differently)."""
    api_key = get_openai_api_key()
    
    # OpenAI o-series models have built-in reasoning
    return model_generation_sdk.OpenAIChatCompletionsModel(
        model_name="gpt-4.1-2025-04-14",  # Using GPT-4.1 for testing
        api_key=api_key,
    )


def build_xai_model():
    """Build an xAI Grok model."""
    api_key = get_xai_api_key()
    
    return model_generation_http.XAIModel(
        model_name="grok-4-0709",
        api_key=api_key,
        api_options={"stream": True},
    )


# =============================================================================
# Test Functions
# =============================================================================

def run_single_test(
    model,
    model_family: str,
    model_name: str,
    reasoning_enabled: bool,
    reasoning_budget: Optional[int] = None,
) -> TestResult:
    """Run a single test with the given model configuration."""
    result = TestResult(
        model_family=model_family,
        model_name=model_name,
        reasoning_enabled=reasoning_enabled,
        reasoning_budget=reasoning_budget,
        success=False,
    )
    
    start_time = time.time()
    
    try:
        # Create model input using the proper tournament_util class
        model_input = tournament_util.ModelTextInput(
            prompt_text=TEST_PROMPT,
            system_instruction="You are a world-class chess grandmaster. Analyze positions carefully and provide your best move.",
        )
        
        # Generate response using the correct method
        response = model.generate_with_text_input(model_input)
        
        result.wall_time_seconds = time.time() - start_time
        result.success = True
        result.response_text = response.main_response[:500]  # Truncate for report
        result.reasoning_tokens = response.reasoning_tokens
        result.generation_tokens = response.generation_tokens
        result.prompt_tokens = response.prompt_tokens
        
        # Check for thinking content
        thoughts = response.main_response_and_thoughts
        if thoughts and thoughts != response.main_response:
            result.has_thinking_content = True
            # Extract thinking preview (first 200 chars of thinking part)
            thinking_part = thoughts.replace(response.main_response, "").strip()
            result.thinking_content_preview = thinking_part[:200]
        
    except Exception as e:
        result.wall_time_seconds = time.time() - start_time
        result.error = str(e)
    
    return result


def test_anthropic_family(budgets: list[int]) -> FamilyReport:
    """Test Anthropic Claude models."""
    report = FamilyReport(family="Anthropic (Claude)")
    
    print("\n" + "="*60)
    print("🧪 Testing Anthropic Claude")
    print("="*60)
    
    try:
        # Test without reasoning
        print("\n📍 Testing WITHOUT reasoning enabled...")
        model_no_reasoning = build_anthropic_model(reasoning_enabled=False)
        result = run_single_test(
            model_no_reasoning,
            "anthropic",
            "claude-sonnet-4",
            reasoning_enabled=False,
        )
        report.tests.append(result)
        print_test_result(result)
        
        # Test with reasoning at different budgets
        for budget in budgets:
            print(f"\n📍 Testing WITH reasoning (budget={budget})...")
            model_with_reasoning = build_anthropic_model(reasoning_enabled=True, budget=budget)
            result = run_single_test(
                model_with_reasoning,
                "anthropic",
                "claude-sonnet-4",
                reasoning_enabled=True,
                reasoning_budget=budget,
            )
            report.tests.append(result)
            print_test_result(result)
        
        # Analyze results
        analyze_family_results(report)
        
    except ValueError as e:
        report.notes.append(f"⚠️ Skipped: {str(e)}")
        print(f"⚠️ Skipping Anthropic: {e}")
    except Exception as e:
        report.notes.append(f"❌ Error: {str(e)}")
        print(f"❌ Error testing Anthropic: {e}")
    
    return report


def test_gemini_family(budgets: list[int]) -> FamilyReport:
    """Test Google Gemini models."""
    report = FamilyReport(family="Google (Gemini)")
    
    print("\n" + "="*60)
    print("🧪 Testing Google Gemini")
    print("="*60)
    
    try:
        # Test without thoughts
        print("\n📍 Testing WITHOUT thought inclusion...")
        model_no_thoughts = build_gemini_model(include_thoughts=False)
        result = run_single_test(
            model_no_thoughts,
            "gemini",
            "gemini-2.5-flash",
            reasoning_enabled=False,
        )
        report.tests.append(result)
        print_test_result(result)
        
        # Test with thoughts at different budgets
        for budget in budgets:
            print(f"\n📍 Testing WITH thoughts (budget={budget})...")
            model_with_thoughts = build_gemini_model(include_thoughts=True, thinking_budget=budget)
            result = run_single_test(
                model_with_thoughts,
                "gemini",
                "gemini-2.5-flash",
                reasoning_enabled=True,
                reasoning_budget=budget,
            )
            report.tests.append(result)
            print_test_result(result)
        
        # Analyze results
        analyze_family_results(report)
        
    except ValueError as e:
        report.notes.append(f"⚠️ Skipped: {str(e)}")
        print(f"⚠️ Skipping Gemini: {e}")
    except Exception as e:
        report.notes.append(f"❌ Error: {str(e)}")
        print(f"❌ Error testing Gemini: {e}")
    
    return report


def test_openai_family() -> FamilyReport:
    """Test OpenAI models."""
    report = FamilyReport(family="OpenAI")
    
    print("\n" + "="*60)
    print("🧪 Testing OpenAI")
    print("="*60)
    
    try:
        print("\n📍 Testing GPT-4.1...")
        model = build_openai_model()
        result = run_single_test(
            model,
            "openai",
            "gpt-4.1",
            reasoning_enabled=True,  # OpenAI models have implicit reasoning
        )
        report.tests.append(result)
        print_test_result(result)
        
        # OpenAI reasoning is model-specific (o-series), not toggle-able
        report.notes.append("OpenAI reasoning is model-dependent (o3/o4-mini have reasoning built-in)")
        
        # Analyze results
        analyze_family_results(report)
        
    except ValueError as e:
        report.notes.append(f"⚠️ Skipped: {str(e)}")
        print(f"⚠️ Skipping OpenAI: {e}")
    except Exception as e:
        report.notes.append(f"❌ Error: {str(e)}")
        print(f"❌ Error testing OpenAI: {e}")
    
    return report


def test_xai_family() -> FamilyReport:
    """Test xAI Grok models."""
    report = FamilyReport(family="xAI (Grok)")
    
    print("\n" + "="*60)
    print("🧪 Testing xAI Grok")
    print("="*60)
    
    try:
        print("\n📍 Testing Grok-4...")
        model = build_xai_model()
        result = run_single_test(
            model,
            "xai",
            "grok-4",
            reasoning_enabled=True,
        )
        report.tests.append(result)
        print_test_result(result)
        
        report.notes.append("Grok reasoning mode controlled via model variant (grok-4-1-fast-reasoning)")
        
        # Analyze results
        analyze_family_results(report)
        
    except ValueError as e:
        report.notes.append(f"⚠️ Skipped: {str(e)}")
        print(f"⚠️ Skipping xAI: {e}")
    except Exception as e:
        report.notes.append(f"❌ Error: {str(e)}")
        print(f"❌ Error testing xAI: {e}")
    
    return report


# =============================================================================
# Analysis & Reporting
# =============================================================================

def print_test_result(result: TestResult):
    """Print a single test result."""
    status = "✅" if result.success else "❌"
    reasoning_status = "ON" if result.reasoning_enabled else "OFF"
    
    print(f"  {status} Reasoning: {reasoning_status}", end="")
    if result.reasoning_budget:
        print(f" (budget: {result.reasoning_budget})", end="")
    print(f" | Time: {result.wall_time_seconds:.2f}s")
    
    if result.success:
        print(f"     • Reasoning tokens: {result.reasoning_tokens or 'N/A'}")
        print(f"     • Generation tokens: {result.generation_tokens or 'N/A'}")
        print(f"     • Has thinking content: {result.has_thinking_content}")
        if result.thinking_content_preview:
            preview = result.thinking_content_preview[:100].replace('\n', ' ')
            print(f"     • Thinking preview: \"{preview}...\"")
    else:
        print(f"     ❌ Error: {result.error}")


def analyze_family_results(report: FamilyReport):
    """Analyze test results for a model family."""
    successful_tests = [t for t in report.tests if t.success]
    
    if not successful_tests:
        report.reasoning_works = False
        report.notes.append("No successful tests")
        return
    
    # Check if reasoning produces tokens
    reasoning_tests = [t for t in successful_tests if t.reasoning_enabled]
    non_reasoning_tests = [t for t in successful_tests if not t.reasoning_enabled]
    
    if reasoning_tests:
        has_reasoning_tokens = any(t.reasoning_tokens and t.reasoning_tokens > 0 for t in reasoning_tests)
        has_thinking_content = any(t.has_thinking_content for t in reasoning_tests)
        
        report.reasoning_works = has_reasoning_tokens or has_thinking_content
        
        if has_reasoning_tokens:
            report.notes.append("✅ Reasoning tokens reported correctly")
        if has_thinking_content:
            report.notes.append("✅ Thinking content captured")
    
    # Check if budget affects output
    if len(reasoning_tests) >= 2:
        tokens_by_budget = [(t.reasoning_budget, t.reasoning_tokens) for t in reasoning_tests if t.reasoning_tokens]
        if len(tokens_by_budget) >= 2:
            tokens_by_budget.sort(key=lambda x: x[0] or 0)
            # Check if higher budget -> more tokens (allowing for variance)
            if tokens_by_budget[-1][1] > tokens_by_budget[0][1] * 1.1:
                report.budget_affects_output = True
                report.notes.append("✅ Higher budget produces more reasoning tokens")


def generate_final_report(reports: list[FamilyReport]):
    """Generate the final summary report."""
    print("\n")
    print("=" * 70)
    print("📊 FINAL REPORT: Reasoning Mode Test Results")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Summary table
    print("┌" + "─" * 68 + "┐")
    print(f"│ {'Model Family':<25} │ {'Reasoning Works':<15} │ {'Budget Effect':<15} │")
    print("├" + "─" * 68 + "┤")
    
    for report in reports:
        reasoning_status = "✅ Yes" if report.reasoning_works else "❌ No"
        budget_status = "✅ Yes" if report.budget_affects_output else "—"
        
        # Check if skipped
        if any("Skipped" in n for n in report.notes):
            reasoning_status = "⚠️ Skipped"
            budget_status = "—"
        
        print(f"│ {report.family:<25} │ {reasoning_status:<15} │ {budget_status:<15} │")
    
    print("└" + "─" * 68 + "┘")
    
    # Detailed notes
    print("\n📝 Detailed Notes:")
    for report in reports:
        print(f"\n  {report.family}:")
        if report.notes:
            for note in report.notes:
                print(f"    • {note}")
        else:
            print("    • No additional notes")
    
    # Test details
    print("\n📋 Test Details:")
    for report in reports:
        print(f"\n  {report.family}:")
        for test in report.tests:
            status = "✅" if test.success else "❌"
            mode = "reasoning ON" if test.reasoning_enabled else "reasoning OFF"
            budget_str = f" (budget={test.reasoning_budget})" if test.reasoning_budget else ""
            
            print(f"    {status} {mode}{budget_str}")
            if test.success:
                print(f"       Reasoning tokens: {test.reasoning_tokens or 'N/A'}")
                print(f"       Time: {test.wall_time_seconds:.2f}s")
            else:
                print(f"       Error: {test.error}")
    
    # Overall assessment
    print("\n" + "=" * 70)
    all_working = sum(1 for r in reports if r.reasoning_works)
    total = sum(1 for r in reports if not any("Skipped" in n for n in r.notes))
    
    if total > 0:
        print(f"✨ Overall: {all_working}/{total} tested model families have working reasoning mode")
    else:
        print("⚠️ No model families were successfully tested")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test reasoning mode and budget options across model families"
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["anthropic", "gemini", "openai", "xai", "all"],
        default=["all"],
        help="Model families to test (default: all)",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[4000, 16000],
        help="Reasoning budgets to test (default: 4000 16000)",
    )
    
    args = parser.parse_args()
    
    families_to_test = args.families
    if "all" in families_to_test:
        families_to_test = ["anthropic", "gemini", "openai", "xai"]
    
    budgets = args.budgets
    
    print("🧪 Reasoning Mode Test Script")
    print("=" * 60)
    print(f"Testing families: {', '.join(families_to_test)}")
    print(f"Testing budgets: {budgets}")
    print(f"Test prompt: Chess position (Sicilian Najdorf)")
    print("=" * 60)
    
    reports = []
    
    if "anthropic" in families_to_test:
        reports.append(test_anthropic_family(budgets))
    
    if "gemini" in families_to_test:
        reports.append(test_gemini_family(budgets))
    
    if "openai" in families_to_test:
        reports.append(test_openai_family())
    
    if "xai" in families_to_test:
        reports.append(test_xai_family())
    
    generate_final_report(reports)


if __name__ == "__main__":
    main()

