#!/usr/bin/env python3
"""Tests for time-aware prompt generation with experimental features."""

import unittest
from unittest.mock import MagicMock, patch

from game_arena.blitz.prompts.time_aware import (
    create_time_aware_prompt_substitutions,
    create_response_feedback_text,
    PreviousResponseData,
)


class MockClock:
    """Mock clock for testing."""
    def __init__(self, time_remaining: float):
        self.time_remaining = time_remaining


def create_mock_pyspiel_state():
    """Create a properly mocked pyspiel state."""
    mock_state = MagicMock()
    mock_state.to_string.return_value = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    mock_state.current_player.return_value = 0  # White
    
    # Mock the game object chain
    mock_type = MagicMock()
    mock_type.short_name = "chess"
    
    mock_game = MagicMock()
    mock_game.get_type.return_value = mock_type
    
    mock_state.get_game.return_value = mock_game
    mock_state.history.return_value = []  # Empty history
    
    return mock_state


class TestCreateResponseFeedbackText(unittest.TestCase):
    """Tests for response feedback text generation."""
    
    def test_none_previous_data_returns_empty(self):
        """Should return empty string when no previous data."""
        result = create_response_feedback_text(None, 120.0)
        self.assertEqual(result, "")
    
    def test_basic_feedback_includes_time_and_tokens(self):
        """Should include time taken and token count."""
        prev_data = PreviousResponseData(
            time_taken_seconds=8.5,
            thinking_tokens=12000,
            output_tokens=100,
            tokens_per_second=1411.8,
            time_remaining_after=250.0,
        )
        
        result = create_response_feedback_text(prev_data, 250.0)
        
        self.assertIn("8.5 seconds", result)
        self.assertIn("12,000 thinking tokens", result)
        self.assertIn("1412 tokens/second", result)  # Rounded
        self.assertIn("PREVIOUS RESPONSE ANALYSIS", result)
    
    def test_efficiency_guidance_disabled_by_default(self):
        """Should not include efficiency guidance unless enabled."""
        prev_data = PreviousResponseData(
            time_taken_seconds=5.0,
            thinking_tokens=10000,
            output_tokens=100,
            tokens_per_second=2000.0,
            time_remaining_after=100.0,
        )
        
        result = create_response_feedback_text(prev_data, 100.0, include_efficiency_guidance=False)
        
        self.assertNotIn("EFFICIENCY GUIDANCE", result)
        self.assertNotIn("generating", result)
    
    def test_efficiency_guidance_when_enabled(self):
        """Should include token affordability calculations when enabled."""
        prev_data = PreviousResponseData(
            time_taken_seconds=5.0,
            thinking_tokens=10000,
            output_tokens=100,
            tokens_per_second=2000.0,
            time_remaining_after=100.0,
        )
        
        result = create_response_feedback_text(prev_data, 100.0, include_efficiency_guidance=True)
        
        self.assertIn("EFFICIENCY GUIDANCE", result)
        self.assertIn("tokens would take", result)
    
    def test_low_time_warning(self):
        """Should include warning when time is low (under 60s)."""
        prev_data = PreviousResponseData(
            time_taken_seconds=5.0,
            thinking_tokens=10000,
            output_tokens=100,
            tokens_per_second=2000.0,
            time_remaining_after=45.0,
        )
        
        result = create_response_feedback_text(prev_data, 45.0, include_efficiency_guidance=True)
        
        # Should contain a warning about preserving time
        self.assertIn("⚠️", result)
        self.assertIn("shorter reasoning", result)
    
    def test_critical_time_warning(self):
        """Should include critical warning when time is very low (<30s)."""
        prev_data = PreviousResponseData(
            time_taken_seconds=5.0,
            thinking_tokens=10000,
            output_tokens=100,
            tokens_per_second=2000.0,
            time_remaining_after=25.0,
        )
        
        result = create_response_feedback_text(prev_data, 25.0, include_efficiency_guidance=True)
        
        # Should contain critical urgency warning
        self.assertIn("🚨", result)
        self.assertIn("Minimize thinking", result)


class TestCreateTimeAwarePromptSubstitutions(unittest.TestCase):
    """Tests for time-aware prompt substitution generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state = create_mock_pyspiel_state()
        self.player_clock = MockClock(180.0)  # 3 minutes
        self.opponent_clock = MockClock(200.0)
    
    def test_basic_substitutions_always_present(self):
        """Should always include basic game substitutions."""
        result = create_time_aware_prompt_substitutions(
            self.state,
            self.player_clock,
            self.opponent_clock,
            increment_seconds=3,
        )
        
        self.assertIn("readable_state_str", result)
        self.assertIn("move_history", result)
        self.assertIn("player_name", result)
    
    def test_time_pressure_prompt_enabled_by_default(self):
        """Should include time info when enabled (default)."""
        result = create_time_aware_prompt_substitutions(
            self.state,
            self.player_clock,
            self.opponent_clock,
            increment_seconds=3,
            is_blitz=True,
            enable_time_pressure_prompt=True,
        )
        
        self.assertIn("time_info", result)
        self.assertIn("BLITZ CHESS TIME INFORMATION", result["time_info"])
        self.assertIn("3:00", result["time_info"])  # 180 seconds formatted
        self.assertIn("CRITICAL TIME RULES", result["time_info"])
    
    def test_time_pressure_prompt_disabled(self):
        """Should NOT include time info when disabled (ablation)."""
        result = create_time_aware_prompt_substitutions(
            self.state,
            self.player_clock,
            self.opponent_clock,
            increment_seconds=3,
            is_blitz=True,
            enable_time_pressure_prompt=False,
        )
        
        self.assertIn("time_info", result)
        self.assertEqual(result["time_info"], "")
        self.assertEqual(result["dramatic_time_pressure"], "")
    
    def test_dramatic_prompts_disabled_by_default(self):
        """Should use standard prompts by default."""
        result = create_time_aware_prompt_substitutions(
            self.state,
            self.player_clock,
            self.opponent_clock,
            increment_seconds=3,
            is_blitz=True,
            use_dramatic_pressure=False,
        )
        
        self.assertEqual(result["dramatic_time_pressure"], "")
        self.assertNotEqual(result["time_info"], "")
    
    def test_dramatic_prompts_enabled(self):
        """Should include dramatic ALL-CAPS content when enabled."""
        low_time_clock = MockClock(45.0)  # Under 1 minute
        
        result = create_time_aware_prompt_substitutions(
            self.state,
            low_time_clock,
            self.opponent_clock,
            increment_seconds=3,
            is_blitz=True,
            use_dramatic_pressure=True,
        )
        
        # Dramatic prompts should be present
        self.assertIn("dramatic_time_pressure", result)
        self.assertIn("dramatic_instruction", result)
        # Should contain urgency markers
        dramatic = result["dramatic_time_pressure"]
        self.assertTrue(
            "ALERT" in dramatic or "PRESSURE" in dramatic or "🚨" in dramatic,
            f"Expected urgency markers in: {dramatic}"
        )
    
    def test_response_feedback_disabled_by_default(self):
        """Should NOT include response feedback when disabled."""
        prev_data = PreviousResponseData(
            time_taken_seconds=5.0,
            thinking_tokens=10000,
            output_tokens=100,
            tokens_per_second=2000.0,
            time_remaining_after=175.0,
        )
        
        result = create_time_aware_prompt_substitutions(
            self.state,
            self.player_clock,
            self.opponent_clock,
            increment_seconds=3,
            previous_response_data=prev_data,
            enable_response_feedback=False,
        )
        
        self.assertEqual(result["response_feedback"], "")
    
    def test_response_feedback_enabled(self):
        """Should include response feedback when enabled."""
        prev_data = PreviousResponseData(
            time_taken_seconds=8.5,
            thinking_tokens=12000,
            output_tokens=100,
            tokens_per_second=1411.8,
            time_remaining_after=175.0,
        )
        
        result = create_time_aware_prompt_substitutions(
            self.state,
            self.player_clock,
            self.opponent_clock,
            increment_seconds=3,
            previous_response_data=prev_data,
            enable_response_feedback=True,
        )
        
        feedback = result["response_feedback"]
        self.assertIn("8.5 seconds", feedback)
        self.assertIn("12,000", feedback)
    
    def test_response_feedback_without_previous_data(self):
        """Should return empty feedback when no previous data even if enabled."""
        result = create_time_aware_prompt_substitutions(
            self.state,
            self.player_clock,
            self.opponent_clock,
            increment_seconds=3,
            previous_response_data=None,
            enable_response_feedback=True,
        )
        
        self.assertEqual(result["response_feedback"], "")
    
    def test_efficiency_guidance_enabled(self):
        """Should include efficiency guidance when both flags are set."""
        prev_data = PreviousResponseData(
            time_taken_seconds=5.0,
            thinking_tokens=10000,
            output_tokens=100,
            tokens_per_second=2000.0,
            time_remaining_after=50.0,
        )
        
        low_time_clock = MockClock(50.0)
        
        result = create_time_aware_prompt_substitutions(
            self.state,
            low_time_clock,
            self.opponent_clock,
            increment_seconds=3,
            previous_response_data=prev_data,
            enable_response_feedback=True,
            enable_efficiency_guidance=True,
        )
        
        feedback = result["response_feedback"]
        self.assertIn("EFFICIENCY GUIDANCE", feedback)
    
    def test_pressure_level_changes_with_time(self):
        """Should show different pressure levels based on time remaining."""
        # Comfortable (>120s)
        comfortable_clock = MockClock(180.0)
        result = create_time_aware_prompt_substitutions(
            self.state, comfortable_clock, self.opponent_clock, 3, is_blitz=True
        )
        self.assertIn("🟢 LOW", result["time_info"])
        
        # Medium (60-120s)
        medium_clock = MockClock(90.0)
        result = create_time_aware_prompt_substitutions(
            self.state, medium_clock, self.opponent_clock, 3, is_blitz=True
        )
        self.assertIn("🟡 MEDIUM", result["time_info"])
        
        # High (<60s)
        high_clock = MockClock(45.0)
        result = create_time_aware_prompt_substitutions(
            self.state, high_clock, self.opponent_clock, 3, is_blitz=True
        )
        self.assertIn("🔴 HIGH", result["time_info"])
    
    def test_non_blitz_has_no_time_info(self):
        """Should not include time info for non-blitz games."""
        result = create_time_aware_prompt_substitutions(
            self.state,
            self.player_clock,
            self.opponent_clock,
            increment_seconds=3,
            is_blitz=False,
        )
        
        self.assertEqual(result["time_info"], "")
        self.assertEqual(result["dramatic_time_pressure"], "")


class TestCombinedFlags(unittest.TestCase):
    """Test combinations of experimental flags."""
    
    def setUp(self):
        self.state = create_mock_pyspiel_state()
        self.player_clock = MockClock(45.0)  # Low time for urgency
        self.opponent_clock = MockClock(200.0)
        self.prev_data = PreviousResponseData(
            time_taken_seconds=8.0,
            thinking_tokens=10000,
            output_tokens=100,
            tokens_per_second=1250.0,
            time_remaining_after=50.0,
        )
    
    def test_all_features_enabled(self):
        """Should include all features when all flags are on."""
        result = create_time_aware_prompt_substitutions(
            self.state,
            self.player_clock,
            self.opponent_clock,
            increment_seconds=3,
            is_blitz=True,
            use_dramatic_pressure=True,
            enable_time_pressure_prompt=True,
            previous_response_data=self.prev_data,
            enable_response_feedback=True,
            enable_efficiency_guidance=True,
        )
        
        # Dramatic prompts present
        self.assertNotEqual(result["dramatic_time_pressure"], "")
        
        # Response feedback present with guidance
        feedback = result["response_feedback"]
        self.assertIn("PREVIOUS RESPONSE", feedback)
        self.assertIn("EFFICIENCY GUIDANCE", feedback)
    
    def test_ablation_disables_time_but_not_feedback(self):
        """Disabling time prompt should not affect response feedback."""
        result = create_time_aware_prompt_substitutions(
            self.state,
            self.player_clock,
            self.opponent_clock,
            increment_seconds=3,
            is_blitz=True,
            enable_time_pressure_prompt=False,  # Ablation
            previous_response_data=self.prev_data,
            enable_response_feedback=True,
        )
        
        # No time info
        self.assertEqual(result["time_info"], "")
        
        # But response feedback still present
        self.assertIn("PREVIOUS RESPONSE", result["response_feedback"])
    
    def test_dramatic_without_time_pressure(self):
        """Dramatic flag should have no effect when time pressure is disabled."""
        result = create_time_aware_prompt_substitutions(
            self.state,
            self.player_clock,
            self.opponent_clock,
            increment_seconds=3,
            is_blitz=True,
            enable_time_pressure_prompt=False,
            use_dramatic_pressure=True,  # Should be ignored
        )
        
        self.assertEqual(result["time_info"], "")
        self.assertEqual(result["dramatic_time_pressure"], "")


class TestPreviousResponseData(unittest.TestCase):
    """Test PreviousResponseData dataclass."""
    
    def test_creation(self):
        """Should create data object correctly."""
        data = PreviousResponseData(
            time_taken_seconds=5.5,
            thinking_tokens=8000,
            output_tokens=150,
            tokens_per_second=1454.5,
            time_remaining_after=120.0,
        )
        
        self.assertEqual(data.time_taken_seconds, 5.5)
        self.assertEqual(data.thinking_tokens, 8000)
        self.assertEqual(data.output_tokens, 150)
        self.assertAlmostEqual(data.tokens_per_second, 1454.5, places=1)
        self.assertEqual(data.time_remaining_after, 120.0)


if __name__ == "__main__":
    unittest.main()
