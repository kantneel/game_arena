#!/usr/bin/env python3
"""Tests for the model stats service."""

import pytest
from services.model_stats_service import ModelStatsService


class TestModelStatsService:
    """Tests for ModelStatsService."""
    
    def test_get_all_models_empty(self, temp_results_dir):
        """Should return empty list for empty results dir."""
        service = ModelStatsService(temp_results_dir)
        result = service.get_all_models()
        assert result == []
    
    def test_get_all_models_with_matches(self, multiple_matches):
        """Should return all models from matches."""
        results_dir, _ = multiple_matches
        service = ModelStatsService(results_dir)
        
        result = service.get_all_models()
        
        assert len(result) > 0
        
        # Check structure
        model = result[0]
        assert "model_id" in model
        assert "display_name" in model
        assert "matches" in model
        assert "wins" in model
        assert "losses" in model
        assert "win_rate" in model
    
    def test_get_all_models_includes_both_participants(self, sample_match_dir):
        """Should include both models from a match."""
        service = ModelStatsService(sample_match_dir.parent)
        
        result = service.get_all_models()
        
        model_ids = [m["model_id"] for m in result]
        assert "claude-sonnet-4.5" in model_ids
        assert "gemini-3-flash" in model_ids
    
    def test_get_model_profile_returns_none_for_unknown(self, temp_results_dir):
        """Should return None for unknown model."""
        service = ModelStatsService(temp_results_dir)
        result = service.get_model_profile("unknown-model")
        assert result is None
    
    def test_get_model_profile_basic(self, sample_match_dir):
        """Should return profile for known model."""
        service = ModelStatsService(sample_match_dir.parent)
        
        result = service.get_model_profile("claude-sonnet-4.5")
        
        assert result is not None
        assert result.model_id == "claude-sonnet-4.5"
        assert result.total_matches == 1
        assert result.total_moves > 0
    
    def test_get_model_profile_has_pressure_stats(self, sample_match_dir):
        """Should include pressure statistics."""
        service = ModelStatsService(sample_match_dir.parent)
        
        result = service.get_model_profile("claude-sonnet-4.5")
        
        assert len(result.pressure_stats) > 0
    
    def test_get_model_profile_has_recent_matches(self, sample_match_dir):
        """Should include recent match history."""
        service = ModelStatsService(sample_match_dir.parent)
        
        result = service.get_model_profile("claude-sonnet-4.5")
        
        assert len(result.recent_matches) > 0
        match = result.recent_matches[0]
        assert match.opponent == "gemini-3-flash"
        assert match.result == "win"
    
    def test_get_model_profile_aggregate_multiple_matches(self, multiple_matches):
        """Should aggregate stats across multiple matches."""
        results_dir, _ = multiple_matches
        service = ModelStatsService(results_dir)
        
        result = service.get_model_profile("claude-sonnet-4.5")
        
        assert result is not None
        assert result.total_matches == 3
    
    def test_get_model_comparison(self, multiple_matches):
        """Should compare multiple models."""
        results_dir, _ = multiple_matches
        service = ModelStatsService(results_dir)
        
        result = service.get_model_comparison(["claude-sonnet-4.5", "gemini-3-flash"])
        
        assert "models" in result
        assert len(result["models"]) >= 1  # At least one found
        
        # Check structure
        if result["models"]:
            model = result["models"][0]
            assert "model_id" in model
            assert "speed_adaptation" in model
            assert "quality_degradation" in model

