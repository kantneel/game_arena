#!/usr/bin/env python3
"""Tests for the analysis service."""

import pytest
from services.analysis_service import (
    AnalysisService,
    categorize_pressure,
    PRESSURE_THRESHOLDS,
)


class TestCategorizePresure:
    """Tests for pressure categorization."""
    
    def test_critical_pressure(self):
        """Time under 30s should be critical."""
        assert categorize_pressure(0) == "critical"
        assert categorize_pressure(15) == "critical"
        assert categorize_pressure(29.9) == "critical"
    
    def test_high_pressure(self):
        """Time 30-60s should be high."""
        assert categorize_pressure(30) == "high"
        assert categorize_pressure(45) == "high"
        assert categorize_pressure(59.9) == "high"
    
    def test_medium_pressure(self):
        """Time 60-120s should be medium."""
        assert categorize_pressure(60) == "medium"
        assert categorize_pressure(90) == "medium"
        assert categorize_pressure(119.9) == "medium"
    
    def test_comfortable_pressure(self):
        """Time over 120s should be comfortable."""
        assert categorize_pressure(120) == "comfortable"
        assert categorize_pressure(300) == "comfortable"
        assert categorize_pressure(1000) == "comfortable"


class TestAnalysisService:
    """Tests for the AnalysisService."""
    
    def test_analyze_match_returns_none_for_missing(self, temp_results_dir):
        """Should return None for non-existent match."""
        service = AnalysisService(temp_results_dir)
        result = service.analyze_match("nonexistent_match")
        assert result is None
    
    def test_analyze_match_basic(self, sample_match_dir):
        """Should return analysis for valid match."""
        service = AnalysisService(sample_match_dir.parent)
        match_id = sample_match_dir.name
        
        result = service.analyze_match(match_id)
        
        assert result is not None
        assert result.match_id == match_id
        assert result.model_a_profile.model_name == "claude-sonnet-4.5"
        assert result.model_b_profile.model_name == "gemini-3-flash"
    
    def test_analyze_match_has_pressure_stats(self, sample_match_dir):
        """Should compute pressure statistics."""
        service = AnalysisService(sample_match_dir.parent)
        match_id = sample_match_dir.name
        
        result = service.analyze_match(match_id)
        
        assert len(result.model_a_profile.pressure_stats) > 0
        
        # Check that stats have expected fields
        for stat in result.model_a_profile.pressure_stats:
            assert stat.pressure_level in ["comfortable", "medium", "high", "critical"]
            assert stat.move_count > 0
            assert stat.avg_move_time > 0
    
    def test_analyze_match_computes_adaptation_ratio(self, sample_match_dir):
        """Should compute speed adaptation ratio."""
        service = AnalysisService(sample_match_dir.parent)
        match_id = sample_match_dir.name
        
        result = service.analyze_match(match_id)
        
        # Claude should show adaptation (speeds up under pressure)
        # Based on our fixture, Claude uses less time when time_remaining < 60
        assert result.model_a_profile.speed_adaptation_ratio < 1.0
        
        # Gemini uses constant time, so ratio should be ~1.0
        assert 0.8 < result.model_b_profile.speed_adaptation_ratio < 1.2
    
    def test_analyze_match_with_move_analysis(self, sample_move_analysis):
        """Should include centipawn loss when available."""
        service = AnalysisService(sample_move_analysis.parent)
        match_id = sample_move_analysis.name
        
        result = service.analyze_match(match_id)
        
        # Should have centipawn loss data
        has_cpl = False
        for stat in result.model_a_profile.pressure_stats:
            if stat.avg_centipawn_loss is not None:
                has_cpl = True
                break
        
        assert has_cpl, "Should have centipawn loss data when analysis file exists"
    
    def test_analyze_match_generates_insights(self, sample_match_dir):
        """Should generate human-readable insights."""
        service = AnalysisService(sample_match_dir.parent)
        match_id = sample_match_dir.name
        
        result = service.analyze_match(match_id)
        
        assert len(result.insights) > 0
        assert all(isinstance(i, str) for i in result.insights)
    
    def test_get_pressure_scatter_data(self, sample_match_dir):
        """Should return scatter plot data."""
        service = AnalysisService(sample_match_dir.parent)
        match_id = sample_match_dir.name
        
        result = service.get_pressure_scatter_data(match_id)
        
        assert result is not None
        assert "model_a" in result
        assert "model_b" in result
        assert "points" in result
        assert len(result["points"]) > 0
        
        # Check point structure
        point = result["points"][0]
        assert "model" in point
        assert "time_remaining" in point
        assert "move_time" in point
    
    def test_get_thinking_by_pressure(self, sample_match_dir):
        """Should return thinking tokens grouped by pressure."""
        service = AnalysisService(sample_match_dir.parent)
        match_id = sample_match_dir.name
        
        result = service.get_thinking_by_pressure(match_id)
        
        assert result is not None
        assert "data" in result
        assert len(result["data"]) == 4  # 4 pressure levels
        
        # Check structure
        for item in result["data"]:
            assert "pressure" in item
            assert "model_a_avg_tokens" in item
            assert "model_b_avg_tokens" in item


class TestAggregateAnalysis:
    """Tests for aggregate model analysis."""
    
    def test_analyze_model_aggregate(self, multiple_matches):
        """Should aggregate stats across multiple matches."""
        results_dir, match_ids = multiple_matches
        service = AnalysisService(results_dir)
        
        result = service.analyze_model_aggregate("claude-sonnet-4.5", match_ids)
        
        assert result is not None
        assert result.model_name == "claude-sonnet-4.5"
        assert result.total_moves > 0
    
    def test_analyze_model_aggregate_returns_none_for_unknown(self, multiple_matches):
        """Should return None for model not in any match."""
        results_dir, match_ids = multiple_matches
        service = AnalysisService(results_dir)
        
        result = service.analyze_model_aggregate("unknown-model", match_ids)
        
        assert result is None

