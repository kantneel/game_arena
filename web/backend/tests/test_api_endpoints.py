#!/usr/bin/env python3
"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import json
import pandas as pd

# We need to set up the test environment before importing the app
import sys


@pytest.fixture
def test_client(sample_match_dir):
    """Create a test client with sample data."""
    # Import app here to avoid circular imports
    from main import app
    from services.match_service import MatchService
    from services.analysis_service import AnalysisService
    from services.model_stats_service import ModelStatsService
    
    # Override services with test data
    results_dir = sample_match_dir.parent
    app.state.match_service = MatchService(results_dir)
    app.state.match_service.scan_results()
    app.state.analysis_service = AnalysisService(results_dir)
    app.state.model_stats_service = ModelStatsService(results_dir)
    
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""
    
    def test_health_returns_ok(self, test_client):
        """Health endpoint should return ok."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestMatchesEndpoints:
    """Tests for matches API endpoints."""
    
    def test_list_matches(self, test_client):
        """Should list all matches."""
        response = test_client.get("/api/matches")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_get_match_detail(self, test_client, sample_match_dir):
        """Should get match details."""
        match_id = sample_match_dir.name
        response = test_client.get(f"/api/matches/{match_id}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["match_id"] == match_id
        assert "games" in data
    
    def test_get_match_not_found(self, test_client):
        """Should return 404 for unknown match."""
        response = test_client.get("/api/matches/nonexistent")
        assert response.status_code == 404


class TestAnalysisEndpoints:
    """Tests for analysis API endpoints."""
    
    def test_get_match_analysis(self, test_client, sample_match_dir):
        """Should return match analysis."""
        match_id = sample_match_dir.name
        response = test_client.get(f"/api/analysis/matches/{match_id}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["match_id"] == match_id
        assert "model_a" in data
        assert "model_b" in data
        assert "insights" in data
    
    def test_get_pressure_scatter(self, test_client, sample_match_dir):
        """Should return scatter plot data."""
        match_id = sample_match_dir.name
        response = test_client.get(f"/api/analysis/matches/{match_id}/scatter")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "points" in data
        assert len(data["points"]) > 0
    
    def test_get_thinking_by_pressure(self, test_client, sample_match_dir):
        """Should return thinking tokens by pressure."""
        match_id = sample_match_dir.name
        response = test_client.get(f"/api/analysis/matches/{match_id}/thinking")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "data" in data


class TestModelsEndpoints:
    """Tests for models API endpoints."""
    
    def test_list_models(self, test_client):
        """Should list all models."""
        response = test_client.get("/api/models")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_get_model_profile(self, test_client):
        """Should get model profile."""
        response = test_client.get("/api/models/claude-sonnet-4.5")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_id"] == "claude-sonnet-4.5"
        assert "pressure_stats" in data
        assert "recent_matches" in data
    
    def test_get_model_not_found(self, test_client):
        """Should return 404 for unknown model."""
        response = test_client.get("/api/models/nonexistent-model")
        assert response.status_code == 404

