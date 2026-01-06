#!/usr/bin/env python3
"""Tests for the offline evaluation module."""

import json
import tempfile
from pathlib import Path

import pytest

from game_arena.blitz.offline_eval.position_dataset import (
    ChessPosition,
    PositionDataset,
    create_standard_dataset,
    save_dataset,
    load_dataset,
)
from game_arena.blitz.offline_eval.evaluator import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationSession,
    PromptStyle,
)
from game_arena.blitz.offline_eval.analysis import (
    aggregate_results,
    OfflineAnalyzer,
)


class TestChessPosition:
    """Tests for ChessPosition dataclass."""
    
    def test_create_basic_position(self):
        """Should create a position with required fields."""
        pos = ChessPosition(
            position_id="test_001",
            fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            category="opening",
            difficulty="easy",
            description="After 1. e4",
        )
        
        assert pos.position_id == "test_001"
        assert "e4" in pos.fen or "4P3" in pos.fen
        assert pos.category == "opening"
        assert pos.difficulty == "easy"
    
    def test_position_with_analysis(self):
        """Should store Stockfish analysis data."""
        pos = ChessPosition(
            position_id="test_002",
            fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            category="opening",
            difficulty="easy",
            description="Italian Game",
            best_move="Bb5",
            best_move_eval=45,
            second_best_move="Bc4",
            second_best_eval=35,
        )
        
        assert pos.best_move == "Bb5"
        assert pos.best_move_eval == 45
        assert pos.second_best_move == "Bc4"
    
    def test_position_tags(self):
        """Should support tags for categorization."""
        pos = ChessPosition(
            position_id="test_003",
            fen="8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
            category="endgame",
            difficulty="easy",
            description="K+P vs K",
            tags=["pawn_endgame", "opposition", "basic"],
        )
        
        assert "pawn_endgame" in pos.tags
        assert len(pos.tags) == 3


class TestPositionDataset:
    """Tests for PositionDataset."""
    
    @pytest.fixture
    def sample_positions(self):
        """Create sample positions for testing."""
        return [
            ChessPosition(
                position_id="open_001",
                fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                category="opening",
                difficulty="easy",
                description="After 1. e4",
            ),
            ChessPosition(
                position_id="mid_001",
                fen="r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
                category="middlegame",
                difficulty="medium",
                description="Italian middlegame",
            ),
            ChessPosition(
                position_id="end_001",
                fen="8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
                category="endgame",
                difficulty="easy",
                description="K+P vs K",
            ),
            ChessPosition(
                position_id="tac_001",
                fen="r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                category="tactical",
                difficulty="easy",
                description="Scholar's mate",
            ),
        ]
    
    def test_create_dataset(self, sample_positions):
        """Should create a dataset from positions."""
        dataset = PositionDataset(
            name="test_dataset",
            description="Test positions",
            positions=sample_positions,
        )
        
        assert len(dataset) == 4
        assert dataset.name == "test_dataset"
    
    def test_filter_by_category(self, sample_positions):
        """Should filter positions by category."""
        dataset = PositionDataset(
            name="test",
            description="Test",
            positions=sample_positions,
        )
        
        opening_only = dataset.filter_by_category("opening")
        assert len(opening_only) == 1
        assert opening_only.positions[0].position_id == "open_001"
    
    def test_filter_by_difficulty(self, sample_positions):
        """Should filter positions by difficulty."""
        dataset = PositionDataset(
            name="test",
            description="Test",
            positions=sample_positions,
        )
        
        easy_only = dataset.filter_by_difficulty("easy")
        assert len(easy_only) == 3  # open_001, end_001, tac_001
    
    def test_iterate_dataset(self, sample_positions):
        """Should iterate over positions."""
        dataset = PositionDataset(
            name="test",
            description="Test",
            positions=sample_positions,
        )
        
        ids = [p.position_id for p in dataset]
        assert len(ids) == 4
        assert "open_001" in ids


class TestDatasetPersistence:
    """Tests for saving and loading datasets."""
    
    def test_save_and_load(self):
        """Should save dataset to JSON and load it back."""
        positions = [
            ChessPosition(
                position_id="test_001",
                fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                category="opening",
                difficulty="easy",
                description="After 1. e4",
                best_move="e5",
                tags=["open_game"],
            ),
        ]
        
        original = PositionDataset(
            name="test_persist",
            description="Test persistence",
            positions=positions,
            version="1.0",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.json"
            save_dataset(original, path)
            
            assert path.exists()
            
            loaded = load_dataset(path)
            
            assert loaded.name == original.name
            assert loaded.description == original.description
            assert len(loaded.positions) == 1
            assert loaded.positions[0].position_id == "test_001"
            assert loaded.positions[0].best_move == "e5"
            assert "open_game" in loaded.positions[0].tags


class TestStandardDataset:
    """Tests for the standard evaluation dataset."""
    
    def test_create_standard_dataset(self):
        """Should create standard dataset with diverse positions."""
        dataset = create_standard_dataset()
        
        assert len(dataset) > 10
        assert dataset.name == "standard_eval_v1"
    
    def test_standard_dataset_has_all_categories(self):
        """Standard dataset should cover all categories."""
        dataset = create_standard_dataset()
        
        categories = set(p.category for p in dataset.positions)
        
        assert "opening" in categories
        assert "endgame" in categories
        # Either tactical or middlegame should be present
        assert "tactical" in categories or "middlegame" in categories
    
    def test_standard_dataset_has_difficulty_levels(self):
        """Standard dataset should have various difficulty levels."""
        dataset = create_standard_dataset()
        
        difficulties = set(p.difficulty for p in dataset.positions)
        
        assert len(difficulties) >= 2  # At least easy and medium/hard


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""
    
    def test_default_config(self):
        """Should have sensible defaults."""
        config = EvaluationConfig()
        
        assert 300 in config.time_levels  # Comfortable
        assert 15 in config.time_levels   # Critical
        assert config.samples_per_condition >= 1
        assert config.reasoning_budget > 0
    
    def test_custom_config(self):
        """Should accept custom values."""
        config = EvaluationConfig(
            time_levels=[180, 60, 20],
            samples_per_condition=5,
            prompt_style=PromptStyle.NONE,
        )
        
        assert config.time_levels == [180, 60, 20]
        assert config.samples_per_condition == 5
        # PromptStyle.NONE should disable time pressure prompt
        assert config.enable_time_pressure_prompt is False
    
    def test_prompt_style_properties(self):
        """Should derive boolean flags from prompt_style."""
        # NONE style
        config = EvaluationConfig(prompt_style=PromptStyle.NONE)
        assert config.enable_time_pressure_prompt is False
        assert config.use_dramatic_prompts is False
        
        # TIME_INFO_ONLY style
        config = EvaluationConfig(prompt_style=PromptStyle.TIME_INFO_ONLY)
        assert config.enable_time_pressure_prompt is True
        assert config.use_dramatic_prompts is False
        
        # STANDARD style
        config = EvaluationConfig(prompt_style=PromptStyle.STANDARD)
        assert config.enable_time_pressure_prompt is True
        assert config.use_dramatic_prompts is False
        
        # DRAMATIC style
        config = EvaluationConfig(prompt_style=PromptStyle.DRAMATIC)
        assert config.enable_time_pressure_prompt is True
        assert config.use_dramatic_prompts is True
    
    def test_timeout_config(self):
        """Should have timeout simulation settings."""
        config = EvaluationConfig()
        
        # Default should simulate clock
        assert config.simulate_clock is True
        
        # Should allow disabling
        config = EvaluationConfig(simulate_clock=False)
        assert config.simulate_clock is False
    
    def test_latency_calibration_config(self):
        """Should have latency calibration settings."""
        config = EvaluationConfig()
        
        # Default should calibrate
        assert config.calibrate_latency is True
        assert config.calibration_samples >= 1
        
        # Should allow customization
        config = EvaluationConfig(calibrate_latency=False, calibration_samples=5)
        assert config.calibrate_latency is False
        assert config.calibration_samples == 5


class TestEvaluationResult:
    """Tests for EvaluationResult."""
    
    def test_create_result(self):
        """Should create evaluation result with all fields."""
        result = EvaluationResult(
            position_id="test_001",
            model_id="gemini-3-pro",
            time_remaining=60.0,
            sample_index=0,
            move_played="Nf3",
            thinking_tokens=5000,
            output_tokens=50,
            response_time_seconds=8.5,
            full_response="The best move is Nf3...",
        )
        
        assert result.position_id == "test_001"
        assert result.model_id == "gemini-3-pro"
        assert result.time_remaining == 60.0
        assert result.move_played == "Nf3"
        assert result.thinking_tokens == 5000
    
    def test_result_timeout_tracking(self):
        """Should track timeout/forfeit status."""
        # Result that would timeout
        result = EvaluationResult(
            position_id="test_001",
            model_id="gemini-3-pro",
            time_remaining=10.0,  # 10 seconds left
            sample_index=0,
            move_played="Nf3",
            thinking_tokens=5000,
            output_tokens=50,
            response_time_seconds=15.0,  # Took 15 seconds - would timeout!
            full_response="The best move is Nf3...",
            would_timeout=True,
            time_after_move=-2.0,  # 10 - 15 + 3 increment = -2
        )
        
        assert result.would_timeout is True
        assert result.time_after_move < 0  # Would be flagged
        
        # Result that wouldn't timeout
        result2 = EvaluationResult(
            position_id="test_002",
            model_id="gemini-3-pro",
            time_remaining=60.0,
            sample_index=0,
            move_played="e4",
            thinking_tokens=2000,
            output_tokens=20,
            response_time_seconds=8.0,
            full_response="1. e4...",
            would_timeout=False,
            time_after_move=55.0,  # 60 - 8 + 3 = 55
        )
        
        assert result2.would_timeout is False
        assert result2.time_after_move == 55.0
    
    def test_result_prompt_style_tracking(self):
        """Should track which prompt style was used."""
        result = EvaluationResult(
            position_id="test_001",
            model_id="gemini-3-pro",
            time_remaining=60.0,
            sample_index=0,
            move_played="Nf3",
            thinking_tokens=5000,
            output_tokens=50,
            response_time_seconds=8.5,
            full_response="...",
            prompt_style="time_info_only",
            network_latency=0.5,
        )
        
        assert result.prompt_style == "time_info_only"
        assert result.network_latency == 0.5
    
    def test_result_quality_metrics(self):
        """Should store quality metrics."""
        result = EvaluationResult(
            position_id="test_001",
            model_id="gemini-3-pro",
            time_remaining=30.0,
            sample_index=0,
            move_played="Qxf7",
            thinking_tokens=3000,
            output_tokens=30,
            response_time_seconds=5.0,
            full_response="Qxf7#",
            centipawn_loss=0.0,
            is_best_move=True,
            is_blunder=False,
            move_rank=1,
        )
        
        assert result.is_best_move is True
        assert result.centipawn_loss == 0.0
        assert result.move_rank == 1


class TestEvaluationSession:
    """Tests for EvaluationSession."""
    
    @pytest.fixture
    def sample_session(self):
        """Create a sample session with results."""
        config = EvaluationConfig(
            time_levels=[60, 30],
            samples_per_condition=2,
        )
        
        session = EvaluationSession(
            session_id="test_session_001",
            model_id="gemini-3-pro",
            dataset_name="test_dataset",
            config=config,
            start_time="2024-01-01T10:00:00",
        )
        
        # Add some results
        for time_level in [60, 30]:
            for sample in range(2):
                result = EvaluationResult(
                    position_id="pos_001",
                    model_id="gemini-3-pro",
                    time_remaining=time_level,
                    sample_index=sample,
                    move_played="e4",
                    thinking_tokens=5000 - (60 - time_level) * 50,  # Fewer tokens under pressure
                    output_tokens=30,
                    response_time_seconds=8.0 - (60 - time_level) * 0.1,  # Faster under pressure
                    full_response="Move: e4",
                )
                session.add_result(result)
        
        session.end_time = "2024-01-01T10:30:00"
        return session
    
    def test_add_result(self, sample_session):
        """Should add results to session."""
        assert len(sample_session.results) == 4
    
    def test_save_and_load_session(self, sample_session):
        """Should persist session to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.json"
            sample_session.save(path)
            
            assert path.exists()
            
            loaded = EvaluationSession.load(path)
            
            assert loaded.session_id == sample_session.session_id
            assert loaded.model_id == sample_session.model_id
            assert len(loaded.results) == len(sample_session.results)
            assert loaded.config.time_levels == sample_session.config.time_levels


class TestAggregateResults:
    """Tests for result aggregation."""
    
    def test_aggregate_by_time_level(self):
        """Should aggregate results by time level."""
        results = [
            {"model_id": "gemini-3-pro", "time_remaining": 60, 
             "response_time_seconds": 8.0, "thinking_tokens": 5000, "output_tokens": 30},
            {"model_id": "gemini-3-pro", "time_remaining": 60, 
             "response_time_seconds": 7.5, "thinking_tokens": 4800, "output_tokens": 28},
            {"model_id": "gemini-3-pro", "time_remaining": 30, 
             "response_time_seconds": 5.0, "thinking_tokens": 3000, "output_tokens": 25},
            {"model_id": "gemini-3-pro", "time_remaining": 30, 
             "response_time_seconds": 4.5, "thinking_tokens": 2800, "output_tokens": 22},
        ]
        
        df = aggregate_results(results)
        
        assert len(df) == 2  # Two time levels
        
        row_60 = df[df['time_remaining'] == 60].iloc[0]
        assert row_60['response_time_seconds_mean'] == pytest.approx(7.75, 0.01)
        
        row_30 = df[df['time_remaining'] == 30].iloc[0]
        assert row_30['response_time_seconds_mean'] == pytest.approx(4.75, 0.01)
    
    def test_aggregate_multiple_models(self):
        """Should aggregate results for multiple models."""
        results = [
            {"model_id": "gemini-3-pro", "time_remaining": 60, 
             "response_time_seconds": 8.0, "thinking_tokens": 5000, "output_tokens": 30},
            {"model_id": "gemini-3-flash", "time_remaining": 60, 
             "response_time_seconds": 4.0, "thinking_tokens": 2000, "output_tokens": 25},
        ]
        
        df = aggregate_results(results)
        
        assert len(df) == 2  # Two models
        assert set(df['model_id'].values) == {"gemini-3-pro", "gemini-3-flash"}


class TestOfflineAnalyzer:
    """Tests for OfflineAnalyzer."""
    
    @pytest.fixture
    def sample_sessions(self):
        """Create sample sessions for analysis."""
        sessions = []
        
        for model in ["gemini-3-pro", "gemini-3-flash"]:
            config = EvaluationConfig(time_levels=[60, 30])
            session = EvaluationSession(
                session_id=f"test_{model}",
                model_id=model,
                dataset_name="test",
                config=config,
            )
            
            for pos in ["pos_001", "pos_002"]:
                for time_level in [60, 30]:
                    for sample in range(2):
                        # Flash is faster but uses fewer tokens
                        base_time = 8.0 if model == "gemini-3-pro" else 4.0
                        base_tokens = 5000 if model == "gemini-3-pro" else 2000
                        
                        result = EvaluationResult(
                            position_id=pos,
                            model_id=model,
                            time_remaining=time_level,
                            sample_index=sample,
                            move_played="e4",
                            thinking_tokens=int(base_tokens * (time_level / 60)),
                            output_tokens=30,
                            response_time_seconds=base_time * (time_level / 60),
                            full_response="e4",
                        )
                        session.add_result(result)
            
            sessions.append(session)
        
        return sessions
    
    def test_to_dataframe(self, sample_sessions):
        """Should convert sessions to DataFrame."""
        analyzer = OfflineAnalyzer(sessions=sample_sessions)
        df = analyzer.to_dataframe()
        
        assert len(df) == 16  # 2 models * 2 positions * 2 time levels * 2 samples
        assert "model_id" in df.columns
        assert "time_remaining" in df.columns
        assert "response_time_seconds" in df.columns
    
    def test_get_time_pressure_curves(self, sample_sessions):
        """Should compute time pressure response curves."""
        analyzer = OfflineAnalyzer(sessions=sample_sessions)
        curves = analyzer.get_time_pressure_curves()
        
        assert "gemini-3-pro" in curves
        assert "gemini-3-flash" in curves
        
        pro_curve = curves["gemini-3-pro"]
        assert len(pro_curve) == 2  # Two time levels
    
    def test_get_variance_analysis(self, sample_sessions):
        """Should analyze variance across samples."""
        analyzer = OfflineAnalyzer(sessions=sample_sessions)
        variance = analyzer.get_variance_analysis()
        
        assert "time_cv" in variance.columns
        assert "unique_moves" in variance.columns
    
    def test_compare_models(self, sample_sessions):
        """Should compare models at each time level."""
        analyzer = OfflineAnalyzer(sessions=sample_sessions)
        comparison = analyzer.compare_models()
        
        assert "gemini-3-pro" in comparison.columns
        assert "gemini-3-flash" in comparison.columns
    
    def test_generate_report(self, sample_sessions):
        """Should generate markdown report."""
        analyzer = OfflineAnalyzer(sessions=sample_sessions)
        report = analyzer.generate_report()
        
        assert "# Offline Evaluation Analysis Report" in report
        assert "gemini-3-pro" in report
        assert "gemini-3-flash" in report
    
    def test_load_from_directory(self, sample_sessions):
        """Should load sessions from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Save sessions
            for session in sample_sessions:
                session.save(tmpdir / f"{session.session_id}.json")
            
            # Load from directory
            analyzer = OfflineAnalyzer(session_dir=tmpdir)
            
            assert len(analyzer.sessions) == 2
            
            df = analyzer.to_dataframe()
            assert len(df) == 16


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

