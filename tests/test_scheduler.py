"""
Test suite for living meta-analysis scheduler functionality.
"""

import pytest
import time
import tempfile
import os
import pandas as pd
from unittest.mock import patch, MagicMock

from pymeta.living.scheduler import (
    SchedulerConfig,
    LiveMetaAnalysis,
    start_living_analysis
)
from pymeta import MetaAnalysisConfig


class TestScheduler:
    """Test cases for living meta-analysis scheduler."""
    
    def create_test_data_file(self):
        """Create a test CSV file for scheduler testing."""
        data = pd.DataFrame({
            'study': ['Study_1', 'Study_2', 'Study_3', 'Study_4'],
            'effect': [0.3, 0.5, 0.4, 0.6],
            'variance': [0.1, 0.08, 0.12, 0.09]
        })
        
        fd, filepath = tempfile.mkstemp(suffix='.csv')
        try:
            data.to_csv(filepath, index=False)
            return filepath
        finally:
            os.close(fd)
    
    def test_scheduler_config_creation(self):
        """Test SchedulerConfig creation and validation."""
        data_file = self.create_test_data_file()
        
        try:
            config = SchedulerConfig(
                data_source=data_file,
                update_interval_seconds=3600,
                meta_config=MetaAnalysisConfig(use_hksj=True)
            )
            
            assert config.data_source == data_file
            assert config.update_interval_seconds == 3600
            assert config.meta_config.use_hksj is True
            assert config.max_retries == 3  # default
            
        finally:
            os.unlink(data_file)
    
    def test_scheduler_config_validation(self):
        """Test SchedulerConfig input validation."""
        data_file = self.create_test_data_file()
        
        try:
            # Test invalid update interval
            with pytest.raises(ValueError, match="at least 60 seconds"):
                SchedulerConfig(
                    data_source=data_file,
                    update_interval_seconds=30  # Too short
                )
            
            # Test invalid max_retries
            with pytest.raises(ValueError, match="non-negative"):
                SchedulerConfig(
                    data_source=data_file,
                    max_retries=-1
                )
            
            # Test invalid retry_delay
            with pytest.raises(ValueError, match="non-negative"):
                SchedulerConfig(
                    data_source=data_file,
                    retry_delay_seconds=-1
                )
                
        finally:
            os.unlink(data_file)
    
    def test_live_meta_analysis_creation(self):
        """Test LiveMetaAnalysis object creation."""
        data_file = self.create_test_data_file()
        
        try:
            config = SchedulerConfig(
                data_source=data_file,
                update_interval_seconds=3600
            )
            
            live_analysis = LiveMetaAnalysis(config)
            
            assert live_analysis.config == config
            assert live_analysis.is_running is False
            assert live_analysis.last_result is None
            assert live_analysis.update_count == 0
            
        finally:
            os.unlink(data_file)
    
    def test_apscheduler_detection(self):
        """Test APScheduler availability detection."""
        data_file = self.create_test_data_file()
        
        try:
            config = SchedulerConfig(data_source=data_file)
            live_analysis = LiveMetaAnalysis(config)
            
            # Should detect whether APScheduler is available
            assert isinstance(live_analysis.use_apscheduler, bool)
            
        finally:
            os.unlink(data_file)
    
    def test_single_analysis_run(self):
        """Test running a single meta-analysis update."""
        data_file = self.create_test_data_file()
        
        try:
            config = SchedulerConfig(
                data_source=data_file,
                update_interval_seconds=3600
            )
            
            live_analysis = LiveMetaAnalysis(config)
            
            # Run single analysis
            result = live_analysis._run_analysis()
            
            assert result is not None
            assert live_analysis.update_count == 1
            assert live_analysis.last_result == result
            assert result.effect is not None
            
        finally:
            os.unlink(data_file)
    
    def test_analysis_with_retry(self):
        """Test analysis with retry mechanism."""
        data_file = self.create_test_data_file()
        
        try:
            config = SchedulerConfig(
                data_source=data_file,
                max_retries=2,
                retry_delay_seconds=1  # Short delay for testing
            )
            
            live_analysis = LiveMetaAnalysis(config)
            
            # This should succeed on first try
            result = live_analysis._run_analysis_with_retry()
            
            assert result is not None
            assert live_analysis.update_count == 1
            
        finally:
            os.unlink(data_file)
    
    def test_analysis_failure_handling(self):
        """Test handling of analysis failures."""
        # Use non-existent file to trigger failure
        config = SchedulerConfig(
            data_source="nonexistent_file.csv",
            max_retries=1,
            retry_delay_seconds=1
        )
        
        live_analysis = LiveMetaAnalysis(config)
        
        # This should fail and return None
        result = live_analysis._run_analysis_with_retry()
        
        assert result is None
        assert live_analysis.update_count == 0
    
    def test_output_directory_creation(self):
        """Test output directory creation."""
        data_file = self.create_test_data_file()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = os.path.join(tmpdir, "output")
                
                config = SchedulerConfig(
                    data_source=data_file,
                    output_dir=output_dir,
                    save_results=True
                )
                
                live_analysis = LiveMetaAnalysis(config)
                
                # Output directory should be created
                assert os.path.exists(output_dir)
                
        finally:
            os.unlink(data_file)
    
    @patch('pymeta.living.scheduler.logger')
    def test_save_results(self, mock_logger):
        """Test saving results to file."""
        data_file = self.create_test_data_file()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = SchedulerConfig(
                    data_source=data_file,
                    output_dir=tmpdir,
                    save_results=True
                )
                
                live_analysis = LiveMetaAnalysis(config)
                result = live_analysis._run_analysis()
                
                # Check that result file was created
                files = os.listdir(tmpdir)
                result_files = [f for f in files if f.startswith('meta_results_')]
                assert len(result_files) > 0
                
                # Check file content
                result_file = os.path.join(tmpdir, result_files[0])
                with open(result_file, 'r') as f:
                    content = f.read()
                    assert "Meta-Analysis Results" in content
                    assert "Effect:" in content
                    
        finally:
            os.unlink(data_file)
    
    def test_status_reporting(self):
        """Test status reporting functionality."""
        data_file = self.create_test_data_file()
        
        try:
            config = SchedulerConfig(
                data_source=data_file,
                update_interval_seconds=3600
            )
            
            live_analysis = LiveMetaAnalysis(config)
            
            # Get initial status
            status = live_analysis.get_status()
            
            assert isinstance(status, dict)
            assert 'is_running' in status
            assert 'update_count' in status
            assert 'last_effect' in status
            assert 'use_apscheduler' in status
            assert 'config' in status
            
            assert status['is_running'] is False
            assert status['update_count'] == 0
            assert status['last_effect'] is None
            
            # Run analysis and check status
            live_analysis._run_analysis()
            status = live_analysis.get_status()
            
            assert status['update_count'] == 1
            assert status['last_effect'] is not None
            
        finally:
            os.unlink(data_file)
    
    def test_start_living_analysis_convenience(self):
        """Test convenience function for starting living analysis."""
        data_file = self.create_test_data_file()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                live_analysis = start_living_analysis(
                    data_source=data_file,
                    update_interval_seconds=3600,
                    use_hksj=True,
                    output_dir=tmpdir
                )
                
                assert isinstance(live_analysis, LiveMetaAnalysis)
                assert live_analysis.config.meta_config.use_hksj is True
                assert live_analysis.config.output_dir == tmpdir
                
                # Stop the analysis
                live_analysis.stop()
                
        finally:
            os.unlink(data_file)
    
    def test_scheduler_start_stop_simple(self):
        """Test starting and stopping scheduler with simple polling."""
        data_file = self.create_test_data_file()
        
        try:
            config = SchedulerConfig(
                data_source=data_file,
                update_interval_seconds=60  # 1 minute
            )
            
            live_analysis = LiveMetaAnalysis(config)
            
            # Force simple polling (no APScheduler)
            live_analysis.use_apscheduler = False
            
            # Start scheduler
            live_analysis.start()
            assert live_analysis.is_running is True
            
            # Wait a short time
            time.sleep(1)
            
            # Stop scheduler
            live_analysis.stop()
            assert live_analysis.is_running is False
            
        finally:
            os.unlink(data_file)
    
    @patch('pymeta.living.scheduler.time.sleep')
    def test_change_notification(self, mock_sleep):
        """Test change notification functionality."""
        data_file = self.create_test_data_file()
        
        try:
            config = SchedulerConfig(
                data_source=data_file,
                notify_on_change=True,
                change_threshold=0.1
            )
            
            live_analysis = LiveMetaAnalysis(config)
            
            # Run first analysis
            result1 = live_analysis._run_analysis()
            
            # Modify the result to simulate a change
            result2 = result1
            result2.effect = result1.effect + 0.2  # Significant change
            
            # Mock the notification method
            with patch.object(live_analysis, '_notify_change') as mock_notify:
                live_analysis.last_result = result1
                live_analysis._run_analysis = MagicMock(return_value=result2)
                
                # Run analysis again (should trigger notification)
                live_analysis._run_analysis_with_retry()
                
                # Notification should have been called
                mock_notify.assert_called_once()
                
        finally:
            os.unlink(data_file)
    
    def test_hksj_configuration_propagation(self):
        """Test that HKSJ configuration is properly propagated."""
        data_file = self.create_test_data_file()
        
        try:
            config = SchedulerConfig(
                data_source=data_file,
                meta_config=MetaAnalysisConfig(use_hksj=True)
            )
            
            live_analysis = LiveMetaAnalysis(config)
            result = live_analysis._run_analysis()
            
            assert result.use_hksj is True
            assert "HKSJ" in result.method
            
        finally:
            os.unlink(data_file)