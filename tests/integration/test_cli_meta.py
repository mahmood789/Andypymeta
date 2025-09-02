"""Integration tests for CLI interface."""

import subprocess
import tempfile
import csv
from pathlib import Path
import pytest
import pandas as pd
import numpy as np


@pytest.mark.integration
class TestCLIMetaAnalysis:
    """Test suite for command-line meta-analysis interface."""
    
    def test_cli_version_command(self):
        """Test version command."""
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'version'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 0
        assert 'pymeta version' in result.stdout
        assert '0.0.1' in result.stdout
    
    def test_cli_help_command(self):
        """Test help command."""
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', '--help'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 0
        assert 'pymeta: Meta-analysis toolkit' in result.stdout
        assert 'meta' in result.stdout
        assert 'version' in result.stdout
    
    def test_cli_meta_help(self):
        """Test meta analysis command help."""
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta', '--help'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 0
        assert '--input' in result.stdout
        assert '--method' in result.stdout
        assert '--output' in result.stdout
    
    def test_cli_meta_basic_analysis(self, test_output_dir):
        """Test basic meta-analysis through CLI."""
        # Create test data file
        test_data = pd.DataFrame({
            'study_id': ['Study_A', 'Study_B', 'Study_C', 'Study_D'],
            'effect_size': [0.3, 0.5, 0.4, 0.6],
            'standard_error': [0.1, 0.12, 0.11, 0.13],
            'sample_size': [100, 120, 110, 130]
        })
        
        input_file = test_output_dir / 'test_data.csv'
        test_data.to_csv(input_file, index=False)
        
        # Run meta-analysis
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta', 
             '--input', str(input_file),
             '--method', 'RE'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 0
        
        # Check output contains expected results
        output = result.stdout
        assert 'Meta-analysis Results' in output
        assert 'Random Effects' in output
        assert 'Pooled effect size:' in output
        assert 'Standard error:' in output
        assert '95% CI:' in output
        assert 'P-value:' in output
        assert 'Heterogeneity:' in output
    
    def test_cli_meta_fixed_effects(self, test_output_dir):
        """Test fixed effects meta-analysis through CLI."""
        # Create test data
        test_data = pd.DataFrame({
            'effect_size': [0.2, 0.4, 0.3, 0.5, 0.35],
            'standard_error': [0.08, 0.10, 0.09, 0.11, 0.095]
        })
        
        input_file = test_output_dir / 'fe_test_data.csv'
        test_data.to_csv(input_file, index=False)
        
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file),
             '--method', 'FE'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 0
        assert 'Fixed Effects' in result.stdout
        assert 'Pooled effect size:' in result.stdout
    
    def test_cli_meta_with_output_file(self, test_output_dir):
        """Test meta-analysis with output file."""
        # Create test data
        test_data = pd.DataFrame({
            'effect_size': [0.3, 0.5, 0.4],
            'standard_error': [0.1, 0.12, 0.11]
        })
        
        input_file = test_output_dir / 'input_data.csv'
        output_file = test_output_dir / 'results.csv'
        test_data.to_csv(input_file, index=False)
        
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file),
             '--output', str(output_file),
             '--method', 'RE'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 0
        assert 'Results saved to' in result.stdout
        
        # Check output file was created
        assert output_file.exists()
        
        # Check output file contains results
        results_df = pd.read_csv(output_file)
        assert 'pooled_effect' in results_df.columns
        assert 'se' in results_df.columns
        assert 'method' in results_df.columns
    
    def test_cli_meta_with_plot(self, test_output_dir):
        """Test meta-analysis with plot generation."""
        # Create test data with study IDs
        test_data = pd.DataFrame({
            'study_id': ['RCT_001', 'RCT_002', 'RCT_003'],
            'effect_size': [0.3, 0.5, 0.4],
            'standard_error': [0.1, 0.12, 0.11]
        })
        
        input_file = test_output_dir / 'plot_test_data.csv'
        test_data.to_csv(input_file, index=False)
        
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file),
             '--method', 'RE',
             '--plot'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 0
        assert 'Forest plot saved to' in result.stdout
        
        # Check plot file was created
        plot_file = test_output_dir / 'plot_test_data_forest.png'
        # Note: The plot is saved in the input file directory
        expected_plot_file = input_file.parent / f'{input_file.stem}_forest.png'
        assert expected_plot_file.exists()
    
    def test_cli_meta_custom_columns(self, test_output_dir):
        """Test meta-analysis with custom column names."""
        # Create test data with different column names
        test_data = pd.DataFrame({
            'study': ['A', 'B', 'C'],
            'effect': [0.3, 0.5, 0.4],
            'se': [0.1, 0.12, 0.11],
            'n': [100, 120, 110]
        })
        
        input_file = test_output_dir / 'custom_cols.csv'
        test_data.to_csv(input_file, index=False)
        
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file),
             '--effect-col', 'effect',
             '--se-col', 'se',
             '--method', 'FE'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 0
        assert 'Pooled effect size:' in result.stdout
    
    def test_cli_meta_missing_file(self):
        """Test error handling for missing input file."""
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', 'nonexistent_file.csv'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 1
        assert 'Error:' in result.stderr
    
    def test_cli_meta_invalid_method(self, test_output_dir):
        """Test error handling for invalid method."""
        # Create minimal test data
        test_data = pd.DataFrame({
            'effect_size': [0.3],
            'standard_error': [0.1]
        })
        
        input_file = test_output_dir / 'test.csv'
        test_data.to_csv(input_file, index=False)
        
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file),
             '--method', 'INVALID'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 2  # argparse error
    
    def test_cli_meta_missing_columns(self, test_output_dir):
        """Test error handling for missing required columns."""
        # Create data missing required columns
        test_data = pd.DataFrame({
            'study': ['A', 'B'],
            'effect': [0.3, 0.5]
            # Missing standard_error column
        })
        
        input_file = test_output_dir / 'missing_cols.csv'
        test_data.to_csv(input_file, index=False)
        
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file)],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 1
        assert 'Error:' in result.stderr


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_cli_module_import(self):
        """Test that CLI module can be imported and run."""
        result = subprocess.run(
            ['python', '-c', 'from pymeta.cli import main; print("Import successful")'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 0
        assert 'Import successful' in result.stdout
    
    def test_cli_as_script(self):
        """Test running CLI as script."""
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        # Should show help when no command given
        assert result.returncode == 0
        assert 'pymeta: Meta-analysis toolkit' in result.stdout
    
    def test_cli_data_pipeline(self, test_output_dir):
        """Test complete data analysis pipeline through CLI."""
        np.random.seed(42)
        
        # Create realistic meta-analysis dataset
        n_studies = 8
        study_ids = [f'RCT_{i:03d}' for i in range(1, n_studies + 1)]
        true_effects = np.random.normal(0.4, 0.15, n_studies)
        sample_sizes = np.random.randint(50, 200, n_studies)
        standard_errors = 2 / np.sqrt(sample_sizes)
        effect_sizes = np.random.normal(true_effects, standard_errors)
        
        test_data = pd.DataFrame({
            'study_id': study_ids,
            'effect_size': effect_sizes,
            'standard_error': standard_errors,
            'sample_size': sample_sizes,
            'year': np.random.randint(2010, 2024, n_studies),
            'country': np.random.choice(['USA', 'UK', 'Canada'], n_studies)
        })
        
        input_file = test_output_dir / 'pipeline_data.csv'
        output_file = test_output_dir / 'pipeline_results.csv'
        test_data.to_csv(input_file, index=False)
        
        # Run analysis
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file),
             '--output', str(output_file),
             '--method', 'RE',
             '--plot'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 0
        
        # Check all outputs were created
        assert output_file.exists()
        
        # Verify results file content
        results = pd.read_csv(output_file)
        assert len(results) == 1  # One row of results
        assert 'pooled_effect' in results.columns
        assert 'ci_lower' in results.columns
        assert 'ci_upper' in results.columns
        assert 'tau2' in results.columns
        
        # Check that results are reasonable
        assert np.isfinite(results['pooled_effect'].iloc[0])
        assert results['ci_lower'].iloc[0] < results['ci_upper'].iloc[0]
    
    def test_cli_error_propagation(self, test_output_dir):
        """Test that errors are properly propagated from analysis."""
        # Create data that might cause analysis issues
        test_data = pd.DataFrame({
            'effect_size': [np.nan, 0.5, 0.4],
            'standard_error': [0.1, 0.0, 0.11]  # Zero SE might cause issues
        })
        
        input_file = test_output_dir / 'problematic_data.csv'
        test_data.to_csv(input_file, index=False)
        
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file)],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        # Might succeed with warnings or fail gracefully
        # Either way, should not crash without informative output
        if result.returncode != 0:
            assert 'Error:' in result.stderr
        else:
            # If it succeeds, should handle the problematic data
            assert 'Meta-analysis Results' in result.stdout
    
    def test_cli_large_dataset(self, test_output_dir):
        """Test CLI with larger dataset."""
        np.random.seed(123)
        n_studies = 50  # Larger dataset
        
        effect_sizes = np.random.normal(0.3, 0.2, n_studies)
        standard_errors = np.random.uniform(0.05, 0.15, n_studies)
        
        test_data = pd.DataFrame({
            'study_id': [f'Study_{i:03d}' for i in range(n_studies)],
            'effect_size': effect_sizes,
            'standard_error': standard_errors
        })
        
        input_file = test_output_dir / 'large_dataset.csv'
        test_data.to_csv(input_file, index=False)
        
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file),
             '--method', 'RE'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta',
            timeout=30  # Should complete reasonably quickly
        )
        
        assert result.returncode == 0
        assert 'Meta-analysis Results' in result.stdout
        assert 'τ²' in result.stdout  # Should report tau-squared


@pytest.mark.integration
class TestCLIEdgeCases:
    """Test edge cases for CLI interface."""
    
    def test_cli_single_study(self, test_output_dir):
        """Test CLI with single study."""
        test_data = pd.DataFrame({
            'effect_size': [0.5],
            'standard_error': [0.1]
        })
        
        input_file = test_output_dir / 'single_study.csv'
        test_data.to_csv(input_file, index=False)
        
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file)],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 0
        # Should handle single study gracefully
        assert 'Pooled effect size:' in result.stdout
    
    def test_cli_empty_file(self, test_output_dir):
        """Test CLI with empty data file."""
        # Create empty CSV with headers only
        empty_data = pd.DataFrame(columns=['effect_size', 'standard_error'])
        
        input_file = test_output_dir / 'empty_data.csv'
        empty_data.to_csv(input_file, index=False)
        
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file)],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        
        assert result.returncode == 1
        assert 'Error:' in result.stderr
    
    def test_cli_permission_denied(self, test_output_dir):
        """Test CLI behavior with permission issues."""
        # Create test data
        test_data = pd.DataFrame({
            'effect_size': [0.3, 0.5],
            'standard_error': [0.1, 0.12]
        })
        
        input_file = test_output_dir / 'permission_test.csv'
        test_data.to_csv(input_file, index=False)
        
        # Try to write to a directory without write permission
        # This test might be platform-specific
        try:
            result = subprocess.run(
                ['python', '-m', 'pymeta.cli', 'meta',
                 '--input', str(input_file),
                 '--output', '/root/forbidden.csv'],  # Typically no permission
                capture_output=True,
                text=True,
                cwd='/home/runner/work/Andypymeta/Andypymeta',
                timeout=10
            )
            
            # Should handle permission error gracefully
            if result.returncode != 0:
                assert 'Error:' in result.stderr
        except subprocess.TimeoutExpired:
            # If it hangs, that's also a test failure
            pytest.fail("CLI command timed out")
    
    def test_cli_interrupt_handling(self, test_output_dir):
        """Test CLI handling of interruption (basic test)."""
        # Create test data
        test_data = pd.DataFrame({
            'effect_size': [0.3, 0.5],
            'standard_error': [0.1, 0.12]
        })
        
        input_file = test_output_dir / 'interrupt_test.csv'
        test_data.to_csv(input_file, index=False)
        
        # Run a normal command - this is more of a smoke test
        # since interruption testing is complex in subprocess
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file)],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta',
            timeout=5  # Short timeout
        )
        
        # Should complete normally within timeout
        assert result.returncode == 0


@pytest.mark.integration
class TestCLIPerformance:
    """Test CLI performance characteristics."""
    
    def test_cli_startup_time(self):
        """Test CLI startup time."""
        import time
        
        start_time = time.time()
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'version'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta'
        )
        end_time = time.time()
        
        assert result.returncode == 0
        
        # Should start up reasonably quickly
        startup_time = end_time - start_time
        assert startup_time < 5.0  # Less than 5 seconds
    
    def test_cli_memory_usage(self, test_output_dir):
        """Test CLI memory usage (basic smoke test)."""
        # Create moderate-sized dataset
        np.random.seed(456)
        n_studies = 100
        
        test_data = pd.DataFrame({
            'effect_size': np.random.normal(0.3, 0.2, n_studies),
            'standard_error': np.random.uniform(0.05, 0.15, n_studies)
        })
        
        input_file = test_output_dir / 'memory_test.csv'
        test_data.to_csv(input_file, index=False)
        
        # Run analysis - should not crash due to memory issues
        result = subprocess.run(
            ['python', '-m', 'pymeta.cli', 'meta',
             '--input', str(input_file),
             '--method', 'RE'],
            capture_output=True,
            text=True,
            cwd='/home/runner/work/Andypymeta/Andypymeta',
            timeout=30
        )
        
        assert result.returncode == 0
        assert 'Meta-analysis Results' in result.stdout