"""Unit tests for binary effects module."""

import numpy as np
import pandas as pd
import pytest

from pymeta.effects import binary_effects


class TestBinaryEffects:
    """Test suite for binary effects calculations."""
    
    def test_odds_ratio_basic(self, binary_table_2x2):
        """Test basic odds ratio calculation."""
        data = binary_table_2x2
        
        result = binary_effects(
            data['events_treatment'],
            data['n_treatment'], 
            data['events_control'],
            data['n_control'],
            effect_measure="OR"
        )
        
        # Check return structure
        assert isinstance(result, pd.DataFrame)
        assert 'effect_size' in result.columns
        assert 'variance' in result.columns
        assert 'se' in result.columns
        assert 'measure' in result.columns
        assert all(result['measure'] == 'log_OR')
        
        # Check dimensions
        assert len(result) == len(data['events_treatment'])
        
        # Check that effect sizes are finite
        assert np.all(np.isfinite(result['effect_size']))
        assert np.all(np.isfinite(result['variance']))
        assert np.all(result['variance'] > 0)
        
        # Check SE calculation
        np.testing.assert_allclose(result['se'], np.sqrt(result['variance']))
    
    @pytest.mark.parametrize("continuity_correction", [0.0, 0.5, 1.0])
    def test_continuity_corrections(self, binary_table_with_zeros, continuity_correction):
        """Test different continuity corrections for zero events."""
        data = binary_table_with_zeros
        
        result = binary_effects(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'], 
            data['n_control'],
            effect_measure="OR",
            continuity_correction=continuity_correction
        )
        
        # Should handle zero events without NaN/inf
        assert np.all(np.isfinite(result['effect_size']))
        assert np.all(np.isfinite(result['variance']))
        assert np.all(result['variance'] > 0)
    
    def test_risk_ratio_calculation(self, binary_table_2x2):
        """Test risk ratio calculation."""
        data = binary_table_2x2
        
        result = binary_effects(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control'],
            effect_measure="RR"
        )
        
        assert all(result['measure'] == 'log_RR')
        assert np.all(np.isfinite(result['effect_size']))
        assert np.all(np.isfinite(result['variance']))
        
        # Risk ratio should be different from odds ratio
        or_result = binary_effects(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control'],
            effect_measure="OR"
        )
        
        # Effect sizes should be different (unless very rare events)
        assert not np.allclose(result['effect_size'], or_result['effect_size'])
    
    def test_risk_difference_calculation(self, binary_table_2x2):
        """Test risk difference calculation."""
        data = binary_table_2x2
        
        result = binary_effects(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control'],
            effect_measure="RD"
        )
        
        assert all(result['measure'] == 'RD')
        assert np.all(np.isfinite(result['effect_size']))
        assert np.all(np.isfinite(result['variance']))
        
        # Risk differences should be in reasonable range
        assert np.all(result['effect_size'] >= -1)
        assert np.all(result['effect_size'] <= 1)
    
    def test_manual_calculation_verification(self):
        """Test against manual calculation for a simple case."""
        # Simple case: 10/50 vs 5/50
        events_treat = np.array([10])
        n_treat = np.array([50])
        events_control = np.array([5])
        n_control = np.array([50])
        
        result = binary_effects(events_treat, n_treat, events_control, n_control, 
                               effect_measure="OR", continuity_correction=0)
        
        # Manual calculation
        or_manual = (10 / 40) / (5 / 45)  # (a/c) / (b/d)
        log_or_manual = np.log(or_manual)
        var_manual = 1/10 + 1/40 + 1/5 + 1/45
        
        np.testing.assert_allclose(result['effect_size'][0], log_or_manual, rtol=1e-10)
        np.testing.assert_allclose(result['variance'][0], var_manual, rtol=1e-10)
    
    def test_zero_events_handling(self):
        """Test specific handling of zero events."""
        # Case with zero events in both groups
        events_treat = np.array([0, 10])
        n_treat = np.array([50, 60])
        events_control = np.array([0, 5])
        n_control = np.array([50, 55])
        
        result = binary_effects(events_treat, n_treat, events_control, n_control,
                               effect_measure="OR", continuity_correction=0.5)
        
        # Should apply continuity correction
        assert np.all(np.isfinite(result['effect_size']))
        assert np.all(np.isfinite(result['variance']))
        
        # First study should use continuity correction
        # Second study should not
        # Check that first study used adjusted values
        adj_or_1 = (0.5 / 49.5) / (0.5 / 49.5)  # Should be 1, log_or = 0
        np.testing.assert_allclose(result['effect_size'][0], 0, atol=1e-10)
    
    def test_single_study(self):
        """Test with single study data."""
        result = binary_effects([15], [100], [10], [100], effect_measure="OR")
        
        assert len(result) == 1
        assert np.isfinite(result['effect_size'][0])
        assert np.isfinite(result['variance'][0])
        assert result['variance'][0] > 0
    
    def test_array_inputs(self):
        """Test that function accepts various array-like inputs."""
        # Lists
        result1 = binary_effects([10, 15], [50, 60], [5, 8], [50, 55])
        
        # Numpy arrays
        result2 = binary_effects(
            np.array([10, 15]), 
            np.array([50, 60]),
            np.array([5, 8]), 
            np.array([50, 55])
        )
        
        # Pandas series
        result3 = binary_effects(
            pd.Series([10, 15]),
            pd.Series([50, 60]),
            pd.Series([5, 8]),
            pd.Series([50, 55])
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result1, result3)
    
    def test_invalid_effect_measure(self, binary_table_2x2):
        """Test error handling for invalid effect measures."""
        data = binary_table_2x2
        
        with pytest.raises(ValueError, match="Unknown effect measure"):
            binary_effects(
                data['events_treatment'],
                data['n_treatment'],
                data['events_control'],
                data['n_control'],
                effect_measure="INVALID"
            )
    
    def test_negative_continuity_correction(self, binary_table_2x2):
        """Test with negative continuity correction."""
        data = binary_table_2x2
        
        # Should work but might give different results
        result = binary_effects(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control'],
            continuity_correction=-0.1
        )
        
        # Should still produce finite results for this data
        assert np.all(np.isfinite(result['effect_size']))
    
    def test_extreme_event_rates(self):
        """Test with extreme event rates (very rare or very common)."""
        # Very rare events
        rare_result = binary_effects([1], [1000], [1], [1000], effect_measure="OR")
        assert np.isfinite(rare_result['effect_size'][0])
        
        # Very common events  
        common_result = binary_effects([999], [1000], [998], [1000], effect_measure="OR")
        assert np.isfinite(common_result['effect_size'][0])
    
    @pytest.mark.parametrize("measure", ["OR", "RR", "RD"])
    def test_all_measures_consistency(self, binary_table_2x2, measure):
        """Test that all effect measures produce consistent output structure."""
        data = binary_table_2x2
        
        result = binary_effects(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control'],
            effect_measure=measure
        )
        
        # Check structure consistency
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data['events_treatment'])
        assert 'effect_size' in result.columns
        assert 'variance' in result.columns
        assert 'se' in result.columns
        assert 'measure' in result.columns
        
        # All values should be finite
        assert np.all(np.isfinite(result['effect_size']))
        assert np.all(np.isfinite(result['variance']))
        assert np.all(result['variance'] > 0)
        
        # SE should equal sqrt(variance)
        np.testing.assert_allclose(result['se'], np.sqrt(result['variance']))


class TestBinaryEffectsEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_input(self):
        """Test with empty arrays."""
        result = binary_effects([], [], [], [])
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched array lengths."""
        with pytest.raises((ValueError, IndexError)):
            binary_effects([10, 15], [50], [5, 8], [50, 55])
    
    def test_zero_sample_sizes(self):
        """Test handling of zero sample sizes."""
        # This should raise an error or handle gracefully
        with pytest.raises((ValueError, ZeroDivisionError)):
            binary_effects([5], [0], [3], [10])
    
    def test_events_exceeding_sample_size(self):
        """Test when events exceed sample size."""
        # This is invalid data - should handle gracefully
        result = binary_effects([60], [50], [10], [50])  # 60 > 50
        
        # Depending on implementation, might produce NaN or handle differently
        # At minimum, should not crash
        assert len(result) == 1