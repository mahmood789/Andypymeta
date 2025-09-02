"""Tests for binary effects calculations."""

import pytest
import numpy as np
from pymeta.effects.binary import (
    calculate_log_odds_ratio, calculate_risk_ratio, calculate_risk_difference,
    calculate_from_2x2_tables, odds_ratio_to_cohens_d, cohens_d_to_odds_ratio
)
from pymeta.errors import DataError


class TestBinaryEffects:
    """Test binary effect size calculations."""
    
    def test_log_odds_ratio_basic(self):
        """Test basic log odds ratio calculation."""
        # Simple 2x2 table
        a, b, c, d = 10, 5, 20, 15
        log_or, variance = calculate_log_odds_ratio(a, b, c, d)
        
        # Expected: log((10*15)/(5*20)) = log(1.5) ≈ 0.405
        expected_log_or = np.log((a * d) / (b * c))
        expected_variance = (1/a) + (1/b) + (1/c) + (1/d)
        
        assert abs(log_or - expected_log_or) < 1e-10
        assert abs(variance - expected_variance) < 1e-10
    
    def test_log_odds_ratio_with_zeros(self):
        """Test log odds ratio with zero cells."""
        # Table with zero cell
        a, b, c, d = 0, 5, 20, 15
        log_or, variance = calculate_log_odds_ratio(a, b, c, d, continuity_correction=0.5)
        
        # Should apply continuity correction
        assert log_or is not None
        assert variance is not None
        assert np.isfinite(log_or)
        assert np.isfinite(variance)
    
    def test_log_odds_ratio_invalid_input(self):
        """Test log odds ratio with invalid input."""
        with pytest.raises(DataError):
            calculate_log_odds_ratio(-1, 5, 20, 15)
    
    def test_risk_ratio_basic(self):
        """Test basic risk ratio calculation."""
        a, b, c, d = 10, 5, 20, 15
        log_rr, variance = calculate_risk_ratio(a, b, c, d)
        
        # Manual calculation
        n1, n2 = a + c, b + d
        risk_treatment = a / n1
        risk_control = b / n2
        expected_log_rr = np.log(risk_treatment / risk_control)
        expected_variance = (c / (a * n1)) + (d / (b * n2))
        
        assert abs(log_rr - expected_log_rr) < 1e-10
        assert abs(variance - expected_variance) < 1e-10
    
    def test_risk_difference_basic(self):
        """Test basic risk difference calculation."""
        a, b, c, d = 10, 5, 20, 15
        rd, variance = calculate_risk_difference(a, b, c, d)
        
        # Manual calculation
        n1, n2 = a + c, b + d
        risk_treatment = a / n1
        risk_control = b / n2
        expected_rd = risk_treatment - risk_control
        expected_variance = (risk_treatment * (1 - risk_treatment) / n1) + \
                           (risk_control * (1 - risk_control) / n2)
        
        assert abs(rd - expected_rd) < 1e-10
        assert abs(variance - expected_variance) < 1e-10
    
    def test_multiple_2x2_tables(self):
        """Test calculation from multiple 2x2 tables."""
        tables = [
            np.array([[10, 5], [20, 15]]),
            np.array([[8, 12], [15, 25]]),
            np.array([[15, 8], [10, 20]])
        ]
        
        points = calculate_from_2x2_tables(tables, effect_type="log_or")
        
        assert len(points) == 3
        for point in points:
            assert hasattr(point, 'effect')
            assert hasattr(point, 'variance')
            assert np.isfinite(point.effect)
            assert point.variance > 0
    
    def test_odds_ratio_to_cohens_d(self):
        """Test conversion from odds ratio to Cohen's d."""
        log_or = 0.5
        log_or_variance = 0.1
        
        d, d_variance = odds_ratio_to_cohens_d(log_or, log_or_variance)
        
        # Check conversion factor
        conversion_factor = np.sqrt(3) / np.pi
        expected_d = log_or * conversion_factor
        expected_variance = log_or_variance * (conversion_factor ** 2)
        
        assert abs(d - expected_d) < 1e-10
        assert abs(d_variance - expected_variance) < 1e-10
    
    def test_cohens_d_to_odds_ratio(self):
        """Test conversion from Cohen's d to odds ratio."""
        d = 0.3
        d_variance = 0.05
        
        log_or, log_or_variance = cohens_d_to_odds_ratio(d, d_variance)
        
        # Check conversion factor
        conversion_factor = np.pi / np.sqrt(3)
        expected_log_or = d * conversion_factor
        expected_variance = d_variance * (conversion_factor ** 2)
        
        assert abs(log_or - expected_log_or) < 1e-10
        assert abs(log_or_variance - expected_variance) < 1e-10
    
    def test_round_trip_conversion(self):
        """Test round-trip conversion OR -> d -> OR."""
        original_log_or = 0.4
        original_variance = 0.08
        
        # Convert to Cohen's d
        d, d_variance = odds_ratio_to_cohens_d(original_log_or, original_variance)
        
        # Convert back to log OR
        recovered_log_or, recovered_variance = cohens_d_to_odds_ratio(d, d_variance)
        
        assert abs(recovered_log_or - original_log_or) < 1e-10
        assert abs(recovered_variance - original_variance) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__])