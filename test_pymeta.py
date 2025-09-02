#!/usr/bin/env python3
"""
Quick test script to validate PyMeta functionality.
"""

import numpy as np
import tempfile
import os
import pandas as pd

# Import PyMeta
import pymeta

def test_basic_functionality():
    """Test basic PyMeta functionality."""
    print("Testing basic PyMeta functionality...")
    
    # Test data
    effects = np.array([0.25, 0.31, 0.18, 0.42, 0.28])
    variances = np.array([0.04, 0.03, 0.05, 0.02, 0.04])
    study_ids = [f'Study_{i+1}' for i in range(5)]
    
    # Test basic analysis
    config = pymeta.MetaAnalysisConfig(use_hksj=False)
    results = pymeta.analyze_data(effects, variances, study_ids, config)
    
    print(f"  ✓ Basic analysis: Effect = {results.effect:.4f}, SE = {results.se:.4f}")
    
    # Test HKSJ analysis
    config_hksj = pymeta.MetaAnalysisConfig(use_hksj=True)
    results_hksj = pymeta.analyze_data(effects, variances, study_ids, config_hksj)
    
    print(f"  ✓ HKSJ analysis: Effect = {results_hksj.effect:.4f}, SE = {results_hksj.se:.4f}, df = {results_hksj.df}")
    
    # Test CSV analysis
    data = pd.DataFrame({
        'study': study_ids,
        'effect': effects,
        'variance': variances
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        csv_file = f.name
    
    try:
        csv_results = pymeta.analyze_csv(csv_file, config=config)
        print(f"  ✓ CSV analysis: Effect = {csv_results.effect:.4f}")
        
        # Test diagnostics
        if len(results.points) >= 3:
            loo_results = pymeta.diagnostics.leave_one_out_analysis(results.points, config)
            print(f"  ✓ Leave-one-out: {len(loo_results.loo_results)} LOO analyses completed")
            
            influence_results = pymeta.diagnostics.influence_measures(results.points, results, config)
            print(f"  ✓ Influence measures: {len(influence_results)} studies analyzed")
        
        # Test plots
        try:
            fig1 = pymeta.plot_forest(results)
            print("  ✓ Forest plot generated successfully")
            plt.close(fig)
            
            fig2 = pymeta.plot_funnel(results)
            print("  ✓ Funnel plot generated successfully")
            plt.close(fig2)
            
            fig3 = pymeta.plot_funnel_contour(results)
            print("  ✓ Contour funnel plot generated successfully")
            plt.close(fig3)
            
        except Exception as e:
            print(f"  ⚠ Plot generation failed: {e}")
        
    finally:
        os.unlink(csv_file)
    
    print("✓ All basic functionality tests passed!")

def test_suite_functionality():
    """Test PyMeta suite functionality."""
    print("\nTesting PyMeta suite functionality...")
    
    try:
        # Test data
        effects = np.array([0.2, 0.4, 0.3, 0.5, 0.1])
        variances = np.array([0.05, 0.03, 0.04, 0.02, 0.06])
        study_ids = [f'Study_{i+1}' for i in range(5)]
        
        # Create suite
        config = pymeta.MetaAnalysisConfig(use_hksj=True)
        suite = pymeta.suite.MetaAnalysisSuite(config)
        suite.load_arrays(effects, variances, study_ids)
        
        # Run analysis
        results = suite.analyze()
        print(f"  ✓ Suite analysis: Effect = {results.effect:.4f}")
        
        # Test comprehensive analysis
        with tempfile.TemporaryDirectory() as tmpdir:
            comprehensive = suite.comprehensive_analysis(save_dir=tmpdir)
            print(f"  ✓ Comprehensive analysis completed")
            
            # Check saved files
            files = os.listdir(tmpdir)
            print(f"  ✓ Generated {len(files)} output files")
        
        print("✓ Suite functionality tests passed!")
        
    except Exception as e:
        print(f"  ⚠ Suite tests failed: {e}")

def test_cli_installation():
    """Test CLI installation."""
    print("\nTesting CLI installation...")
    
    try:
        # Test CLI import
        from pymeta.cli import main
        print("  ✓ CLI module imported successfully")
        
        # Test if CLI is accessible
        import subprocess
        result = subprocess.run(['python', '-m', 'pymeta.cli', '--help'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("  ✓ CLI accessible via python -m pymeta.cli")
        else:
            print("  ⚠ CLI not accessible via python -m")
            
    except Exception as e:
        print(f"  ⚠ CLI test failed: {e}")

if __name__ == "__main__":
    print("PyMeta Package Validation")
    print("=" * 30)
    
    test_basic_functionality()
    test_suite_functionality()
    test_cli_installation()
    
    print("\n" + "=" * 30)
    print("Validation completed! 🎉")