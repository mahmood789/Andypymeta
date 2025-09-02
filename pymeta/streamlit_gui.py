"""
Streamlit GUI interface for PyMeta package.

Provides an interactive web interface for meta-analysis with:
- File upload and data preview
- HKSJ configuration controls
- Interactive plotting
- Results visualization
- Download capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

from . import (
    MetaAnalysisConfig,
    MetaPoint,
    analyze_data,
    analyze_csv
)
from .plots import plot_forest, plot_funnel, plot_funnel_contour
from .diagnostics import leave_one_out_analysis, influence_measures


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PyMeta - Meta-Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 PyMeta - Comprehensive Meta-Analysis Tool")
    st.markdown("*Advanced meta-analysis with HKSJ variance adjustment and diagnostics*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model = st.selectbox(
            "Model Type",
            options=["RE", "FE"],
            index=0,
            help="Random Effects (RE) or Fixed Effects (FE)"
        )
        
        # Tau² method (only for RE)
        tau2_method = "REML"
        if model == "RE":
            tau2_method = st.selectbox(
                "Tau² Estimation Method",
                options=["REML", "DL", "PM", "ML"],
                index=0,
                help="Method for estimating between-study variance"
            )
        
        # HKSJ setting
        use_hksj = st.checkbox(
            "Use HKSJ Variance Adjustment",
            value=False,
            help="Apply Hartung-Knapp-Sidik-Jonkman correction with t-distribution"
        )
        
        # Alpha level
        alpha = st.slider(
            "Significance Level (α)",
            min_value=0.001,
            max_value=0.1,
            value=0.05,
            step=0.001,
            format="%.3f"
        )
        
        # Create config
        config = MetaAnalysisConfig(
            model=model,
            tau2_method=tau2_method,
            use_hksj=use_hksj,
            alpha=alpha
        )
        
        st.divider()
        st.markdown("### About HKSJ")
        st.markdown("""
        The Hartung-Knapp-Sidik-Jonkman (HKSJ) method:
        - Uses t-distribution instead of normal
        - Adjusts variance for uncertainty in τ²
        - Generally provides more conservative CIs
        - Recommended for small number of studies
        """)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data & Analysis", "📊 Plots", "🔍 Diagnostics", "📝 Export"])
    
    with tab1:
        _data_analysis_tab(config)
    
    with tab2:
        _plots_tab(config)
    
    with tab3:
        _diagnostics_tab(config)
    
    with tab4:
        _export_tab()


def _data_analysis_tab(config: MetaAnalysisConfig):
    """Data input and analysis tab."""
    st.header("Data Input & Analysis")
    
    # Data input method selection
    input_method = st.radio(
        "Choose data input method:",
        options=["Upload CSV File", "Manual Entry", "Example Data"],
        horizontal=True
    )
    
    data = None
    results = None
    
    if input_method == "Upload CSV File":
        data, results = _handle_csv_upload(config)
    elif input_method == "Manual Entry":
        data, results = _handle_manual_entry(config)
    elif input_method == "Example Data":
        data, results = _handle_example_data(config)
    
    # Store in session state for other tabs
    if data is not None:
        st.session_state['data'] = data
    if results is not None:
        st.session_state['results'] = results
        st.session_state['config'] = config


def _handle_csv_upload(config: MetaAnalysisConfig):
    """Handle CSV file upload."""
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV file should contain columns for effect sizes, variances, and study IDs"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            data = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(data)
            
            # Column mapping
            col1, col2, col3 = st.columns(3)
            
            with col1:
                effect_col = st.selectbox(
                    "Effect Size Column",
                    options=data.columns.tolist(),
                    index=0 if "effect" in data.columns else 0
                )
            
            with col2:
                variance_col = st.selectbox(
                    "Variance Column",
                    options=data.columns.tolist(),
                    index=1 if len(data.columns) > 1 else 0
                )
            
            with col3:
                study_col = st.selectbox(
                    "Study ID Column",
                    options=data.columns.tolist(),
                    index=2 if len(data.columns) > 2 else 0
                )
            
            # Validate data
            if st.button("Run Analysis", type="primary"):
                try:
                    # Create temporary file for analyze_csv
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                        data.to_csv(tmp.name, index=False)
                        
                        results = analyze_csv(
                            filepath=tmp.name,
                            effect_col=effect_col,
                            variance_col=variance_col,
                            study_col=study_col,
                            config=config
                        )
                    
                    _display_results(results)
                    
                    return data, results
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    return data, None
                    
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return None, None
    
    return None, None


def _handle_manual_entry(config: MetaAnalysisConfig):
    """Handle manual data entry."""
    st.subheader("Manual Data Entry")
    
    # Number of studies
    n_studies = st.number_input(
        "Number of studies",
        min_value=2,
        max_value=50,
        value=5,
        step=1
    )
    
    # Create input form
    with st.form("manual_data_form"):
        st.write("Enter data for each study:")
        
        data = []
        for i in range(n_studies):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                study_id = st.text_input(f"Study {i+1} ID", value=f"Study_{i+1}")
            
            with col2:
                effect = st.number_input(f"Effect Size", value=0.0, key=f"effect_{i}")
            
            with col3:
                variance = st.number_input(f"Variance", value=1.0, min_value=0.001, key=f"variance_{i}")
            
            data.append({
                'study': study_id,
                'effect': effect,
                'variance': variance
            })
        
        if st.form_submit_button("Run Analysis", type="primary"):
            try:
                df = pd.DataFrame(data)
                
                # Create MetaPoint objects
                points = [
                    MetaPoint(
                        effect=row['effect'],
                        variance=row['variance'],
                        study_id=row['study']
                    )
                    for _, row in df.iterrows()
                ]
                
                # Run analysis
                effects = np.array([p.effect for p in points])
                variances = np.array([p.variance for p in points])
                study_ids = [p.study_id for p in points]
                
                results = analyze_data(effects, variances, study_ids, config)
                
                _display_results(results)
                
                return df, results
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                return None, None
    
    return None, None


def _handle_example_data(config: MetaAnalysisConfig):
    """Handle example data selection."""
    st.subheader("Example Datasets")
    
    example_choice = st.selectbox(
        "Choose example dataset:",
        options=[
            "Education Intervention",
            "Medical Treatment",
            "Psychology Study",
            "High Heterogeneity Example"
        ]
    )
    
    # Generate example data based on choice
    if example_choice == "Education Intervention":
        data = pd.DataFrame({
            'study': [f'School_{i+1}' for i in range(8)],
            'effect': [0.25, 0.31, 0.18, 0.42, 0.28, 0.15, 0.35, 0.22],
            'variance': [0.04, 0.03, 0.05, 0.02, 0.04, 0.06, 0.03, 0.05]
        })
    elif example_choice == "Medical Treatment":
        data = pd.DataFrame({
            'study': [f'Trial_{i+1}' for i in range(6)],
            'effect': [0.65, 0.82, 0.58, 0.75, 0.69, 0.71],
            'variance': [0.08, 0.06, 0.09, 0.07, 0.08, 0.07]
        })
    elif example_choice == "Psychology Study":
        data = pd.DataFrame({
            'study': [f'Lab_{i+1}' for i in range(10)],
            'effect': [0.12, 0.28, 0.15, 0.33, 0.08, 0.21, 0.19, 0.25, 0.17, 0.29],
            'variance': [0.02, 0.03, 0.04, 0.02, 0.05, 0.03, 0.03, 0.02, 0.04, 0.03]
        })
    else:  # High Heterogeneity Example
        data = pd.DataFrame({
            'study': [f'Study_{i+1}' for i in range(7)],
            'effect': [-0.15, 0.45, 0.12, 0.78, -0.08, 0.52, 0.23],
            'variance': [0.06, 0.04, 0.08, 0.03, 0.07, 0.05, 0.06]
        })
    
    st.dataframe(data)
    
    if st.button("Run Analysis with Example Data", type="primary"):
        try:
            # Create MetaPoint objects
            points = [
                MetaPoint(
                    effect=row['effect'],
                    variance=row['variance'],
                    study_id=row['study']
                )
                for _, row in data.iterrows()
            ]
            
            # Run analysis
            effects = np.array([p.effect for p in points])
            variances = np.array([p.variance for p in points])
            study_ids = [p.study_id for p in points]
            
            results = analyze_data(effects, variances, study_ids, config)
            
            _display_results(results)
            
            return data, results
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None, None
    
    return None, None


def _display_results(results):
    """Display meta-analysis results."""
    st.subheader("📊 Meta-Analysis Results")
    
    # Main results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Pooled Effect",
            value=f"{results.effect:.4f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="95% CI Width",
            value=f"{results.ci_width:.4f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="P-value",
            value=f"{results.p_value:.4f}",
            delta=None
        )
    
    # Detailed results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Effect Size & Uncertainty**")
        st.write(f"• Effect: {results.effect:.6f}")
        st.write(f"• Standard Error: {results.se:.6f}")
        st.write(f"• 95% CI: [{results.ci_lower:.6f}, {results.ci_upper:.6f}]")
        st.write(f"• Method: {results.method}")
        
        if results.use_hksj and results.df is not None:
            st.write(f"• Degrees of Freedom (HKSJ): {results.df}")
    
    with col2:
        st.markdown("**Heterogeneity Statistics**")
        st.write(f"• Tau²: {results.tau2:.6f}")
        st.write(f"• I²: {results.i2:.2f}%")
        st.write(f"• H²: {results.h2:.6f}")
        st.write(f"• Q: {results.q_stat:.6f} (p = {results.q_p_value:.6f})")
        
        if results.points:
            st.write(f"• Number of Studies: {len(results.points)}")
    
    # Interpretation
    st.subheader("📝 Interpretation")
    
    if results.p_value < 0.05:
        st.success(f"The pooled effect is statistically significant (p = {results.p_value:.4f})")
    else:
        st.info(f"The pooled effect is not statistically significant (p = {results.p_value:.4f})")
    
    if results.i2 < 25:
        st.info("Low heterogeneity detected (I² < 25%)")
    elif results.i2 < 75:
        st.warning("Moderate heterogeneity detected (25% ≤ I² < 75%)")
    else:
        st.error("High heterogeneity detected (I² ≥ 75%)")
    
    if results.use_hksj:
        st.info("HKSJ variance adjustment applied - confidence intervals may be wider but more robust")


def _plots_tab(config: MetaAnalysisConfig):
    """Plotting tab."""
    st.header("Visualization")
    
    if 'results' not in st.session_state:
        st.info("Please run an analysis first in the 'Data & Analysis' tab")
        return
    
    results = st.session_state['results']
    
    # Plot type selection
    plot_type = st.selectbox(
        "Select plot type:",
        options=["Forest Plot", "Funnel Plot", "Contour Funnel Plot"],
        index=0
    )
    
    # Plot settings
    col1, col2 = st.columns(2)
    
    with col1:
        fig_width = st.slider("Figure Width", min_value=6, max_value=16, value=10)
    
    with col2:
        fig_height = st.slider("Figure Height", min_value=4, max_value=12, value=8)
    
    try:
        if plot_type == "Forest Plot":
            fig = plot_forest(results, figsize=(fig_width, fig_height))
        elif plot_type == "Funnel Plot":
            fig = plot_funnel(results, figsize=(fig_width, fig_height))
        else:  # Contour Funnel Plot
            fig = plot_funnel_contour(results, figsize=(fig_width, fig_height))
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Download button
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        
        st.download_button(
            label="Download Plot",
            data=img_buffer.getvalue(),
            file_name=f"{plot_type.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")


def _diagnostics_tab(config: MetaAnalysisConfig):
    """Diagnostics tab."""
    st.header("Influence Diagnostics")
    
    if 'results' not in st.session_state or 'data' not in st.session_state:
        st.info("Please run an analysis first in the 'Data & Analysis' tab")
        return
    
    results = st.session_state['results']
    data = st.session_state['data']
    
    # Create points for diagnostics
    points = [
        MetaPoint(
            effect=row['effect'],
            variance=row['variance'],
            study_id=str(row['study'])
        )
        for _, row in data.iterrows()
    ]
    
    # Diagnostic type selection
    diagnostic_type = st.selectbox(
        "Select diagnostic analysis:",
        options=["Leave-One-Out Analysis", "Influence Measures"],
        index=0
    )
    
    if diagnostic_type == "Leave-One-Out Analysis":
        _show_leave_one_out(points, config)
    else:
        _show_influence_measures(points, results, config)


def _show_leave_one_out(points, config):
    """Show leave-one-out analysis."""
    if len(points) < 3:
        st.warning("Need at least 3 studies for meaningful leave-one-out analysis")
        return
    
    try:
        with st.spinner("Performing leave-one-out analysis..."):
            loo_results = leave_one_out_analysis(points, config)
        
        st.subheader("Leave-One-Out Results")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Original Effect",
                f"{loo_results.original_result.effect:.4f}"
            )
        
        with col2:
            st.metric(
                "Max Effect Change",
                f"{loo_results.max_effect_change:.4f}"
            )
        
        with col3:
            st.metric(
                "Most Influential Study",
                loo_results.most_influential_study
            )
        
        # Detailed results table
        st.subheader("Detailed Results")
        df_loo = loo_results.to_dataframe()
        st.dataframe(df_loo)
        
        # Plot effect changes
        st.subheader("Effect Changes Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        studies = loo_results.study_ids
        changes = loo_results.effect_changes
        
        bars = ax.barh(range(len(studies)), changes)
        ax.set_yticks(range(len(studies)))
        ax.set_yticklabels(studies)
        ax.set_xlabel('Change in Effect Size')
        ax.set_title('Leave-One-Out: Effect Size Changes')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3)
        
        # Color bars based on magnitude
        for i, bar in enumerate(bars):
            if abs(changes[i]) > loo_results.max_effect_change * 0.7:
                bar.set_color('red')
            elif abs(changes[i]) > loo_results.max_effect_change * 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('blue')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Error in leave-one-out analysis: {str(e)}")


def _show_influence_measures(points, results, config):
    """Show influence measures."""
    try:
        with st.spinner("Calculating influence measures..."):
            influence_results = influence_measures(points, results, config)
        
        st.subheader("Influence Measures")
        
        # Create DataFrame for display
        influence_data = []
        for inf in influence_results:
            influence_data.append({
                'Study': inf.study_id,
                'Effect': inf.effect,
                'Variance': inf.variance,
                'Weight': inf.weight,
                'Std. Residual': inf.standardized_residual,
                'Leverage': inf.leverage,
                "Cook's Distance": inf.cook_distance,
                'DFFITS': inf.dffits,
                'DFBETAS': inf.dfbetas
            })
        
        df_influence = pd.DataFrame(influence_data)
        st.dataframe(df_influence)
        
        # Identify potential outliers
        st.subheader("Potential Outliers")
        
        from .diagnostics.influence import identify_outliers
        outliers = identify_outliers(influence_results)
        
        if any(outliers.values()):
            for criterion, studies in outliers.items():
                if studies and criterion != 'any_flag':
                    st.warning(f"{criterion.replace('_', ' ').title()}: {', '.join(studies)}")
        else:
            st.success("No studies flagged as potential outliers")
        
        # Influence plot
        st.subheader("Influence Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        leverages = [inf.leverage for inf in influence_results]
        residuals = [inf.standardized_residual for inf in influence_results]
        cook_distances = [inf.cook_distance for inf in influence_results]
        
        scatter = ax.scatter(leverages, residuals, c=cook_distances, 
                           s=60, alpha=0.7, cmap='Reds')
        
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residual')
        ax.set_title('Influence Plot (color = Cook\'s Distance)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Cook's Distance")
        
        # Label high influence points
        for i, inf in enumerate(influence_results):
            if inf.cook_distance > 1.0 or abs(inf.standardized_residual) > 2.0:
                ax.annotate(inf.study_id, (inf.leverage, inf.standardized_residual),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Error calculating influence measures: {str(e)}")


def _export_tab():
    """Export results tab."""
    st.header("Export Results")
    
    if 'results' not in st.session_state:
        st.info("Please run an analysis first in the 'Data & Analysis' tab")
        return
    
    results = st.session_state['results']
    
    # Results summary for export
    export_data = {
        'Method': results.method,
        'Effect': results.effect,
        'Standard_Error': results.se,
        'CI_Lower': results.ci_lower,
        'CI_Upper': results.ci_upper,
        'P_Value': results.p_value,
        'Tau2': results.tau2,
        'I2': results.i2,
        'H2': results.h2,
        'Q_Statistic': results.q_stat,
        'Q_P_Value': results.q_p_value,
        'Use_HKSJ': results.use_hksj
    }
    
    if results.df is not None:
        export_data['HKSJ_DF'] = results.df
    
    # Create downloadable CSV
    df_export = pd.DataFrame([export_data])
    csv_buffer = io.StringIO()
    df_export.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="Download Results (CSV)",
        data=csv_buffer.getvalue(),
        file_name="meta_analysis_results.csv",
        mime="text/csv"
    )
    
    # Create downloadable text report
    report = f"""
PyMeta Analysis Report
=====================

Method: {results.method}
Effect Size: {results.effect:.6f}
Standard Error: {results.se:.6f}
95% Confidence Interval: [{results.ci_lower:.6f}, {results.ci_upper:.6f}]
P-value: {results.p_value:.6f}

Heterogeneity Statistics:
- Tau²: {results.tau2:.6f}
- I²: {results.i2:.2f}%
- H²: {results.h2:.6f}
- Q: {results.q_stat:.6f} (p = {results.q_p_value:.6f})

"""
    
    if results.use_hksj and results.df is not None:
        report += f"HKSJ Adjustment Applied (df = {results.df})\n"
    
    if results.points:
        report += f"Number of Studies: {len(results.points)}\n"
    
    st.download_button(
        label="Download Report (TXT)",
        data=report,
        file_name="meta_analysis_report.txt",
        mime="text/plain"
    )
    
    # Display current results
    st.subheader("Current Results Summary")
    st.dataframe(df_export)


if __name__ == "__main__":
    main()