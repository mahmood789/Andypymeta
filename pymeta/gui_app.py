"""Advanced Streamlit GUI application for PyMeta."""

import streamlit as st
import pandas as pd
import numpy as np
import io
from pathlib import Path

# Check if streamlit is available
try:
    from .suite import PyMeta
    from .io.datasets import create_meta_points_from_dataframe, create_example_data
    from .config import config
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st.error("Streamlit dependencies not available")


def main():
    """Main Streamlit application."""
    if not STREAMLIT_AVAILABLE:
        st.error("Cannot run Streamlit app - dependencies missing")
        return
    
    st.set_page_config(
        page_title="PyMeta - Meta-Analysis Suite",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 PyMeta - Meta-Analysis Suite")
    st.markdown("**Comprehensive modular meta-analysis toolkit**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a panel:",
        ["Analyze", "Living Meta (TSA)", "Diagnostics/Plots", "Data Simulation", "About"]
    )
    
    if page == "Analyze":
        analyze_panel()
    elif page == "Living Meta (TSA)":
        tsa_panel()
    elif page == "Diagnostics/Plots":
        plots_panel()
    elif page == "Data Simulation":
        simulation_panel()
    else:
        about_panel()


def analyze_panel():
    """Main analysis panel."""
    st.header("🔍 Meta-Analysis")
    
    # Data input section
    st.subheader("Data Input")
    
    data_source = st.radio(
        "Choose data source:",
        ["Upload CSV file", "Use example data", "Manual entry"]
    )
    
    points = None
    
    if data_source == "Upload CSV file":
        uploaded_file = st.file_uploader(
            "Upload CSV file with meta-analysis data",
            type=['csv']
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Data preview:")
            st.dataframe(df.head())
            
            # Column mapping
            col1, col2, col3 = st.columns(3)
            with col1:
                effect_col = st.selectbox("Effect size column:", df.columns)
            with col2:
                variance_col = st.selectbox("Variance column:", df.columns, index=1 if len(df.columns) > 1 else 0)
            with col3:
                label_col = st.selectbox("Study label column (optional):", ["None"] + list(df.columns))
            
            if st.button("Load Data"):
                try:
                    points = create_meta_points_from_dataframe(
                        df, 
                        effect_col=effect_col,
                        variance_col=variance_col,
                        label_col=None if label_col == "None" else label_col
                    )
                    st.success(f"Loaded {len(points)} studies successfully!")
                    st.session_state.points = points
                except Exception as e:
                    st.error(f"Error loading data: {e}")
    
    elif data_source == "Use example data":
        n_studies = st.slider("Number of studies:", 3, 20, 8)
        true_effect = st.slider("True effect size:", -2.0, 2.0, 0.5, 0.1)
        tau2 = st.slider("Between-study variance (tau²):", 0.0, 1.0, 0.1, 0.05)
        seed = st.number_input("Random seed (optional):", value=42, min_value=0)
        
        if st.button("Generate Example Data"):
            points = create_example_data(n_studies, true_effect, tau2, seed)
            st.success(f"Generated {len(points)} example studies!")
            st.session_state.points = points
    
    else:  # Manual entry
        st.info("Manual data entry - coming soon in future version")
    
    # Get points from session state
    if 'points' not in st.session_state:
        points = None
    else:
        points = st.session_state.points
    
    if points:
        # Analysis settings
        st.subheader("Analysis Settings")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            model_type = st.selectbox(
                "Model type:",
                ["random_effects", "fixed_effects", "glmm_binomial"]
            )
        with col2:
            tau2_estimator = st.selectbox(
                "Tau² estimator:",
                ["DL", "PM", "REML", "ML"]
            )
        with col3:
            alpha = st.slider("Significance level:", 0.01, 0.10, 0.05, 0.01)
        
        # Perform analysis
        if st.button("Run Analysis", type="primary"):
            try:
                with st.spinner("Running meta-analysis..."):
                    meta = PyMeta(points, model_type=model_type, tau2_estimator=tau2_estimator, alpha=alpha)
                    results = meta.analyze()
                    st.session_state.meta = meta
                    st.session_state.results = results
                
                st.success("Analysis completed!")
                
                # Display results
                display_results(results)
                
                # Publication bias tests
                st.subheader("Publication Bias Tests")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Egger Test"):
                        try:
                            egger_result = meta.test_bias('egger')
                            st.write(f"**Egger's Test**")
                            st.write(f"Statistic: {egger_result.statistic:.4f}")
                            st.write(f"P-value: {egger_result.p_value:.4f}")
                            st.write(f"Interpretation: {egger_result.interpretation}")
                        except Exception as e:
                            st.error(f"Egger test failed: {e}")
                
                with col2:
                    if st.button("Begg Test"):
                        try:
                            begg_result = meta.test_bias('begg')
                            st.write(f"**Begg's Test**")
                            st.write(f"Statistic: {begg_result.statistic:.4f}")
                            st.write(f"P-value: {begg_result.p_value:.4f}")
                            st.write(f"Interpretation: {begg_result.interpretation}")
                        except Exception as e:
                            st.error(f"Begg test failed: {e}")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")


def tsa_panel():
    """Trial Sequential Analysis panel."""
    st.header("📈 Living Meta-Analysis (TSA)")
    
    if 'points' not in st.session_state:
        st.warning("Please load data in the Analyze panel first.")
        return
    
    points = st.session_state.points
    
    # TSA settings
    st.subheader("TSA Settings")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        delta = st.slider("Clinically relevant effect (δ):", 0.1, 2.0, 0.5, 0.1)
    with col2:
        tsa_alpha = st.slider("Type I error (α):", 0.01, 0.10, 0.05, 0.01)
    with col3:
        tsa_beta = st.slider("Type II error (β):", 0.05, 0.30, 0.20, 0.05)
    
    col1, col2 = st.columns(2)
    with col1:
        boundary_type = st.selectbox("Boundary type:", ["obrien_fleming", "pocock"])
    with col2:
        model_type = st.selectbox("Model for cumulative analysis:", ["fixed_effects", "random_effects"])
    
    if st.button("Run TSA Analysis", type="primary"):
        try:
            with st.spinner("Performing Trial Sequential Analysis..."):
                meta = PyMeta(points)
                tsa_result = meta.perform_tsa(
                    delta=delta,
                    alpha=tsa_alpha,
                    beta=tsa_beta,
                    model_type=model_type,
                    boundary_type=boundary_type
                )
                st.session_state.tsa_result = tsa_result
            
            st.success("TSA completed!")
            
            # Display TSA results
            st.subheader("TSA Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Required Information Size", f"{tsa_result.required_information_size:.1f}")
                st.metric("Monitoring Boundary Reached", tsa_result.monitoring_boundary_reached)
            with col2:
                st.metric("Superiority Reached", tsa_result.superiority_reached)
                st.metric("Futility Reached", tsa_result.futility_reached)
            
            # TSA interpretation
            if tsa_result.superiority_reached:
                st.success("🎯 Superiority boundary crossed - consider stopping for efficacy")
            elif tsa_result.futility_reached:
                st.warning("⚠️ Futility boundary crossed - consider stopping for futility")
            else:
                current_info = sum(p.weight for p in points)
                info_fraction = current_info / tsa_result.required_information_size
                st.info(f"📊 Current information fraction: {info_fraction:.2f}")
                if info_fraction < 1.0:
                    st.info("More studies needed to reach required information size")
            
            # Plot TSA
            try:
                fig = meta.plot_tsa(tsa_result)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"TSA plot failed: {e}")
                
        except Exception as e:
            st.error(f"TSA analysis failed: {e}")


def plots_panel():
    """Diagnostics and plots panel."""
    st.header("📊 Diagnostics & Plots")
    
    if 'points' not in st.session_state:
        st.warning("Please load data in the Analyze panel first.")
        return
    
    points = st.session_state.points
    meta = PyMeta(points)
    
    # Plot selection
    plot_type = st.selectbox(
        "Select plot type:",
        ["Forest Plot", "Funnel Plot", "Baujat Plot", "Radial Plot", "GOSH Plot"]
    )
    
    # Plot style settings
    col1, col2 = st.columns(2)
    with col1:
        style = st.selectbox("Plot style:", ["default", "publication", "presentation"])
    with col2:
        if plot_type == "GOSH Plot":
            gosh_subsets = st.slider("GOSH subsets:", 100, 5000, 1000, 100)
    
    if st.button("Generate Plot", type="primary"):
        try:
            with st.spinner(f"Generating {plot_type.lower()}..."):
                config.set_plot_style(style)
                
                if plot_type == "Forest Plot":
                    results = meta.analyze()
                    fig = meta.plot_forest(style=style)
                elif plot_type == "Funnel Plot":
                    fig = meta.plot_funnel(style=style)
                elif plot_type == "Baujat Plot":
                    if len(points) < 3:
                        st.error("Baujat plot requires at least 3 studies")
                        return
                    fig = meta.plot_baujat(style=style)
                elif plot_type == "Radial Plot":
                    fig = meta.plot_radial(style=style)
                elif plot_type == "GOSH Plot":
                    if len(points) < 4:
                        st.error("GOSH plot requires at least 4 studies")
                        return
                    fig = meta.plot_gosh(max_subsets=gosh_subsets, style=style)
            
            st.pyplot(fig)
            
            # Offer download
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            st.download_button(
                label="Download Plot",
                data=buffer,
                file_name=f"{plot_type.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Plot generation failed: {e}")


def simulation_panel():
    """Data simulation panel."""
    st.header("🎲 Data Simulation")
    
    st.markdown("Generate simulated meta-analysis data for testing and demonstration.")
    
    # Simulation parameters
    col1, col2 = st.columns(2)
    with col1:
        n_studies = st.slider("Number of studies:", 3, 50, 10)
        true_effect = st.slider("True effect size:", -3.0, 3.0, 0.5, 0.1)
    with col2:
        tau2 = st.slider("Between-study variance (tau²):", 0.0, 2.0, 0.1, 0.05)
        seed = st.number_input("Random seed:", value=42, min_value=0)
    
    if st.button("Generate Simulated Data", type="primary"):
        try:
            points = create_example_data(n_studies, true_effect, tau2, seed)
            st.session_state.points = points
            
            # Display simulated data
            data = []
            for i, point in enumerate(points):
                data.append({
                    'Study': point.label,
                    'Effect': round(point.effect, 4),
                    'Variance': round(point.variance, 4),
                    'SE': round(np.sqrt(point.variance), 4),
                    'Weight': round(point.weight, 4)
                })
            
            df = pd.DataFrame(data)
            st.subheader("Simulated Data")
            st.dataframe(df)
            
            # Quick analysis of simulated data
            meta = PyMeta(points)
            results = meta.analyze()
            
            st.subheader("Quick Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pooled Effect", f"{results.pooled_effect:.4f}")
            with col2:
                st.metric("I² (%)", f"{results.i_squared:.1f}")
            with col3:
                st.metric("P-value", f"{results.p_value:.4f}")
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="simulated_data.csv",
                mime="text/csv"
            )
            
            st.success(f"Generated {len(points)} studies! You can now use this data in other panels.")
            
        except Exception as e:
            st.error(f"Simulation failed: {e}")


def about_panel():
    """About panel with package information."""
    st.header("ℹ️ About PyMeta")
    
    st.markdown(f"""
    **PyMeta version {config.version}**
    
    A comprehensive modular meta-analysis package for Python.
    
    ### Features:
    - Multiple tau² estimators: DerSimonian-Laird, Paule-Mandel, REML
    - Model types: Fixed Effects, Random Effects, GLMM Binomial
    - Publication bias tests: Egger, Begg
    - Trial Sequential Analysis (TSA)
    - Comprehensive plotting suite: Forest, Funnel, Baujat, Radial, GOSH
    - Command-line interface
    - Streamlit GUI application
    
    ### Available Models:
    """)
    
    for model in ['fixed_effects', 'random_effects', 'glmm_binomial']:
        st.write(f"- {model}")
    
    st.markdown("### Available Estimators:")
    for estimator in ['DL', 'PM', 'REML', 'ML']:
        st.write(f"- {estimator}")
    
    st.markdown("### Available Plot Types:")
    for plot_type in ['Forest', 'Funnel', 'Baujat', 'Radial', 'GOSH']:
        st.write(f"- {plot_type}")
    
    st.markdown("""
    ### Usage:
    1. Load your data in the **Analyze** panel
    2. Run meta-analysis with your preferred model
    3. Perform publication bias tests
    4. Use **Living Meta (TSA)** for trial sequential analysis
    5. Generate diagnostic plots in **Diagnostics/Plots**
    6. Simulate test data in **Data Simulation**
    
    ### Data Format:
    Your CSV file should contain columns for:
    - Effect sizes
    - Variances or standard errors
    - Study labels (optional)
    
    ### License:
    Apache License 2.0
    """)


def display_results(results):
    """Display meta-analysis results in a formatted way."""
    st.subheader("Analysis Results")
    
    # Main results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pooled Effect", f"{results.pooled_effect:.4f}")
    with col2:
        st.metric("Standard Error", f"{results.pooled_se:.4f}")
    with col3:
        st.metric("P-value", f"{results.p_value:.4f}")
    
    # Confidence interval
    st.write(f"**95% Confidence Interval:** [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]")
    
    # Heterogeneity
    st.subheader("Heterogeneity Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tau²", f"{results.tau2:.4f}")
    with col2:
        st.metric("I² (%)", f"{results.i_squared:.1f}")
    with col3:
        st.metric("Q Statistic", f"{results.q_statistic:.4f}")
    
    st.write(f"**Q Test P-value:** {results.q_p_value:.4f}")
    
    # Interpretation
    if results.p_value < 0.05:
        st.success("🎯 Statistically significant result")
    else:
        st.info("📊 Non-significant result")
    
    if results.i_squared > 75:
        st.warning("⚠️ High heterogeneity detected (I² > 75%)")
    elif results.i_squared > 50:
        st.info("📊 Moderate heterogeneity (I² > 50%)")
    else:
        st.success("✅ Low heterogeneity")


if __name__ == '__main__':
    main()