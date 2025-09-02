"""Enhanced MetaAnalysisConfig with plot styles and extras."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt


@dataclass
class PlotStyle:
    """Configuration for plot styling."""
    figure_size: tuple = (10, 8)
    dpi: int = 300
    font_family: str = 'serif'
    font_size: int = 12
    color_palette: List[str] = field(default_factory=lambda: ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    grid: bool = True
    grid_alpha: float = 0.3
    spine_style: str = 'default'  # 'default', 'minimal', 'publication'
    legend_style: Dict[str, Any] = field(default_factory=lambda: {'frameon': True, 'shadow': True})
    
    def apply_style(self):
        """Apply this style to matplotlib."""
        plt.rcParams.update({
            'figure.figsize': self.figure_size,
            'figure.dpi': self.dpi,
            'font.family': self.font_family,
            'font.size': self.font_size,
            'axes.grid': self.grid,
            'axes.grid.alpha': self.grid_alpha,
        })


@dataclass
class MetaAnalysisConfig:
    """Enhanced configuration for meta-analysis operations."""
    
    # Version information
    version: str = "4.1-modular"
    
    # Analysis parameters
    alpha: float = 0.05
    confidence_level: float = 0.95
    continuity_correction: float = 0.5
    
    # Tau2 estimation
    default_tau2_estimator: str = "DL"
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # Plotting
    plot_styles: Dict[str, PlotStyle] = field(default_factory=lambda: {
        'default': PlotStyle(),
        'publication': PlotStyle(
            figure_size=(12, 8),
            font_family='serif',
            font_size=14,
            spine_style='publication',
            grid=False
        ),
        'presentation': PlotStyle(
            figure_size=(16, 10),
            font_size=16,
            color_palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
    })
    current_plot_style: str = 'default'
    
    # GOSH sampling
    gosh_max_subsets: int = 10000
    gosh_seed: Optional[int] = None
    
    # TSA parameters
    tsa_alpha: float = 0.05
    tsa_beta: float = 0.20
    tsa_delta: Optional[float] = None
    
    # Optional dependencies handling
    optional_deps: Dict[str, bool] = field(default_factory=lambda: {
        'statsmodels': True,
        'streamlit': True,
        'apscheduler': True
    })
    
    # Bias test parameters
    egger_intercept_threshold: float = 0.05
    
    # Model preferences
    prefer_glmm_for_binary: bool = True
    glmm_fallback: str = "random_effects"
    
    def get_plot_style(self, name: Optional[str] = None) -> PlotStyle:
        """Get plot style by name or current style."""
        style_name = name or self.current_plot_style
        if style_name not in self.plot_styles:
            return self.plot_styles['default']
        return self.plot_styles[style_name]
    
    def set_plot_style(self, name: str):
        """Set current plot style."""
        if name in self.plot_styles:
            self.current_plot_style = name
            self.get_plot_style(name).apply_style()
        else:
            raise ValueError(f"Unknown plot style: {name}")
    
    def check_dependency(self, name: str) -> bool:
        """Check if optional dependency is available."""
        if name not in self.optional_deps:
            return False
        
        if not self.optional_deps[name]:
            return False
            
        try:
            if name == 'statsmodels':
                import statsmodels
                return True
            elif name == 'streamlit':
                import streamlit
                return True
            elif name == 'apscheduler':
                import apscheduler
                return True
        except ImportError:
            self.optional_deps[name] = False
            return False
        
        return True
    
    @property
    def z_critical(self) -> float:
        """Critical Z value for current confidence level."""
        from scipy import stats
        return stats.norm.ppf(1 - self.alpha / 2)


# Global configuration instance
config = MetaAnalysisConfig()