"""
Living meta-analysis scheduler with APScheduler integration.

This module provides functionality for automated periodic meta-analysis updates,
supporting both APScheduler-based scheduling and simple time-based polling.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any, List
import threading
import os

from .. import MetaAnalysisConfig, MetaResults, analyze_csv


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Configuration for living meta-analysis scheduler."""
    
    # Data source configuration
    data_source: str  # Path to CSV file or data source
    effect_col: str = "effect"
    variance_col: str = "variance"
    study_col: str = "study"
    
    # Analysis configuration  
    meta_config: MetaAnalysisConfig = field(default_factory=MetaAnalysisConfig)
    
    # Scheduling configuration
    update_interval_seconds: int = 3600  # 1 hour default
    max_retries: int = 3
    retry_delay_seconds: int = 300  # 5 minutes
    
    # Notification configuration
    notify_on_change: bool = True
    change_threshold: float = 0.1  # Minimum effect change to trigger notification
    
    # Output configuration
    output_dir: Optional[str] = None
    save_plots: bool = True
    save_results: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.update_interval_seconds < 60:
            raise ValueError("Update interval must be at least 60 seconds")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")


class LiveMetaAnalysis:
    """
    Living meta-analysis scheduler supporting both APScheduler and simple polling.
    """
    
    def __init__(self, config: SchedulerConfig):
        """
        Initialize living meta-analysis.
        
        Args:
            config: SchedulerConfig object
        """
        self.config = config
        self.is_running = False
        self.last_result: Optional[MetaResults] = None
        self.update_count = 0
        self.scheduler = None
        self._stop_event = threading.Event()
        
        # Try to import APScheduler
        self.use_apscheduler = self._try_import_apscheduler()
        
        # Set up output directory
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _try_import_apscheduler(self) -> bool:
        """
        Try to import APScheduler and set up scheduler.
        
        Returns:
            True if APScheduler is available, False otherwise
        """
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.interval import IntervalTrigger
            
            self.scheduler = BackgroundScheduler()
            logger.info("APScheduler available - using advanced scheduling")
            return True
            
        except ImportError:
            logger.info("APScheduler not available - using simple polling")
            return False
    
    def _run_analysis(self) -> Optional[MetaResults]:
        """
        Run a single meta-analysis update.
        
        Returns:
            MetaResults if successful, None if failed
        """
        try:
            logger.info(f"Running meta-analysis update #{self.update_count + 1}")
            
            # Load and analyze data
            result = analyze_csv(
                filepath=self.config.data_source,
                effect_col=self.config.effect_col,
                variance_col=self.config.variance_col,
                study_col=self.config.study_col,
                config=self.config.meta_config
            )
            
            self.update_count += 1
            
            # Check for significant changes
            if self.last_result is not None and self.config.notify_on_change:
                effect_change = abs(result.effect - self.last_result.effect)
                if effect_change >= self.config.change_threshold:
                    logger.warning(
                        f"Significant change detected: effect changed by {effect_change:.3f} "
                        f"(from {self.last_result.effect:.3f} to {result.effect:.3f})"
                    )
                    self._notify_change(result, self.last_result)
            
            # Save results if configured
            if self.config.save_results and self.config.output_dir:
                self._save_results(result)
            
            # Save plots if configured
            if self.config.save_plots and self.config.output_dir:
                self._save_plots(result)
            
            self.last_result = result
            logger.info(f"Analysis completed successfully. Effect: {result.effect:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Meta-analysis update failed: {str(e)}")
            return None
    
    def _run_analysis_with_retry(self) -> Optional[MetaResults]:
        """
        Run analysis with retry logic.
        
        Returns:
            MetaResults if successful, None if all retries failed
        """
        for attempt in range(self.config.max_retries + 1):
            result = self._run_analysis()
            if result is not None:
                return result
                
            if attempt < self.config.max_retries:
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in "
                    f"{self.config.retry_delay_seconds} seconds..."
                )
                time.sleep(self.config.retry_delay_seconds)
            else:
                logger.error("All retry attempts failed")
                
        return None
    
    def _save_results(self, result: MetaResults) -> None:
        """Save analysis results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"meta_results_{timestamp}.txt"
            filepath = os.path.join(self.config.output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(f"Meta-Analysis Results - {datetime.now()}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Method: {result.method}\n")
                f.write(f"Effect: {result.effect:.6f}\n")
                f.write(f"Standard Error: {result.se:.6f}\n")
                f.write(f"95% CI: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]\n")
                f.write(f"P-value: {result.p_value:.6f}\n")
                f.write(f"Tau²: {result.tau2:.6f}\n")
                f.write(f"I²: {result.i2:.2f}%\n")
                f.write(f"Q statistic: {result.q_stat:.6f}\n")
                f.write(f"Q p-value: {result.q_p_value:.6f}\n")
                
                if result.use_hksj and result.df is not None:
                    f.write(f"Degrees of freedom (HKSJ): {result.df}\n")
                    
                f.write(f"\nNumber of studies: {len(result.points) if result.points else 'N/A'}\n")
                f.write(f"Update count: {self.update_count}\n")
            
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def _save_plots(self, result: MetaResults) -> None:
        """Save analysis plots to files."""
        try:
            from ..plots import plot_forest, plot_funnel_contour
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save forest plot
            forest_fig = plot_forest(result)
            forest_path = os.path.join(self.config.output_dir, f"forest_{timestamp}.png")
            forest_fig.savefig(forest_path, dpi=300, bbox_inches='tight')
            forest_fig.close()
            
            # Save funnel plot
            funnel_fig = plot_funnel_contour(result)
            funnel_path = os.path.join(self.config.output_dir, f"funnel_{timestamp}.png")
            funnel_fig.savefig(funnel_path, dpi=300, bbox_inches='tight')
            funnel_fig.close()
            
            logger.info(f"Plots saved to {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save plots: {str(e)}")
    
    def _notify_change(self, new_result: MetaResults, old_result: MetaResults) -> None:
        """
        Notify about significant changes (placeholder for notification system).
        
        Args:
            new_result: Current analysis results
            old_result: Previous analysis results
        """
        # This is a placeholder - in practice, you might want to:
        # - Send email notifications
        # - Post to Slack/Teams
        # - Write to a log file
        # - Trigger webhooks
        # etc.
        
        effect_change = new_result.effect - old_result.effect
        logger.warning(
            f"CHANGE NOTIFICATION: Meta-analysis effect changed by {effect_change:.3f} "
            f"(threshold: {self.config.change_threshold:.3f})"
        )
    
    def start(self) -> None:
        """Start the living meta-analysis scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info(f"Starting living meta-analysis scheduler")
        logger.info(f"Data source: {self.config.data_source}")
        logger.info(f"Update interval: {self.config.update_interval_seconds} seconds")
        logger.info(f"Using APScheduler: {self.use_apscheduler}")
        
        self.is_running = True
        self._stop_event.clear()
        
        if self.use_apscheduler:
            self._start_with_apscheduler()
        else:
            self._start_with_polling()
    
    def _start_with_apscheduler(self) -> None:
        """Start scheduler using APScheduler."""
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Add job to scheduler
        self.scheduler.add_job(
            func=self._run_analysis_with_retry,
            trigger=IntervalTrigger(seconds=self.config.update_interval_seconds),
            id='meta_analysis_update',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("APScheduler started successfully")
        
        # Run initial analysis
        self._run_analysis_with_retry()
    
    def _start_with_polling(self) -> None:
        """Start scheduler using simple polling in a separate thread."""
        def polling_loop():
            # Run initial analysis
            self._run_analysis_with_retry()
            
            while not self._stop_event.is_set():
                # Wait for the specified interval
                if self._stop_event.wait(timeout=self.config.update_interval_seconds):
                    break  # Stop event was set
                
                # Run analysis
                if not self._stop_event.is_set():
                    self._run_analysis_with_retry()
        
        # Start polling in background thread
        self.polling_thread = threading.Thread(target=polling_loop, daemon=True)
        self.polling_thread.start()
        logger.info("Polling scheduler started successfully")
    
    def stop(self) -> None:
        """Stop the living meta-analysis scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        logger.info("Stopping living meta-analysis scheduler")
        
        self.is_running = False
        self._stop_event.set()
        
        if self.use_apscheduler and self.scheduler:
            self.scheduler.shutdown(wait=True)
            logger.info("APScheduler stopped")
        else:
            # Wait for polling thread to finish
            if hasattr(self, 'polling_thread'):
                self.polling_thread.join(timeout=5)
            logger.info("Polling scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the living meta-analysis.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'is_running': self.is_running,
            'update_count': self.update_count,
            'last_update': None,
            'last_effect': None,
            'use_apscheduler': self.use_apscheduler,
            'config': {
                'data_source': self.config.data_source,
                'update_interval_seconds': self.config.update_interval_seconds,
                'use_hksj': self.config.meta_config.use_hksj,
                'model': self.config.meta_config.model
            }
        }
        
        if self.last_result:
            status['last_effect'] = self.last_result.effect
            status['last_update'] = 'Available'  # In practice, track actual timestamp
        
        return status


# Convenience function for quick setup
def start_living_analysis(
    data_source: str,
    update_interval_seconds: int = 3600,
    use_hksj: bool = False,
    output_dir: Optional[str] = None,
    **kwargs
) -> LiveMetaAnalysis:
    """
    Convenience function to quickly start a living meta-analysis.
    
    Args:
        data_source: Path to CSV file
        update_interval_seconds: How often to update (default: 1 hour)
        use_hksj: Whether to use HKSJ variance adjustment
        output_dir: Directory to save results and plots
        **kwargs: Additional configuration options
        
    Returns:
        LiveMetaAnalysis instance (already started)
    """
    meta_config = MetaAnalysisConfig(use_hksj=use_hksj)
    
    scheduler_config = SchedulerConfig(
        data_source=data_source,
        update_interval_seconds=update_interval_seconds,
        meta_config=meta_config,
        output_dir=output_dir,
        **kwargs
    )
    
    live_analysis = LiveMetaAnalysis(scheduler_config)
    live_analysis.start()
    
    return live_analysis