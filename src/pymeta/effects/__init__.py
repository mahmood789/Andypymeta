"""Binary effects module for meta-analysis of binary outcomes."""

import numpy as np
import pandas as pd


def binary_effects(events_treatment, n_treatment, events_control, n_control, 
                  effect_measure="OR", continuity_correction=0.5):
    """
    Calculate effect sizes for binary outcomes.
    
    Parameters
    ----------
    events_treatment : array_like
        Number of events in treatment group
    n_treatment : array_like  
        Sample size in treatment group
    events_control : array_like
        Number of events in control group
    n_control : array_like
        Sample size in control group
    effect_measure : str, default "OR"
        Effect measure: "OR" (odds ratio), "RR" (risk ratio), "RD" (risk difference)
    continuity_correction : float, default 0.5
        Continuity correction for zero events
        
    Returns
    -------
    pd.DataFrame
        DataFrame with effect sizes and variances
    """
    events_treatment = np.asarray(events_treatment)
    n_treatment = np.asarray(n_treatment)
    events_control = np.asarray(events_control)
    n_control = np.asarray(n_control)
    
    # Apply continuity correction for zero events
    adj_events_treatment = np.where(
        (events_treatment == 0) | (events_control == 0), 
        events_treatment + continuity_correction, 
        events_treatment
    )
    adj_events_control = np.where(
        (events_treatment == 0) | (events_control == 0),
        events_control + continuity_correction, 
        events_control
    )
    adj_n_treatment = np.where(
        (events_treatment == 0) | (events_control == 0),
        n_treatment + 2 * continuity_correction,
        n_treatment
    )
    adj_n_control = np.where(
        (events_treatment == 0) | (events_control == 0),
        n_control + 2 * continuity_correction,
        n_control
    )
    
    if effect_measure == "OR":
        # Log odds ratio
        log_or = (np.log(adj_events_treatment) - np.log(adj_n_treatment - adj_events_treatment) - 
                 np.log(adj_events_control) + np.log(adj_n_control - adj_events_control))
        var_log_or = (1/adj_events_treatment + 1/(adj_n_treatment - adj_events_treatment) + 
                     1/adj_events_control + 1/(adj_n_control - adj_events_control))
        
        return pd.DataFrame({
            'effect_size': log_or,
            'variance': var_log_or,
            'se': np.sqrt(var_log_or),
            'measure': 'log_OR'
        })
        
    elif effect_measure == "RR":
        # Log risk ratio
        log_rr = np.log(adj_events_treatment/adj_n_treatment) - np.log(adj_events_control/adj_n_control)
        var_log_rr = (1/adj_events_treatment - 1/adj_n_treatment + 
                     1/adj_events_control - 1/adj_n_control)
        
        return pd.DataFrame({
            'effect_size': log_rr,
            'variance': var_log_rr,
            'se': np.sqrt(var_log_rr),
            'measure': 'log_RR'
        })
        
    elif effect_measure == "RD":
        # Risk difference
        rd = adj_events_treatment/adj_n_treatment - adj_events_control/adj_n_control
        var_rd = (adj_events_treatment * (adj_n_treatment - adj_events_treatment) / adj_n_treatment**3 + 
                 adj_events_control * (adj_n_control - adj_events_control) / adj_n_control**3)
        
        return pd.DataFrame({
            'effect_size': rd,
            'variance': var_rd,
            'se': np.sqrt(var_rd),
            'measure': 'RD'
        })
    
    else:
        raise ValueError(f"Unknown effect measure: {effect_measure}")