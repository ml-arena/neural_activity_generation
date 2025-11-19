"""Utility functions for neural activity generation"""

from .metrics import (
    biophysical_representation,
    calculate_frechet_distance,
    fun_trial_matched_metrics,
    fun_frechnet_distance,
    trial_matching
)

__all__ = [
    'biophysical_representation',
    'calculate_frechet_distance',
    'fun_trial_matched_metrics',
    'fun_frechnet_distance',
    'trial_matching'
]
