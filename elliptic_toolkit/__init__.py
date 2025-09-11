"""
Elliptic Bitcoin Dataset Toolkit

A comprehensive Python toolkit for working with the Elliptic Bitcoin dataset,
providing utilities for data loading, temporal analysis, graph neural network
modeling, and evaluation.
"""

__version__ = "0.1.0a1"
__author__ = "Vincenzo Bisogno"
__email__ = "v.bisogno3@studenti.unipi.it"

# Import main classes and functions for easy access
from .dataset import download_dataset, process_dataset, temporal_split, load_labeled_data
from .model_wrappers import GNNBinaryClassifier, MLPWrapper, DropTime
from .temporal_cv import TemporalRollingCV
from .plots import plot_evals, plot_marginals
from .log_parser import parse_search_cv_logs, trim_hyperparameter_results

__all__ = [
    # Dataset utilities
    "download_dataset",
    "process_dataset",
    "temporal_split", 
    "load_labeled_data",
    
    # Model wrappers
    "GNNBinaryClassifier",
    "MLPWrapper",
    "DropTime",
    
    # Cross-validation
    "TemporalRollingCV",
    
    # Plotting utilities
    "plot_evals",
    "plot_marginals",
    
    # Log parsing
    "parse_search_cv_logs",
    "trim_hyperparameter_results",
]