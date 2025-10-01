"""
Elliptic Bitcoin Dataset Toolkit

A comprehensive Python toolkit for working with the Elliptic Bitcoin dataset,
providing utilities for data loading, temporal analysis, graph neural network
modeling, and evaluation.
"""

try:
    from importlib.metadata import version
    __version__ = version(__name__)
except Exception:
    __version__ = "0.0.0"

# Import main classes and functions for easy access
from .dataset import download_dataset, process_dataset, temporal_split, load_labeled_data
from .model_wrappers import GNNBinaryClassifier, MLPBinaryClassifier, DropTime
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
    "MLPBinaryClassifier",
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
