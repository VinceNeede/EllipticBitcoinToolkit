import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import TimeSeriesSplit

class TemporalRollingCV(TimeSeriesSplit):
    """
    Time-based cross-validation iterator that extends scikit-learn's TimeSeriesSplit
    to work with data that has explicit time step values (like the Elliptic Bitcoin dataset).
    
    This class inherits from TimeSeriesSplit and adds functionality to handle datasets
    where multiple samples can belong to the same time step. It maps the time step indices
    to actual row indices in the dataset, allowing it to be used with datasets like
    the Elliptic Bitcoin dataset.
    
    This CV strategy ensures that for each fold:
    1. Training data comes from earlier time periods
    2. The test set is a continuous time window following the training data
    3. Each fold expands the training window and shifts the test window forward
    
    Parameters:
    -----------
    n_splits : int, default=5
        Number of splits to generate
    test_size : int, default=None
        Size of test window in time steps. If None, will be calculated based on n_splits.
    max_train_size : int, default=None
        Maximum number of time steps to use for training. If None, all available time steps
        will be used.
    gap : int, default=0
        Number of time steps to skip between training and test sets
    time_col : str, default='time'
        Name of the column containing time step information
    """
    def __init__(self, n_splits=5, *, test_size=None, max_train_size=None, gap=0, time_col='time'):
        super().__init__(n_splits=n_splits, test_size=test_size, max_train_size=max_train_size, gap=gap)
        self.time_col = time_col

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.
        
        Unlike standard TimeSeriesSplit, this method works with explicit time step values
        and maps them to actual row indices in the dataset. This allows it to handle
        datasets where multiple samples can belong to the same time step.
        
        Parameters:
        -----------
        X : array-like, DataFrame
            Training data. If DataFrame, must contain the column specified by `time_col`.
            Otherwise, time values must be passed through the `groups` parameter.
        y : array-like, optional
            Targets for the training data (ignored)
        groups : array-like, optional
            Time values for each sample if X doesn't have the time column specified by time_col
            
        Yields:
        -------
        train_index : ndarray
            Indices of rows in the training set
        test_index : ndarray
            Indices of rows in the test set
            
        Notes:
        ------
        The yielded indices refer to rows in the original dataset, not time steps.
        This makes the cross-validator compatible with scikit-learn's model selection tools.
        """
        # Get time values
        if hasattr(X, self.time_col) and isinstance(getattr(X, self.time_col), pd.Series):
            times = getattr(X, self.time_col).values
        elif groups is not None:
            times = groups
        else:
            raise ValueError(f"X must have a '{self.time_col}' column or time values must be passed as groups")
        
        if isinstance(times, np.ndarray) or isinstance(times, pd.Series):
            mod = np
        elif isinstance(times, torch.Tensor):
            mod = torch
        else:
            raise ValueError("times must be a numpy array, torch tensor, or pandas Series")        

        # Get unique time steps and sort
        unique_times = mod.unique(times)
        for train_times, test_times in super().split(unique_times):
            train_mask = mod.isin(times, unique_times[train_times])
            test_mask = mod.isin(times, unique_times[test_times])
            train_indices = mod.where(train_mask)[0]
            test_indices = mod.where(test_mask)[0]
            yield train_indices, test_indices
            

