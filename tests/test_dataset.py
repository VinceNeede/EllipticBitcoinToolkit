"""
Tests for the dataset module.
"""
import pytest
import numpy as np
import pandas as pd
import torch
from elliptic_toolkit.dataset import temporal_split


class TestTemporalSplit:
    """Test cases for the temporal_split function."""

    _times = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4])
    _test_size = 0.5
    _train_idx = np.array([0, 1, 2, 3, 4])
    _test_idx = np.array([5, 6, 7, 8])
    _class = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    
    def test_numpy_temporal_split(self):
        """Test temporal split with numpy arrays."""
        times = self._times
        train_idx, test_idx = temporal_split(times, test_size=self._test_size)

        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        
        assert train_idx.dtype == np.int64
        assert test_idx.dtype == np.int64

        assert np.all(train_idx == self._train_idx)
        assert np.all(test_idx == self._test_idx)

    def test_torch_temporal_split(self):
        """Test temporal split with torch tensors."""
        times = torch.tensor(self._times)
        train_idx, test_idx = temporal_split(times, test_size=self._test_size)

        assert isinstance(train_idx, torch.Tensor)
        assert isinstance(test_idx, torch.Tensor)

        assert train_idx.dtype == torch.int64
        assert test_idx.dtype == torch.int64

        assert torch.equal(train_idx, torch.tensor(self._train_idx))
        assert torch.equal(test_idx, torch.tensor(self._test_idx))
        
        # if CUDA is available, test on GPU as well
        if torch.cuda.is_available():
            times_cuda = times.cuda()
            train_idx_cuda, test_idx_cuda = temporal_split(times_cuda, test_size=self._test_size)
            assert train_idx_cuda.is_cuda
            assert test_idx_cuda.is_cuda
            assert torch.equal(train_idx_cuda.cpu(), torch.tensor(self._train_idx))
            assert torch.equal(test_idx_cuda.cpu(), torch.tensor(self._test_idx))

    def test_dataframe_temporal_split(self):
        """Test temporal split with pandas DataFrame."""
        df = pd.DataFrame({
            'time': self._times,
            'feature1': range(len(self._times)),
            'class': self._class
        })

        (X_train, y_train), (X_test, y_test) = temporal_split(df, test_size=self._test_size)

        # Check shapes
        assert len(X_train) + len(X_test) == len(df)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check that class column is removed from features
        assert 'class' not in X_train.columns
        assert 'class' not in X_test.columns
        
        # Check temporal ordering
        assert max(X_train['time']) <= min(X_test['time'])

    def test_dataframe_no_return_xy(self):
        """Test DataFrame temporal split without returning X, y separately."""
        df = pd.DataFrame({
            'time': self._times,
            'feature1': range(len(self._times)),
            'class': self._class
        })

        train_df, test_df = temporal_split(df, test_size=self._test_size, return_X_y=False)
        
        # Check that we get DataFrames
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        
        # Check that class column is preserved
        assert 'class' in train_df.columns
        assert 'class' in test_df.columns

    def test_warning_on_zero_splits(self):
        """Test that warnings are issued when splits result in zero samples."""
        times = np.array([1, 1])  # Only one unique time
        
        with pytest.warns(UserWarning):
            temporal_split(times, test_size=0.9)


if __name__ == "__main__":
    pytest.main([__file__])