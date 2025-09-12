import pytest
import pandas as pd
import numpy as np
import torch

from elliptic_toolkit.temporal_cv import TemporalRollingCV


class TestTemporalCV:

    _times = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    _n_splits = 2
    _class = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    """Test cases for the TemporalCV class."""

    def test_dataframe(self):
        df = pd.DataFrame({
            'time': self._times,
            'feature1': range(len(self._times)),
            'class': self._class
        })
        tscv = TemporalRollingCV(n_splits=self._n_splits)
        splits = list(tscv.split(df))
        for train_idx, test_idx in splits:
            assert all(df.loc[train_idx, 'time']
                       <= df.loc[test_idx, 'time'].min())

    def test_max_train_size(self):
        df = pd.DataFrame({
            'time': self._times,
            'feature1': range(len(self._times)),
            'class': self._class
        })
        tscv = TemporalRollingCV(n_splits=self._n_splits, max_train_size=2)
        splits = list(tscv.split(df))
        for train_idx, test_idx in splits:
            assert len(set(df.loc[train_idx, 'time'])) <= 2

    def test_gap(self):
        df = pd.DataFrame({
            'time': self._times,
            'feature1': range(len(self._times)),
            'class': self._class
        })
        tscv = TemporalRollingCV(n_splits=self._n_splits, gap=1)
        splits = list(tscv.split(df))
        for train_idx, test_idx in splits:
            assert df.loc[train_idx, 'time'].max(
            ) < df.loc[test_idx, 'time'].min() - 1

    def test_test_size(self):
        df = pd.DataFrame({
            'time': self._times,
            'feature1': range(len(self._times)),
            'class': self._class
        })
        tscv = TemporalRollingCV(n_splits=self._n_splits, test_size=2)
        splits = list(tscv.split(df))
        for train_idx, test_idx in splits:
            assert len(set(df.loc[test_idx, 'time'])) == 2

    def test_numpy_array(self):
        times = self._times
        tscv = TemporalRollingCV(n_splits=self._n_splits)
        splits = list(tscv.split(times, groups=times))
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert train_idx.dtype == np.int64
            assert test_idx.dtype == np.int64
            assert all(times[train_idx] <= times[test_idx].min())

    def test_torch_tensor(self):
        times = torch.tensor(self._times)
        tscv = TemporalRollingCV(n_splits=self._n_splits)
        splits = list(tscv.split(times, groups=times))
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, torch.Tensor)
            assert isinstance(test_idx, torch.Tensor)
            assert train_idx.dtype == torch.int64
            assert test_idx.dtype == torch.int64
            assert all(times[train_idx] <= times[test_idx].min())

    def test_torch_tensor_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        times = torch.tensor(self._times).cuda()
        tscv = TemporalRollingCV(n_splits=self._n_splits)
        splits = list(tscv.split(times, groups=times))
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, torch.Tensor)
            assert isinstance(test_idx, torch.Tensor)
            assert train_idx.is_cuda
            assert test_idx.is_cuda
            assert train_idx.dtype == torch.int64
            assert test_idx.dtype == torch.int64
            assert all(times[train_idx] <= times[test_idx].min())


if __name__ == "__main__":
    pytest.main([__file__])
