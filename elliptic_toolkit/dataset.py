import os
from functools import singledispatch
import warnings

import pandas as pd
import numpy as np
import torch

from torch_geometric.io import fs


def download_dataset(
    root: str = "elliptic_bitcoin_dataset",
    raw_file_names=[
        'elliptic_txs_features.csv',
        'elliptic_txs_edgelist.csv',
        'elliptic_txs_classes.csv',
    ],
    force: bool = False,
    url: str = 'https://data.pyg.org/datasets/elliptic',
):
    """Download the Elliptic Bitcoin dataset from PyTorch Geometric's dataset repository.

    Args:
        root (str, optional): The root directory where the dataset will be stored. Defaults to "elliptic_bitcoin_dataset".
        raw_file_names (list, optional): List of raw file names to download. Defaults to [ 'elliptic_txs_features.csv', 'elliptic_txs_edgelist.csv', 'elliptic_txs_classes.csv', ].
        force (bool, optional): Whether to force re-download the dataset if it already exists. Defaults to False.
        url (str, optional): The base URL for the dataset files. Defaults to 'https://data.pyg.org/datasets/elliptic'.
    """
    if not fs.exists(root):
        os.mkdir(root)

    for name in raw_file_names:
        if fs.exists(os.path.join(root, name)):
            if force:
                fs.rm(os.path.join(root, name))
            else:
                continue
        fs.cp(f'{url}/{os.path.basename(name)}.zip', root, extract=True)


def process_dataset(
    folder_path: str = "elliptic_bitcoin_dataset",
    features_file: str = "elliptic_txs_features.csv",
    classes_file: str = "elliptic_txs_classes.csv",
    edges_file: str = "elliptic_txs_edgelist.csv",
):
    """
    Loads, validates, and processes the Elliptic Bitcoin dataset.

    Returns
    -------
    nodes_df : pandas.DataFrame
        DataFrame with shape (203769, 167). Columns:

        - 'time': Discrete time step (int)
        - 'feat_0' ... 'feat_164': Node features (float)
        - 'class': Node label (int: 1 for illicit, 0 for licit, -1 for unknown/missing)

        The 'class' column uses -1 to indicate missing labels (transductive setting).
        The 'txId' column is dropped in the returned DataFrame; its original order matches the input file.

    edges_df : pandas.DataFrame
        DataFrame with shape (234355, 2). Columns:

        - 'txId1': Source node index (int, row index in nodes_df)
        - 'txId2': Target node index (int, row index in nodes_df)

        Each row represents a directed edge in the transaction graph, with node indices corresponding to rows in nodes_df.

    Notes
    -----
    - All IDs in 'edges_df' are mapped to row indices in 'nodes_df'.
    - The function performs strict validation on shapes, unique values, and label distribution.
    """
    classes_path = os.path.join(folder_path, classes_file)
    features_path = os.path.join(folder_path, features_file)
    edges_path = os.path.join(folder_path, edges_file)

    classes_df = pd.read_csv(classes_path)
    features_df = pd.read_csv(features_path, header=None)
    edges_df = pd.read_csv(edges_path)
    # Basic checks

    # features checks
    assert features_df.shape == (203769, 167)
    assert features_df[0].nunique() == 203769  # txId is unique
    assert features_df[1].nunique() == 49  # time has 49 unique values

    # classes checks
    assert all(classes_df.columns == ['txId', 'class'])
    assert classes_df.shape == (203769, 2)
    assert set(classes_df['class'].unique()) == set(['unknown', '1', '2'])
    classes_counts = classes_df['class'].value_counts()
    assert classes_counts['unknown'] == 157205
    assert classes_counts['1'] == 4545
    assert classes_counts['2'] == 42019
    assert set(classes_df['txId']) == set(features_df[0])

    # edges checks
    assert edges_df.shape == (234355, 2)
    assert all(edges_df.columns == ['txId1', 'txId2'])
    assert set(edges_df['txId1']).issubset(set(features_df[0]))
    assert set(edges_df['txId2']).issubset(set(features_df[0]))

    features_names = ['txId', 'time'] + [f'feat_{i}' for i in range(165)]
    features_df.columns = features_names

    class_map = {'unknown': -1, '1': 1, '2': 0}
    classes_df['class'] = classes_df['class'].map(class_map)

    nodes_df = features_df.join(classes_df.set_index('txId')[
                                'class'], on='txId', how='left')

    txid_to_idx = pd.Series(nodes_df.index, index=nodes_df['txId'])

    # Map txId1 and txId2 in edges_df to node indices
    edges_df['txId1'] = edges_df['txId1'].map(txid_to_idx)
    edges_df['txId2'] = edges_df['txId2'].map(txid_to_idx)

    return nodes_df.drop(columns=['txId']), edges_df


@singledispatch
def temporal_split(times, test_size=0.2):
    """
    Split data into temporal train/test sets based on unique time steps.

    Parameters
    ----------
    times : np.ndarray, torch.Tensor, or pandas.DataFrame
        The time information or data to split. For DataFrames, must contain a 'time' column.
    test_size : float, default=0.2
        Proportion of unique time steps to include in the test split (between 0.0 and 1.0).

    Returns
    -------
    For array/tensor input:
        train_indices, test_indices : array-like
            Indices for training and test sets.
    For DataFrame input:
        (X_train, y_train), (X_test, y_test) : tuple of tuples
            X_train : pandas.DataFrame
                Training features (all columns except 'class').
            y_train : pandas.Series
                Training labels (the 'class' column).
            X_test : pandas.DataFrame
                Test features (all columns except 'class').
            y_test : pandas.Series
                Test labels (the 'class' column).
        Or, if return_X_y=False:
            train_df, test_df : pandas.DataFrame
                The full training and test DataFrames, already sliced by time.

    Type-specific behavior
    ---------------------
    - np.ndarray: Uses numpy operations to split by unique time values.
    - torch.Tensor: Uses torch operations to split by unique time values (no CPU/GPU transfer).
    - pandas.DataFrame: Splits based on the 'time' column. If return_X_y=True, unpacks X and y based on the 'class' column; otherwise, returns the sliced DataFrames.

    """
    raise NotImplementedError("temporal_split not implemented for this type")


def _temporal_split(times, mod, test_size):
    """
    Core logic for temporal splitting, used by temporal_split for both numpy and torch arrays.
    Issues a warning if n_train or n_test is zero.
    Parameters
    ----------
    times : array-like
        Array of time values (numpy or torch).
    mod : module
        Module to use (np or torch) for unique, isin, where.
    test_size : float
        Proportion of unique time steps to include in the test split.
    Returns
    -------
    train_indices, test_indices : array-like
        Indices for training and test sets.
    """
    assert 0.0 < test_size < 1.0, "test_size must be between 0.0 and 1.0"
    unique_times = mod.unique(times)
    n_test = int(len(unique_times) * test_size)
    n_train = len(unique_times) - n_test

    if n_train == 0 or n_test == 0:
        msg = (
            f"temporal_split: n_train or n_test is zero. "
            f"n_train={n_train}, n_test={n_test}, total unique_times={len(unique_times)}. "
            f"Check your test_size ({test_size}) and data."
        )
        if n_train == 0:
            msg += " All data assigned to test set."
        if n_test == 0:
            msg += " All data assigned to train set."
        warnings.warn(msg)

    train_times = unique_times[:n_train]
    test_times = unique_times[n_train:]
    train_mask = mod.isin(times, train_times)
    test_mask = mod.isin(times, test_times)

    train_indices = mod.where(train_mask)[0]
    test_indices = mod.where(test_mask)[0]
    return train_indices, test_indices


@temporal_split.register(np.ndarray)
def _(times, test_size=0.2):
    """
    Temporal split for numpy arrays.
    See _temporal_split for details.
    """
    return _temporal_split(times, np, test_size)


@temporal_split.register(torch.Tensor)
def _(times, test_size=0.2):
    """
    Temporal split for torch tensors.
    See _temporal_split for details.
    """
    return _temporal_split(times, torch, test_size)


@temporal_split.register(pd.DataFrame)
def _(nodes_df, test_size=0.2, return_X_y=True):
    """
    Temporal split for pandas DataFrames.
    Splits based on the 'time' column. If return_X_y=True, returns (X_train, y_train), (X_test, y_test) tuples;
    otherwise, returns the full train/test DataFrames.
    """
    train_indices, test_indices = temporal_split(
        nodes_df['time'].values, test_size=test_size)

    train_df = nodes_df.iloc[train_indices].reset_index(drop=True)
    test_df = nodes_df.iloc[test_indices].reset_index(drop=True)

    if not return_X_y:
        return train_df, test_df
    X_train, y_train = train_df.drop(columns=['class']), train_df['class']
    X_test, y_test = test_df.drop(columns=['class']), test_df['class']
    return (X_train, y_train), (X_test, y_test)


def load_labeled_data(test_size=0.2, root="elliptic_bitcoin_dataset"):
    """
    Utility function to load data, select only labeled data and split temporally into train and test sets.
    Parameters
    ----------
    test_size : float, default=0.2
        Proportion of unique time steps to include in the test split (between 0.0 and 1.0).
    root : str, optional
        The root directory where the dataset is stored. Defaults to "elliptic_bitcoin_dataset".
    Returns
    -------
    (X_train, y_train), (X_test, y_test) : tuple of tuples
        X_train, y_train: training features and labels
        X_test, y_test: test features and labels
    """
    nodes_df, edges_df = process_dataset(folder_path=root)
    nodes_df = nodes_df[nodes_df['class'] != -1]  # select only labeled data
    (X_train, y_train), (X_test, y_test) = temporal_split(
        nodes_df, test_size=test_size)
    return (X_train, y_train), (X_test, y_test)
