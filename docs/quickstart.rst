Quickstart
==========

This guide will get you started with the Elliptic Bitcoin Toolkit v0.1.0a4.

Basic Usage
-----------

Here's a simple example to get you started:

.. code-block:: python

   import elliptic_toolkit as et
   import numpy as np

   # Example: Temporal split
   times = np.array([1, 1, 2, 2, 3, 3, 4, 4])
   train_idx, test_idx = et.temporal_split(times, test_size=0.3)

   print(f"Training indices: {train_idx}")
   print(f"Test indices: {test_idx}")

Loading the Dataset
-------------------

To work with the Elliptic Bitcoin dataset:

.. code-block:: python

   from elliptic_toolkit import download_dataset, load_labeled_data

   download_dataset()  # Download the dataset if you haven't already

   # Load and Process for machine learning
   (X_train, y_train), (X_test, y_test) = load_labeled_data()

   print(f"Training dataset shape: {X_train.shape}")
   print(f"Test dataset shape: {X_test.shape}")
   print(f"Class distribution in training set: {y_train.unique()}")
   print(f"Class distribution in test set: {y_test.unique()}")

Temporal Cross-Validation
-------------------------

The toolkit provides specialized cross-validation for temporal data:

.. code-block:: python

   from elliptic_toolkit import TemporalRollingCV
   from sklearn.ensemble import RandomForestClassifier

   # Create temporal CV
   cv = TemporalRollingCV(n_splits=5)

   # Use with any scikit-learn model
   model = RandomForestClassifier(n_estimators=100)

   for train_idx, test_idx in cv.split(X_train, y_train):
         # Your training and evaluation code here
         pass

Graph Neural Networks
---------------------

For graph-based models:

.. code-block:: python

   from elliptic_toolkit import GNNBinaryClassifier
   import torch
   from torch_geometric.data import Data
   from torch_geometric.nn import GCN

   nodes_df, edges_df = process_dataset()
   data = Data(
      x=torch.tensor(nodes_df.drop(columns=['time', 'class']).values, dtype=torch.float),
      edge_index=torch.tensor(edges_df.values.T, dtype=torch.long),
      y=torch.tensor(nodes_df['class'].values, dtype=torch.long),
      time=torch.tensor(nodes_df['time'].values, dtype=torch.long)
   )
   gcn_model = GNNBinaryClassifier(
      data,
      GCN,
   )

Next Steps
----------

- Check out the :doc:`examples <examples/examples>` for more detailed usage
- Read the :doc:`API documentation <api/modules>` for complete reference
- See the GitHub repository at Homepage, https://github.com/VinceNeede/EllipticBitcoinToolkit for source code and issues
