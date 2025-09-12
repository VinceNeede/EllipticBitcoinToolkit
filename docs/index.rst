Elliptic Bitcoin Toolkit Documentation
======================================

Welcome to the Elliptic Bitcoin Toolkit documentation! This comprehensive toolkit provides utilities for working with the Elliptic Bitcoin dataset, including data loading, temporal analysis, graph neural network modeling, and evaluation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/modules
   examples/examples

Overview
--------

The Elliptic Bitcoin Toolkit is designed to make working with the Elliptic Bitcoin dataset easier and more efficient. It provides:

* **Data Processing**: Load and preprocess the dataset with built-in validation
* **Temporal Analysis**: Time-aware cross-validation and data splitting
* **Graph Neural Networks**: Scikit-learn compatible GNN classifiers
* **Model Wrappers**: Enhanced model wrappers with flexible architecture
* **Visualization**: Comprehensive plotting utilities
* **Log Parsing**: Parse and analyze hyperparameter search results

Key Features
------------

Data Loading and Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The toolkit provides robust data loading capabilities with automatic validation:

.. code-block:: python

   from elliptic_toolkit import process_dataset, load_labeled_data
   
   # Load only labeled data with temporal split
   (X_train, y_train), (X_test, y_test) = load_labeled_data(test_size=0.2, root='path_to_dataset')

Temporal Cross-Validation
^^^^^^^^^^^^^^^^^^^^^^^^^

Time-aware cross-validation ensures proper temporal ordering:

.. code-block:: python

   from elliptic_toolkit import TemporalRollingCV
   from sklearn.model_selection import GridSearchCV
   
   cv = TemporalRollingCV(n_splits=5, test_size=10)
   grid_search = GridSearchCV(model, param_grid, cv=cv)

Graph Neural Networks
^^^^^^^^^^^^^^^^^^^^^

Scikit-learn compatible GNN classifiers with early stopping:

.. code-block:: python

   from elliptic_toolkit import GNNBinaryClassifier
   from torch_geometric.nn import GAT
   
   gnn = GNNBinaryClassifier(
       data=graph_data,
       model=GAT,
       hidden_dim=64,
       num_layers=3,
       heads=8
   )
   gnn.fit(train_indices)

Visualization and Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

Comprehensive plotting utilities for model evaluation:

.. code-block:: python

   from elliptic_toolkit import plot_evals, plot_marginals
   
   # Plot model evaluation
   pr_fig, temporal_fig = plot_evals(model, X_test, y_test, y_train)
   
   # Plot hyperparameter marginals
   marginal_figs = plot_marginals(grid_search.cv_results_)

Indices and Tables
==================

* :ref:`search`