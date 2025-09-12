# EllipticBitcoinToolkit

[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://vinceneede.github.io/EllipticBitcoinToolkit/)

A Python toolkit for Bitcoin transaction classification using the Elliptic dataset, developed as a university project for analyzing illicit activities in blockchain transactions.

## Overview

This project provides tools for:
- Loading and preprocessing the Elliptic Bitcoin dataset
- Training machine learning models for transaction classification
- Temporal cross-validation to handle time-dependent data
- Visualization of model performance and results

## Installation

The package hasn't been released on PyPI yet. It can be installed directly from GitHub:

```bash
pip install git+https://github.com/VinceNeede/EllipticBitcoinToolkit
```

For detailed installation instructions and requirements, see the [installation guide](https://vinceneede.github.io/EllipticBitcoinToolkit/installation.html).

## Quick Start

### Loading the Dataset

```python
from elliptic_toolkit.dataset import download_dataset, load_labeled_data

# Download the dataset (if not already available)
download_dataset()

# Load preprocessed data with temporal split
(X_train, y_train), (X_test, y_test) = load_labeled_data()
```

### Training a Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from elliptic_toolkit.model_wrappers import DropTime

# Create a pipeline that removes time features
pipe = Pipeline([
    ("drop_time", DropTime()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipe.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Dataset

This toolkit works with the Elliptic Bitcoin Dataset, which contains Bitcoin transactions labeled as illicit or licit. The dataset includes:
- Transaction features (node attributes)
- Transaction relationships (edges)  
- Time step information
- Class labels (illicit/licit/unknown)

The dataset will be automatically downloaded when using `download_dataset()`.

## Examples

See the `examples/` directory for complete usage examples:
- `random_forest.ipynb`: Complete workflow for training and evaluating a Random Forest classifier

## Project Structure

```
elliptic_toolkit/
├── dataset.py          # Data loading utilities
├── temporal_cv.py      # Time-aware cross-validation
├── model_wrappers.py   # Model utilities
├── plots.py           # Visualization functions
└── log_parser.py      # Log analysis tools
```

## Requirements

- scikit-learn
- pandas
- numpy
- matplotlib
- torch-geometric (for GNN models)

## License

This project is licensed under the MIT License.

## Author

Vincenzo Bisogno - University Project