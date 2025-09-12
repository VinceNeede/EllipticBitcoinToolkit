Installation
============

Requirements
------------

The Elliptic Bitcoin Toolkit requires the following packages:

* scikit-learn
* pandas
* numpy
* matplotlib
* torch
* torch-geometric

Install from GitHub
-------------------

You can install the latest development version directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/VinceNeede/EllipticBitcoinToolkit

Install from Source
-------------------

To install from source for development:

.. code-block:: bash

   git clone https://github.com/VinceNeede/EllipticBitcoinToolkit
   cd EllipticBitcoinToolkit
   pip install -e .

Development Installation
------------------------

For development with additional tools (testing, documentation, etc.):

.. code-block:: bash

   pip install -e ".[dev]"

Dataset Setup
-------------

The toolkit requires the Elliptic Bitcoin Dataset. You can either download it from:

https://www.kaggle.com/ellipticco/elliptic-data-set

Or use the utility function to download it programmatically:

.. code-block:: python

   from elliptic_toolkit import download_dataset
   download_dataset()

Verification
------------

To verify your installation:

.. code-block:: python

   import elliptic_toolkit
   print(elliptic_toolkit.__version__)

   # Test basic functionality
   from elliptic_toolkit import temporal_split
   import numpy as np
   
   times = np.array([1, 1, 2, 2, 3, 3])
   train_idx, test_idx = temporal_split(times, test_size=0.3)
   print("Installation successful!")

GPU Support
-----------

For GPU acceleration with PyTorch, install the appropriate CUDA version of PyTorch following the instructions at https://pytorch.org/get-started/locally/.
