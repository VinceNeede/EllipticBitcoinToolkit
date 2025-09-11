import pytest

import numpy as np
import pandas as pd

import torch
from torch_geometric.nn import GCN, GAT, PairNorm
from torch_geometric.data import Data

from elliptic_toolkit.model_wrappers import DropTime, MLPWrapper, _get_norm_arg, GNNBinaryClassifier

class TestDropTime:
    def test_transform(self):
        dt = DropTime()
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'time': [0, 1, 2]
        })
        df_transformed = dt.transform(df)
        assert 'time' not in df_transformed.columns
        assert df_transformed.shape == (3, 2)

        dt_no_drop = DropTime(drop=False)
        df_no_drop = dt_no_drop.transform(df)
        assert all(df == df_no_drop)
        
class TestMLPWrapper:
                
    def test_initialization(self):
        mlp = MLPWrapper(num_layers=2, hidden_dim=16)
        assert mlp.hidden_layer_sizes == (16, 16)
        assert mlp.shuffle is False
        assert mlp.early_stopping is False

    def test_set_params(self):
        mlp = MLPWrapper()
        mlp.set_params(hidden_dim=32, num_layers=3, alpha=0.01)
        assert mlp.hidden_layer_sizes == (32, 32, 32)
        assert mlp.alpha == 0.01

class TestGNNBinaryClassifier:
    
    _dataset = Data(
        x=torch.randn(100, 16),
        edge_index=torch.randint(0, 100, (2, 300)),
        y=torch.randint(0, 2, (100,)),
        time=torch.randint(0, 5, (100,))
    ).sort_by_time()
    
    _train_idx = torch.arange(70)
    _test_idx = torch.arange(70, 100)
    
    def test_norm_resolution(self):    
        norm, norm_kwargs = _get_norm_arg(None)
        assert norm is None and norm_kwargs == {}
        
        norm, norm_kwargs = _get_norm_arg('batch')
        assert norm == 'batch' and norm_kwargs == {}
        
        norm, norm_kwargs = _get_norm_arg('layer')
        assert norm == 'layer' and norm_kwargs == {}

        norm, norm_kwargs = _get_norm_arg('pair')
        assert norm == 'pair' and norm_kwargs == {}
        
        norm, norm_kwargs = _get_norm_arg('pair_scale=0.5')
        assert isinstance(norm, PairNorm) and norm_kwargs == {} and norm.scale == 0.5
        
        norm, norm_kwargs = _get_norm_arg('layer_mode=node')
        assert norm == 'layer' and norm_kwargs == {'mode': 'node'}

    
    def test_initialization(self):
        
        # heads is GAT specific
        with pytest.warns(UserWarning):
            gnn = GNNBinaryClassifier(
                self._dataset,
                GCN,
                heads=2,
                )
        
        # test missing attributes in Data
        with pytest.raises(ValueError):
            data = self._dataset.clone()
            data.y = None
            gnn = GNNBinaryClassifier(
                data,
                GCN,
                device='cpu',
                )
        with pytest.raises(ValueError):
            data = self._dataset.clone()
            data.x = None
            gnn = GNNBinaryClassifier(
                data,
                GCN,
                device='cpu',
                )
        with pytest.raises(ValueError):
            data = self._dataset.clone()
            data.edge_index = None
            gnn = GNNBinaryClassifier(
                data,
                GCN,
                device='cpu',
                )
        
        # test device handling
        gnn = GNNBinaryClassifier(
            self._dataset,
            GCN,
            device='cpu',
            )
        
        assert gnn.device_ == torch.device('cpu')
        
        with pytest.raises(ValueError):
            gnn = GNNBinaryClassifier(
                self._dataset,
                GCN,
                device='invalid_device',
                )
        
        if torch.cuda.is_available():
            gnn = GNNBinaryClassifier(
                self._dataset,
                GCN,
                device='cuda',
                )
            assert gnn.device_ == torch.device('cuda')
        else:
            with pytest.warns(UserWarning):
                gnn = GNNBinaryClassifier(
                    self._dataset,
                    GCN,
                    device='cuda',
                    )
                assert gnn.device == torch.device('cpu')
        
        # test raises if predict called before fitting
        with pytest.raises(ValueError):
            gnn = GNNBinaryClassifier(
                self._dataset,
                GCN,
                device='cpu',
                )
            # this should raise since the model is not yet fitted
            gnn.predict_proba(self._test_idx)

    def test_fitting_and_prediction(self):
        """Test that predictions output correct shapes and types."""
        gnn = GNNBinaryClassifier(
            self._dataset,
            GCN,
            device='cpu',
            max_iter=2,  # keep max_iter low for testing
            )
        
        with pytest.warns(UserWarning):
            gnn.fit(self._train_idx)
        
        probs = gnn.predict_proba(self._test_idx)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (len(self._test_idx), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        
        preds = gnn.predict(self._test_idx)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(self._test_idx),)
        assert set(preds).issubset({0, 1})
        
        # Test with GPU if available
        if torch.cuda.is_available():
            gnn_gpu = GNNBinaryClassifier(
                self._dataset,
                GCN,
                device='cuda',
                max_iter=2,
                )
            with pytest.warns(UserWarning):
                gnn_gpu.fit(self._train_idx)
            
            probs_gpu = gnn_gpu.predict_proba(self._test_idx)
            assert isinstance(probs_gpu, np.ndarray)
            assert probs_gpu.shape == (len(self._test_idx), 2)
            assert np.allclose(probs_gpu.sum(axis=1), 1.0)
            
            preds_gpu = gnn_gpu.predict(self._test_idx)
            assert isinstance(preds_gpu, np.ndarray)
            assert preds_gpu.shape == (len(self._test_idx),)
            assert set(preds_gpu).issubset({0, 1})
            
    def test_reproducibility(self):
        """Test that setting the random state leads to reproducible results."""
        expected_res = np.array([[0.47748172, 0.5225183 ],
       [0.484989  , 0.515011  ],
       [0.4724241 , 0.5275759 ],
       [0.48015517, 0.51984483],
       [0.4806236 , 0.5193764 ],
       [0.47834802, 0.521652  ],
       [0.4765643 , 0.5234357 ],
       [0.47962344, 0.52037656],
       [0.48003882, 0.5199612 ],
       [0.4328552 , 0.5671448 ],
       [0.46638393, 0.53361607],
       [0.4741711 , 0.5258289 ],
       [0.4856978 , 0.5143022 ],
       [0.47291553, 0.52708447],
       [0.48241138, 0.5175886 ],
       [0.4681645 , 0.5318355 ],
       [0.4726112 , 0.5273888 ],
       [0.4699164 , 0.5300836 ],
       [0.47179937, 0.5282006 ],
       [0.42476243, 0.5752376 ],
       [0.48048282, 0.5195172 ],
       [0.47941303, 0.52058697],
       [0.47677463, 0.52322537],
       [0.4737966 , 0.5262034 ],
       [0.4719388 , 0.5280612 ],
       [0.47255802, 0.527442  ],
       [0.48122054, 0.51877946],
       [0.4792043 , 0.5207957 ],
       [0.46747237, 0.5325276 ],
       [0.46316522, 0.5368348 ]], dtype=np.float32)
        torch.manual_seed(42)
        dataset = Data(
            x=torch.randn(100, 16),
            edge_index=torch.randint(0, 100, (2, 300)),
            y=torch.randint(0, 2, (100,)),
            time=torch.randint(0, 5, (100,))
        ).sort_by_time()
        train_idx = torch.arange(70)
        test_idx = torch.arange(70, 100)

        gnn1 = GNNBinaryClassifier(
            dataset,
            GCN,
            device='cpu',
            max_iter=2,
            )

        with pytest.warns(UserWarning):
            gnn1.fit(train_idx)
        preds1 = gnn1.predict_proba(test_idx)

        assert np.allclose(preds1, expected_res)
        
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("error")
    import sys
    import traceback
    try:
        # Call the specific test directly
        TestGNNBinaryClassifier().test_fitting_and_prediction()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
