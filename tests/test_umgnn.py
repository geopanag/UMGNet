
import pytest 
import torch
import pytest
import pandas as pd
import numpy as np
from utils import run_umgnn

import torch.nn.functional as F

# Dummy loss function matching expected signature
class DummyLoss(torch.nn.Module):
    def forward(self, treatment, out_treat, out_control, outcome):
        pred = treatment * out_treat + (1 - treatment) * out_control
        return F.mse_loss(pred, outcome.float())
    

def dummy_data():
    num_users = 10
    num_products = 5
    xu = torch.rand(num_users, 3)
    xp = torch.rand(num_products, 3)
    outcome = torch.randint(0, 2, (num_users,)).float()
    treatment = torch.randint(0, 2, (num_users,))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_index_df = pd.DataFrame({
        'user': [0, 1, 2, 3],
        'T': [0, 1, 0, 1]
    })
    return {
        'outcome': outcome,
        'treatment': treatment,
        'criterion': DummyLoss(),
        'xu': xu,
        'xp': xp,
        'edge_index': edge_index,
        'edge_index_df': edge_index_df,
        'task': 0,
        'n_hidden': 4,
        'out_channels': 1,
        'no_layers': 1,
        'k': 2,
        'run': 0,
        'model_file': 'temp_model.pt',
        'num_users': num_users,
        'num_products': num_products,
        'with_lp': False,
        'alpha': 0.5,
        'l2_reg': 1e-4,
        'dropout': 0.1,
        'lr': 1e-3,
        'num_epochs': 3,
        'early_thres': 2,
        'repr_balance': False,
        'device': torch.device('cpu'),
        'validation_fraction': 2,
    }

@pytest.fixture(scope="module")
def test_run_umgnn_runs(dummy_data):
    result = run_umgnn(**dummy_data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert not np.any(np.isnan(result)), "Output contains NaNs"