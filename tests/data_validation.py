import pytest
import torch
from torch_geometric.data import HeteroData
from src.data_preperation import RetailHero


@pytest.fixture(scope="module")
def retailhero_data():
    dataset = RetailHero(root="../data/retailhero")
    data = dataset[0]  
    return data

def test_user_node_features_exist(retailhero_data):
    assert "x" in retailhero_data["user"], "'user' node missing 'x' features"
    assert retailhero_data["user"]["x"].dim() == 2, "'x' should be 2D"
    assert retailhero_data["user"]["x"].shape[1] >= 7, "Expected at least 7 features in 'user'"


def test_user_treatment_is_binary(retailhero_data):
    treatment = retailhero_data["user"]["t"]
    assert treatment.dtype == torch.long
    validate_treatment_binary(treatment)

def test_outcome_tensor_shape(retailhero_data):
    y = retailhero_data["user"]["y"]
    assert y.ndim == 1, "'y' outcome must be 1D"
    assert y.shape[0] == retailhero_data["user"]["x"].shape[0], "'y' length must match user count"

def test_edge_index_valid(retailhero_data):
    edge_index = retailhero_data["user", "buys", "product"]["edge_index"]
    num_users = retailhero_data["user"]["x"].shape[0]
    num_products = retailhero_data["products"]["num_products"]
    validate_edge_index(edge_index, num_users, num_products)

def test_heterodata_structure(retailhero_data):
    validate_heterodata_object(retailhero_data, "user", ("user", "buys", "product"))



def validate_edge_index(edge_index: torch.Tensor, num_nodes_src: int, num_nodes_dst: int):
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.max() >= max(num_nodes_src, num_nodes_dst):
        raise ValueError("edge_index contains node ids out of bounds")
    return True

def validate_heterodata_object(data: HeteroData, node_type: str, edge_type: tuple):
    assert isinstance(data, HeteroData), "Data must be a HeteroData object"
    assert node_type in data.node_types, f"Node type '{node_type}' not found"
    assert edge_type in data.edge_types, f"Edge type {edge_type} not found"
    return True

def validate_treatment_binary(treatment_tensor: torch.Tensor):
    unique_vals = torch.unique(treatment_tensor)
    if not torch.all((unique_vals == 0) | (unique_vals == 1)):
        raise ValueError(f"Treatment tensor contains values other than 0 or 1: {unique_vals}")
    return True