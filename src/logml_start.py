#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import json
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from models import BipartiteSAGE2mod
from utils import (
    outcome_regression_loss_l1,
    outcome_regression_loss,
    uplift_score,
    set_seed,
)
import torch.nn.functional as F

from typing import Callable
import torch_geometric as pyg
from torch.optim import Optimizer

from torch.optim import Adam


from causalml.inference.meta import (
    BaseXClassifier,
    BaseSClassifier,
    BaseTClassifier,
    BaseRClassifier,
    BaseDRRegressor,
    BaseXRegressor,
    BaseSRegressor,
    BaseTRegressor,
    BaseRRegressor,
)
from causalml.propensity import ElasticNetPropensityModel
from sklearn.linear_model import LogisticRegression
from causalml.inference.tree import UpliftTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from causalml.inference.tree.causal.causaltree import CausalTreeRegressor


def train(
    mask: np.ndarray,
    model: torch.nn.Module,
    xu: torch.tensor,
    xp: torch.tensor,
    edge_index: torch.tensor,
    treatment: torch.tensor,
    outcome: torch.tensor,
    optimizer: Optimizer,
    criterion: Callable[
        [torch.tensor, torch.tensor, torch.tensor, torch.tensor], torch.tensor
    ],
) -> torch.tensor:
    """
    Trains the model for one epoch.
    """
    model.train()
    optimizer.zero_grad()  # Clear gradients.

    pred_t, pred_c, hidden_treatment, hidden_control = model(xu, xp, edge_index)
    loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])

    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test(
    mask: np.ndarray,
    model: torch.nn.Module,
    xu: torch.tensor,
    xp: torch.tensor,
    edge_index: torch.tensor,
    treatment: torch.tensor,
    outcome: torch.tensor,
    criterion: Callable[
        [torch.tensor, torch.tensor, torch.tensor, torch.tensor], torch.tensor
    ],
) -> torch.tensor:
    """
    Tests the model.
    """
    model.eval()
    pred_t, pred_c, hidden_treatment, hidden_control = model(xu, xp, edge_index)
    loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])
    return loss


def experiment(
    model: torch.nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    edge_index: torch.tensor,
    treatment: torch.tensor,
    outcome: torch.tensor,
    xu: torch.tensor,
    xp: torch.tensor,
    model_file: str,
    print_per_epoch: int,
    patience: int,
    criterion: Callable[
        [torch.tensor, torch.tensor, torch.tensor, torch.tensor], torch.tensor
    ],
) -> (list, list):
    """
    Trains the model for num_epochs epochs and returns the train and validation losses.
    """
    early_stopping = 0
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    print_per_epoch = 50
    for epoch in range(num_epochs):
        train_loss = train(
            train_indices,
            model,
            xu,
            xp,
            edge_index,
            treatment,
            outcome,
            optimizer,
            criterion,
        )
        val_loss = test(
            val_indices, model, xu, xp, edge_index, treatment, outcome, criterion
        )

        train_losses.append(float(train_loss.item()))
        val_losses.append(float(val_loss.item()))

        if val_loss < best_val_loss:
            early_stopping = 0
            best_val_loss = val_loss
            torch.save(model, model_file)
        else:
            early_stopping += 1
            if early_stopping > patience:
                print("early stopping..")
                break

        if epoch % print_per_epoch == 0:
            print(
                f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val loss: {val_loss:.4f}"
            )

    return train_losses, val_losses


def evaluate(
    model: torch.nn.Module,
    test_indices: np.ndarray,
    treatment: torch.tensor,
    outcome: torch.tensor,
    xu: torch.tensor,
    xp: torch.tensor,
    edge_index: torch.tensor,
    criterion,
) -> (float, float, float):
    """
    Evaluates the model on the test set.
    """

    model.eval()

    mask = test_indices
    pred_t, pred_c, hidden_treatment, hidden_control = model(xu, xp, edge_index)

    test_loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])

    treatment_test = treatment[test_indices].detach().cpu().numpy()
    outcome_test = outcome[test_indices].detach().cpu().numpy()
    pred_t = pred_t.detach().cpu().numpy()
    pred_c = pred_c.detach().cpu().numpy()

    uplift = pred_t[test_indices] - pred_c[test_indices]
    uplift = uplift.squeeze()

    up40 = uplift_score(uplift, treatment_test, outcome_test, 0.4)
    up20 = uplift_score(uplift, treatment_test, outcome_test, 0.2)
    return up40, up20, test_loss


def main():

    os.chdir("/home/georgios/Desktop/research/causality/experiment/code/UMGNet")
    data = torch.load("../../data/retailhero/processed/data.pt")[0]

    no_layers = 1
    out_channels = 1
    num_epochs = 300
    results_file_name = "results.csv"
    model_file = "../../models/bipartite_sage.pt"
    early_thres = 50
    validation_fraction = 5
    patience = 50
    print_per_epoch = 50

    seed = 1  # 42
    n_hidden = 32  # 64
    l2_reg = 5e-4
    dropout = 0.2  # 0.2
    k = 10  # 5
    lr = 0.01

    criterion = outcome_regression_loss_l1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xp = torch.eye(data["products"]["num_products"]).to(device)
    xu = data["user"]["x"].to(device)
    treatment = data["user"]["t"].to(device)
    outcome = data["user"]["y"].to(device)

    set_seed(seed)

    kf = KFold(n_splits=abs(k), shuffle=True, random_state=seed)

    # make a data frame to gather the results
    results = []

    for train_indices, test_indices in kf.split(xu):
        test_indices, train_indices = train_indices, test_indices
        torch.cuda.empty_cache()

        # split the test indices to test and validation
        val_indices = train_indices[: int(len(train_indices) / validation_fraction)]
        train_indices = train_indices[int(len(train_indices) / validation_fraction) :]

        ## Keep the graph before the treatment and ONLY the edges of the the train nodes (i.e. after the treatment)
        mask = torch.isin(
            data["user", "buys", "product"]["edge_index"][0, :],
            torch.tensor(train_indices),
        )
        edge_index_up_current = data["user", "buys", "product"]["edge_index"][
            :, (~data["user", "buys", "product"]["treatment"]) | (mask)
        ]

        edge_index_up_current[1] = edge_index_up_current[1] + xu.shape[0]

        edge_index_up_current = torch.cat(
            [edge_index_up_current, edge_index_up_current.flip(dims=[0])], dim=1
        ).to(device)

        model = BipartiteSAGE2mod(
            xu.shape[1], xp.shape[1], n_hidden, out_channels, no_layers, dropout
        ).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

        out = model(xu, xp, edge_index_up_current)  # init params

        train_losses, val_losses = experiment(
            model,
            optimizer,
            num_epochs,
            train_indices,
            val_indices,
            edge_index_up_current,
            treatment,
            outcome,
            xu,
            xp,
            model_file,
            print_per_epoch,
            patience,
            criterion,
        )

        model = torch.load(model_file).to(device)
        up40, up20, test_loss = evaluate(
            model,
            test_indices,
            treatment,
            outcome,
            xu,
            xp,
            edge_index_up_current,
            criterion,
        )

        print(
            f"mse {test_loss:.4f} with avg abs value {torch.mean(torch.abs(outcome[test_indices]))}"
        )
        print(f"up40 {up40:.4f}")
        print(f"up20 {up20:.4f}")

        result_row = []
        # result_row.append(test_loss)
        result_row.append(up40)
        result_row.append(up20)

        # Benchmarks
        train_indices = np.hstack([train_indices, val_indices])

        outcome_np = outcome.cpu().numpy()
        xu_np = xu.cpu().numpy()
        treatment_np = treatment.cpu().numpy()

        learner = BaseTRegressor(learner=XGBRegressor())

        learner.fit(
            X=xu_np[train_indices],
            y=outcome_np[train_indices],
            treatment=treatment_np[train_indices],
        )
        uplift = learner.predict(
            X=xu_np[train_indices], treatment=treatment_np[test_indices]
        ).squeeze()

        uplift = learner.predict(
            X=xu_np[test_indices], treatment=treatment_np[test_indices]
        ).squeeze()

        up40 = uplift_score(
            uplift,
            np.hstack([treatment_np[train_indices], treatment_np[test_indices]]),
            np.hstack([outcome_np[train_indices], outcome_np[test_indices]]),
            rate=0.4,
        )
        up20 = uplift_score(
            uplift,
            np.hstack([treatment_np[train_indices], treatment_np[test_indices]]),
            np.hstack([outcome_np[train_indices], outcome_np[test_indices]]),
            rate=0.2,
        )

        print(f"T-learner up40: {up40:.4f} , up20: {up20:.4f}")
        result_row.append(up40)
        result_row.append(up20)

        propensity_model = ElasticNetPropensityModel()

        propensity_model.fit(X=xu_np[train_indices], y=treatment_np[train_indices])
        p_train = propensity_model.predict(X=xu_np[train_indices])
        p_test = propensity_model.predict(X=xu_np[test_indices])

        learner = BaseXRegressor(learner=XGBRegressor())

        learner.fit(
            X=xu_np[train_indices],
            y=outcome_np[train_indices],
            treatment=treatment_np[train_indices],
            p=p_train,
        )

        uplift = learner.predict(
            X=xu_np[test_indices], treatment=treatment_np[test_indices], p=p_test
        ).squeeze()

        up40 = uplift_score(
            uplift,
            np.hstack([treatment_np[train_indices], treatment_np[test_indices]]),
            np.hstack([outcome_np[train_indices], outcome_np[test_indices]]),
            rate=0.4,
        )
        up20 = uplift_score(
            uplift,
            np.hstack([treatment_np[train_indices], treatment_np[test_indices]]),
            np.hstack([outcome_np[train_indices], outcome_np[test_indices]]),
            rate=0.2,
        )

        print(f"X-learner up40: {up40:.4f} , up20: {up20:.4f}")
        result_row.append(up40)
        result_row.append(up20)

        learner = CausalTreeRegressor(control_name="0")
        X_train = np.hstack(
            (treatment_np[train_indices].reshape(-1, 1), xu_np[train_indices])
        )
        X_test = np.hstack(
            (treatment_np[test_indices].reshape(-1, 1), xu_np[test_indices])
        )
        learner.fit(
            X=X_train,
            treatment=treatment_np[train_indices].astype(str),
            y=outcome_np[train_indices],
        )
        uplift = learner.predict(X=X_test).squeeze()

        up40 = uplift_score(
            uplift,
            np.hstack([treatment_np[train_indices], treatment_np[test_indices]]),
            np.hstack([outcome_np[train_indices], outcome_np[test_indices]]),
            rate=0.4,
        )
        up20 = uplift_score(
            uplift,
            np.hstack([treatment_np[train_indices], treatment_np[test_indices]]),
            np.hstack([outcome_np[train_indices], outcome_np[test_indices]]),
            rate=0.2,
        )

        print(f"Tree up40: {up40:.4f} , up20: {up20:.4f}")
        result_row.append(up40)
        result_row.append(up20)

        results.append(result_row)

    results = pd.DataFrame(
        results,
        columns=[
            "up40_gnn",
            "up20_gnn",
            "up40_t",
            "up20_t",
            "up40_x",
            "up20_x",
            "up40_tree",
            "up20_tree",
        ],
    )

    results.to_csv(results_file_name)

    # print mean and sd of the results
    print(results)
    print(results.mean())
    print(results.std())


if __name__ == "__main__":
    main()
