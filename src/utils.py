from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import KFold
import torch
import numpy as np

from typing import Union

import pandas as pd

from evaluate import uplift_score

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
from causalml.inference.tf import DragonNet
from causalml.inference.tf.utils import regression_loss
from causalml.inference.tree import UpliftTreeClassifier
from causalml.inference.tree.causal.causaltree import CausalTreeRegressor
from causalml.inference.nn import CEVAE
from causalml.propensity import ElasticNetPropensityModel

import random 

from sklearn.linear_model import LogisticRegression
from typing import Callable
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn


from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
from sklearn.cluster import KMeans

import sys
from models import BipartiteSAGE2mod, UserMP


def run_umgnn(
    outcome: torch.tensor,
    treatment: torch.tensor,
    criterion: torch.nn.modules.loss._Loss,
    xu: torch.tensor,
    xp: torch.tensor,
    edge_index: torch.tensor,
    edge_index_df: pd.DataFrame,
    task: int,
    n_hidden: int,
    out_channels: int,
    no_layers: int,
    k: int,
    run: int,
    model_file: str,
    num_users: int,
    num_products: int,
    with_lp: bool,
    alpha: float,
    l2_reg: float,
    dropout: float,
    lr: float,
    num_epochs: int,
    early_thres: int,
    repr_balance: bool,
    device: torch.device,
    validation_fraction: int = 5,
) -> np.ndarray:

    # ------ K fold split
    kf = KFold(n_splits=abs(k), shuffle=True, random_state=run)
    result_fold = []
    if with_lp:
        dummy_product_labels = torch.zeros([num_products, 1]).to(device).squeeze()

    for train_indices, test_indices in kf.split(xu):
        test_indices, train_indices = train_indices, test_indices

        # split the test indices to test and validation
        val_indices = train_indices[: int(len(train_indices) / validation_fraction)]
        subtrain_indices = train_indices[
            int(len(train_indices) / validation_fraction) :
        ]

        ## Keep the graph before the treatment and ONLY the edges of the the train nodes (i.e. after the treatment)
        # remove edge_index_df[ edge_index_df['user'].isin(train_indices)  if you dont want edges from train set
        edge_index_up_current = edge_index[
            :,
            edge_index_df[
                edge_index_df["user"].isin(subtrain_indices) | edge_index_df["T"] == 0
            ].index.values,
        ]
        # make unsupervised and add num_nodes for bipartite message passing
        edge_index_up_current[1] = edge_index_up_current[1] + num_users
        edge_index_up_current = torch.cat(
            [edge_index_up_current, edge_index_up_current.flip(dims=[0])], dim=1
        )

        ###------------------------------------------------------------ Label propagation
        ## Each user will have an estimate of its neighbors train labels (but not its own), mainly to assist semi-supervised learning
        if with_lp:
            label_for_prop = make_outcome_feature(
                xu, train_indices, outcome.type(torch.LongTensor)
            ).to(device)

            label_for_prop = torch.cat([label_for_prop, dummy_product_labels], dim=0)
            model = UserMP().to(device)
            label_for_prop = model(label_for_prop, edge_index_up_current)
            label_for_prop = model(label_for_prop, edge_index_up_current)
            # print(label_for_prop.shape)
            label_for_prop = label_for_prop[:num_users].detach().to(device)
            mean = torch.mean(label_for_prop)
            std_dev = torch.std(label_for_prop)

            # Standardize the vector
            label_for_prop = (label_for_prop - mean) / std_dev

            xu_ = torch.cat([xu, label_for_prop.unsqueeze(1)], dim=1)
        else:
            xu_ = xu

        # ---------------------------------------------------------- Model and Optimizer
        # xu_ : user embeddings e.g. sex, age, coupon issue time etc.
        # xp: product embeddings one-hot encoding
        model = BipartiteSAGE2mod(
            xu_.shape[1], xp.shape[1], n_hidden, out_channels, no_layers, dropout
        ).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

        # init params
        out = model(xu_, xp, edge_index_up_current)

        train_losses = []
        val_losses = []
        best_val_loss = np.inf
        early_stopping = 0

        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()

            out_treatment, out_control, hidden_treatment, hidden_control = model(
                xu_, xp, edge_index_up_current
            )

            loss = criterion(
                treatment[subtrain_indices],
                out_treatment[subtrain_indices],
                out_control[subtrain_indices],
                outcome[subtrain_indices],
            )  # target_labels are your binary labels

            # loss = criterion(out[train_indices], y[train_indices]) # target_labels are your binary labels
            loss.backward()
            optimizer.step()
            total_loss = float(loss.item())

            train_losses.append(total_loss)
            # test validation loss
            if epoch % 5 == 0:
                with torch.no_grad():
                    # no dropout hence no need to rerun
                    model.eval()
                    out_treatment, out_control, hidden_treatment, hidden_control = (
                        model(xu_, xp, edge_index_up_current)
                    )

                    if task == 0:
                        out_treatment = F.sigmoid(out_treatment)
                        out_control = F.sigmoid(out_control)

                    loss = criterion(
                        treatment[val_indices],
                        out_treatment[val_indices],
                        out_control[val_indices],
                        outcome[val_indices],
                    )

                    val_loss = round(float(loss.item()), 3)
                    val_losses.append(val_loss)

                    if val_loss < best_val_loss:
                        early_stopping = 0
                        best_val_loss = val_loss
                        torch.save(model, model_file)
                    else:
                        early_stopping += 1
                        if early_stopping > early_thres:
                            print("early stopping..")
                            break

                    model.train()

                if epoch % 10 == 0:
                    print(train_losses[-1])
                    print(val_losses[-10:])

        model = torch.load(model_file).to(device)
        model.eval()

        out_treatment, out_control, hidden_treatment, hidden_control = model(
            xu_, xp, edge_index_up_current
        )

        if task == 0:
            out_treatment = F.sigmoid(out_treatment)
            out_control = F.sigmoid(out_control)

        # ------------------------ Evaluating
        treatment_test = treatment[test_indices].detach().cpu().numpy()
        outcome_test = outcome[test_indices].detach().cpu().numpy()
        out_treatment = out_treatment.detach().cpu().numpy()
        out_control = out_control.detach().cpu().numpy()

        uplift = out_treatment[test_indices] - out_control[test_indices]
        uplift = uplift.squeeze()

        mse = (
            uplift.mean()
            - (
                outcome_test[treatment_test == 1].mean()
                - outcome_test[treatment_test == 0].mean()
            )
        ) ** 2
        print(f"mse {mse}")
        up40 = uplift_score(uplift, treatment_test, outcome_test, 0.4)
        print(f"up40 {up40}")
        up20 = uplift_score(uplift, treatment_test, outcome_test, 0.2)
        print(f"up20 {up20}")

        result_fold.append((up40, up20))

    return pd.DataFrame(result_fold).mean().values



def causalml_run(
    confounders_train: np.ndarray,
    outcome_train: np.ndarray,
    treatment_train: np.ndarray,
    confounders_test: np.ndarray,
    outcome_test: np.ndarray,
    treatment_test: np.ndarray,
    task: int = 0,
    causal_model_type: str = "X",
    model_class: str = "XGB",
    model_regr: str = "XGBR",
    total_uplift=False,
) -> tuple:
    """
    Run the causalml model on the train and test data using the given models for causal ml, propensity, outcome and effect.
    """
    dic_mod = {"XGB": XGBClassifier, "LR": LogisticRegression, "XGBR": XGBRegressor}

    if causal_model_type == "S":
        if task == 0:
            learner = BaseSClassifier(learner=dic_mod[model_class]())
        else:
            learner = BaseSRegressor(learner=dic_mod[model_regr]())

        learner.fit(X=confounders_train, y=outcome_train, treatment=treatment_train)

        if total_uplift:
            uplift = learner.predict(
                X=np.vstack([confounders_train, confounders_test]),
                treatment=np.hstack([treatment_train, treatment_test]),
            ).squeeze()
        else:
            uplift = learner.predict(
                X=confounders_test, treatment=treatment_test
            ).squeeze()

    elif causal_model_type == "T":
        if task == 0:
            learner = BaseTClassifier(learner=dic_mod[model_class]())
        else:
            learner = BaseTRegressor(learner=dic_mod[model_regr]())

        learner.fit(X=confounders_train, y=outcome_train, treatment=treatment_train)
        uplift = learner.predict(X=confounders_test, treatment=treatment_test).squeeze()
        if total_uplift:
            uplift = learner.predict(
                X=np.vstack([confounders_train, confounders_test]),
                treatment=np.hstack([treatment_train, treatment_test]),
            ).squeeze()
        else:
            uplift = learner.predict(
                X=confounders_test, treatment=treatment_test
            ).squeeze()

    elif causal_model_type == "X":
        propensity_model = ElasticNetPropensityModel()  # dic_mod[model_class]()

        propensity_model.fit(X=confounders_train, y=treatment_train)
        p_train = propensity_model.predict(X=confounders_train)
        p_test = propensity_model.predict(X=confounders_test)

        if task == 0:
            learner = BaseXClassifier(
                outcome_learner=dic_mod[model_class](),
                effect_learner=dic_mod[model_regr](),
            )
        else:
            learner = BaseXRegressor(learner=dic_mod[model_regr]())

        learner.fit(
            X=confounders_train, y=outcome_train, treatment=treatment_train, p=p_train
        )

        if total_uplift:
            uplift = learner.predict(
                X=np.vstack([confounders_train, confounders_test]),
                treatment=np.hstack([treatment_train, treatment_test]),
                p=np.hstack([p_train, p_test]),
            ).squeeze()
        else:
            uplift = learner.predict(
                X=confounders_test, treatment=treatment_test, p=p_test
            ).squeeze()

    elif causal_model_type == "R":
        propensity_model = ElasticNetPropensityModel()  # dic_mod[model_class]()

        propensity_model.fit(X=confounders_train, y=treatment_train)
        p_train = propensity_model.predict(X=confounders_train)
        p_test = propensity_model.predict(X=confounders_test)

        if task == 0:
            learner = BaseRClassifier(
                outcome_learner=dic_mod[model_class](),
                effect_learner=dic_mod[model_regr](),
            )
        else:
            learner = BaseRRegressor(learner=dic_mod[model_regr]())

        learner.fit(
            X=confounders_train, y=outcome_train, treatment=treatment_train, p=p_train
        )

        if total_uplift:
            uplift = learner.predict(
                X=np.vstack([confounders_train, confounders_test]),
                treatment=np.hstack([treatment_train, treatment_test]),
                p=np.hstack([p_train, p_test]),
            ).squeeze()
        else:
            uplift = learner.predict(X=confounders_test).squeeze()

    elif causal_model_type == "D":
        propensity_model = ElasticNetPropensityModel()  # dic_mod[model_class]()

        propensity_model.fit(X=confounders_train, y=treatment_train)
        p_train = propensity_model.predict(X=confounders_train)
        p_test = propensity_model.predict(X=confounders_test)

        if task == 0:
            learner = BaseDRRegressor(
                learner=dic_mod[model_class](),
                treatment_effect_learner=dic_mod[model_regr](),
            )
        else:
            learner = BaseDRRegressor(
                learner=dic_mod[model_regr](),
                treatment_effect_learner=dic_mod[model_regr](),
            )

        learner.fit(
            X=confounders_train, y=outcome_train, treatment=treatment_train, p=p_train
        )

        if total_uplift:
            uplift = learner.predict(
                X=np.vstack([confounders_train, confounders_test]),
                treatment=np.hstack([treatment_train, treatment_test]),
                p=np.hstack([p_train, p_test]),
            ).squeeze()
        else:
            uplift = learner.predict(
                X=confounders_test, treatment=treatment_test, p=p_test
            ).squeeze()

    elif causal_model_type == "Tree":
        if task == 0:
            learner = UpliftTreeClassifier(control_name="0")
        else:
            learner = CausalTreeRegressor(control_name="0")
        X_train = np.hstack((treatment_train.reshape(-1, 1), confounders_train))
        X_test = np.hstack((treatment_test.reshape(-1, 1), confounders_test))
        learner.fit(X=X_train, treatment=treatment_train.astype(str), y=outcome_train)

        if total_uplift:
            uplift = learner.predict(X=np.vstack([X_train, X_test]).squeeze())
            uplift = uplift.argmax(1)
        else:
            uplift = learner.predict(X=X_test).squeeze()

    elif causal_model_type == "Dragon":
        if task == 0:
            learner = DragonNet()
        else:
            learner = DragonNet(loss_func=regression_loss)
        learner.fit(
            X=confounders_train,
            treatment=treatment_train,
            y=outcome_train.astype(np.float32),
        )
        if total_uplift:
            uplift = learner.predict(
                X=np.vstack([confounders_train, confounders_test]),
                treatment=np.hstack([treatment_train, treatment_test]),
            )
        else:
            uplift = learner.predict(X=confounders_test, treatment=treatment_test)
        uplift = uplift[:, 1] - uplift[:, 0]

    elif causal_model_type == "CEVAE":
        if task == 0:
            learner = CEVAE()
        else:
            learner = CEVAE()
        learner.fit(
            X=torch.tensor(confounders_train, dtype=torch.float),
            treatment=torch.tensor(treatment_train, dtype=torch.float),
            y=torch.tensor(outcome_train, dtype=torch.float),
        )

        if total_uplift:
            uplift = learner.predict(
                X=np.vstack([confounders_train, confounders_test]),
                treatment=np.hstack([treatment_train, treatment_test]),
            )
        else:
            uplift = learner.predict(X=confounders_test, treatment=treatment_test)

    if total_uplift:
        score40 = uplift_score(
            uplift,
            np.hstack([treatment_train, treatment_test]),
            np.hstack([outcome_train, outcome_test]),
            rate=0.4,
        )  # uplift_at_k(y_true = outcome_test, uplift=uplift, treatment= treatment_test, strategy='by_group', k=0.4)
        score20 = uplift_score(
            uplift,
            np.hstack([treatment_train, treatment_test]),
            np.hstack([outcome_train, outcome_test]),
            rate=0.2,
        )  # uplift_at_k(y_true = outcome_test, uplift=uplift, treatment= treatment_test, strategy='by_group', k=0.2)

    else:
        score40 = uplift_score(
            uplift, treatment_test, outcome_test, rate=0.4
        )  # uplift_at_k(y_true = outcome_test, uplift=uplift, treatment= treatment_test, strategy='by_group', k=0.4)
        score20 = uplift_score(
            uplift, treatment_test, outcome_test, rate=0.2
        )  # uplift_at_k(y_true = outcome_test, uplift=uplift, treatment= treatment_test, strategy='by_group', k=0.2)
    return score40, score20

def al_lp(
    test_indices: np.ndarray,
    degree: np.ndarray,
    pred_uncertainty: np.ndarray,
    cluster_distance: np.ndarray,
    cluster_assigment: np.ndarray,
    cluster_budget: dict,
    treatment: np.ndarray,
    sample_budget: int,
    a1: float,
    a2: float,
    a3: float,
):
    """
    Solve the active learning linear program with a greedy approach, sorting to maximize the objective function and adding samples that do not violate constraints
    """
    if len(pred_uncertainty) > 0:
        objective = (
            a1 * pred_uncertainty
            + a2 * degree[test_indices]
            + a3 * cluster_distance[test_indices]
        )
    else:
        objective = a2 * degree[test_indices] + a3 * cluster_distance[test_indices]

    new_train_indices = []
    treatment_budget = np.ceil(sample_budget / 2)

    # sorted indices contains the ABSOLUTE indices as stored in the test indices i.e. they point to xu, not relative indices that point to the test_indices vector
    sorted_indices = test_indices[np.argsort(objective)]

    # hence cluster_assignment and treatment are not subset to test_indices, but stay in dimension n which is where test indices point to

    # test the treatment and cluster budgets
    for node in sorted_indices:
        if cluster_budget[cluster_assigment[node]] >= 0:
            if treatment_budget - treatment[node] < 0:
                # can not add more treated subjects
                continue

            cluster_budget[cluster_assigment[node]] -= 1
            treatment_budget = treatment_budget - treatment[node]
            new_train_indices.append(node)
        else:
            # can not add more subjects from this cluster
            continue

    return new_train_indices




def binary_treatment_loss(t_true: torch.tensor, t_pred: torch.tensor) -> torch.tensor:
    """
    Compute cross entropy for propensity score , from Dragonnet
    """
    t_pred = (t_pred + 0.001) / 1.002

    return torch.mean(F.binary_cross_entropy(t_pred.squeeze(), t_true))


def outcome_regression_loss_dragnn(
    t_true: torch.tensor,
    y_treatment_pred: torch.tensor,
    y_control_pred: torch.tensor,
    t_pred: torch.tensor,
    y_true: torch.tensor,
) -> torch.tensor:
    """
    Compute mse for treatment and control output layers using treatment vector for masking
    """

    loss0 = torch.mean(
        (1.0 - t_true) * F.mse_loss(y_control_pred.squeeze(), y_true, reduction="none")
    )
    loss1 = torch.mean(
        t_true * F.mse_loss(y_treatment_pred.squeeze(), y_true, reduction="none")
    )

    # make t_true as dtype float
    lossT = binary_treatment_loss(t_true.float(), F.sigmoid(t_pred))

    return loss0 + loss1 + lossT


def outcome_regression_loss(
    t_true: torch.tensor,
    y_treatment_pred: torch.tensor,
    y_control_pred: torch.tensor,
    y_true: torch.tensor,
) -> torch.tensor:
    """
    Compute mse for treatment and control output layers using treatment vector for masking out the counterfactual predictions
    """
    loss0 = torch.mean(
        (1.0 - t_true) * F.mse_loss(y_control_pred.squeeze(), y_true, reduction="none")
    )
    loss1 = torch.mean(
        t_true * F.mse_loss(y_treatment_pred.squeeze(), y_true, reduction="none")
    )

    return loss0 + loss1


def cluster(features: np.ndarray, budget: int, n_clusters: int = 100) -> tuple:
    """
    Cluster the data points into n_clusters clusters and return the cluster assignments, budget per cluster and negative distance between each sample and the closest centroid
    """
    # Initialize and fit the k-means model
    relative_budget = budget / features.shape[0]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)

    # Get cluster labels for each data point
    cluster_assignments = kmeans.labels_

    # Get cluster centroids
    centroids = kmeans.cluster_centers_
    # Optionally, you can also get the inertia (sum of squared distances to the nearest centroid) of the clustering
    # inertia = kmeans.inertia_
    closest_centroids, distances = pairwise_distances_argmin_min(features, centroids)

    cluster_size = dict(Counter(cluster_assignments))
    cluster_budget = {
        cluster: np.ceil(cluster_size[cluster] * relative_budget)
        for cluster in cluster_size
    }
    return cluster_assignments, cluster_budget, -distances


def mc_dropout(
    model: torch.nn.Module,
    xu_: torch.Tensor,
    xp: torch.Tensor,
    edge_index_up_current: torch.tensor,
    test_indices: torch.Tensor,
    ensembles: int = 5,
) -> tuple:
    """Function to get the monte-carlo samples and uncertainty estimates
    similar to https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
    """
    # model.train()
    dropout_predictions = []  # np.empty((0, n_samples, n_classes))

    for i in range(ensembles):
        # predictions = np.empty((0, n_classes))
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

        pred_treatment, pred_control, hidden_treatment, hidden_control = model(
            xu_, xp, edge_index_up_current
        )
        pred_treatment = pred_treatment[test_indices].detach().cpu().numpy()
        pred_control = pred_control[test_indices].detach().cpu().numpy()
        uplift = pred_treatment - pred_control

        dropout_predictions.append(uplift)

    dropout_predictions = np.hstack(dropout_predictions)
    # print(dropout_predictions.shape)
    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_predictions, axis=1)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes
    variance = np.var(dropout_predictions, axis=1)  # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    return mean, variance, entropy


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_outcome_feature(
    x: torch.Tensor,  # not needed anymore
    train_indices: Union[torch.Tensor, list],
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Make a feature to propagate with non zero value only for the train indices
    """
    mask = torch.ones(y.size(0), dtype=torch.bool)
    mask[train_indices] = 0
    y[mask] = 0
    return y





def run_benchmark(
    confounders: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    k: int,
    task: int = 0,
    causal_model_type: str = "X",
    model_out: str = "XGB",
    random_seed: int = 0,
) -> list:
    """
    Test the causalml model in a kfold cross validation.
    """
    results = []

    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    results = []
    for train_indices, test_indices in kf.split(confounders):
        test_indices, train_indices = train_indices, test_indices

        up40, up20 = causalml_run(
            confounders[train_indices],
            outcome[train_indices],
            treatment[train_indices],
            confounders[test_indices],
            outcome[test_indices],
            treatment[test_indices],
            task,
            causal_model_type,
            model_out,
        )
        results.append((up40, up20))

    return pd.DataFrame(results)