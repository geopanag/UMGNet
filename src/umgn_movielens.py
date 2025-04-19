import pandas as pd
import os
import numpy as np
import json
import torch
from sklearn.preprocessing import StandardScaler

from utils import outcome_regression_loss, run_umgnn
import random


def main():
    # ----------------- Load parameters
    with open("config_Movielens25.json", "r") as config_file:
        config = json.load(config_file)

    path_to_data = config["path_to_data"]
    os.chdir(path_to_data)

    n_hidden = config["n_hidden"]
    no_layers = config["no_layers"]
    out_channels = config["out_channels"]
    num_epochs = config["num_epochs"]
    lr = config["lr"]
    k = config["k_fold"][0]
    results_file_name = config["results_file_name"]
    model_file_name = config["model_file"]
    early_thres = config["early_stopping_threshold"]
    l2_reg = config["l2_reg"]
    dropout = config["dropout"]
    dataset = config["dataset"]
    with_lp = config["with_label_prop"] == 1
    number_of_runs = config["number_of_runs"]

    alpha = 0.5
    task = 3
    repr_balance = False

    edge_index_df = pd.read_csv(config["edge_index_file"])

    num_products = len(edge_index_df["user"].unique())

    features = pd.read_csv(config["feature_file"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = (
        torch.tensor(edge_index_df[["movie", "user"]].values)
        .type(torch.LongTensor)
        .T.to(device)
    )

    treatment = (
        torch.tensor(features.values[:, 0].astype(int))
        .type(torch.LongTensor)
        .to(device)
    )
    # outcome = torch.tensor( features.values[:,1].astype(int)).type(torch.FloatTensor).to(device)

    confounders = StandardScaler().fit_transform(features.values[:, 2:])

    xu = torch.tensor(confounders).type(torch.FloatTensor).to(device)

    xp = torch.eye(num_products).to(device)

    for dat in range(5):
        outcome = (
            torch.tensor(pd.read_csv(f"movielens_y_{dat}.csv").squeeze().values)
            .type(torch.FloatTensor)
            .to(device)
        )

        for k in [5, 20]:
            torch.cuda.empty_cache()
            v = (
                "umgn_"
                + dataset
                + str(lr)
                + "_"
                + str(n_hidden)
                + "_"
                + str(num_epochs)
                + "_"
                + str(dropout)
                + "_"
                + str(with_lp)
                + "_"
                + str(k)
                + "_"
                + str(task)
                + "_"
                + str(dat)
            )

            model_file = model_file_name.replace("version", str(v))
            results_file = results_file_name.replace("version", str(v))

            result_version = []
            for run in range(number_of_runs):
                np.random.seed(run)
                random.seed(run)
                torch.manual_seed(run)

                criterion = outcome_regression_loss

                num_users = int(treatment.shape[0])

                result_fold = run_umgnn(
                    outcome,
                    treatment,
                    criterion,
                    xu,
                    xp,
                    edge_index,
                    edge_index_df,
                    task,
                    n_hidden,
                    out_channels,
                    no_layers,
                    k,
                    run,
                    model_file,
                    num_users,
                    num_products,
                    with_lp,
                    alpha,
                    l2_reg,
                    dropout,
                    lr,
                    num_epochs,
                    early_thres,
                    repr_balance,
                    device,
                )
                result_version.append(result_fold)

                pd.DataFrame(result_version).to_csv(results_file, index=False)


if __name__ == "__main__":
    main()
