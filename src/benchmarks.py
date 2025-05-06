import pandas as pd
import torch

import json
import os

import random

from sklearn.preprocessing import StandardScaler
from utils import run_benchmark

import sys
import argparse

columns_to_use = [
            "age",
            "F",
            "M",
            "U",
            "first_issue_abs_time",
            "first_redeem_abs_time",
            "redeem_delay",
        ]
columns_to_norm = [
    "age",
    "first_issue_abs_time",
    "first_redeem_abs_time",
    "redeem_delay",
]


def benchmarks(config_file):

    with open(config_file, "r") as config_file:
        config = json.load(config_file)

    path_to_data = config["path_to_data"]

    os.chdir(path_to_data)

    dataset = config["dataset"]
    number_of_runs = config["number_of_runs"]

    df = pd.read_csv(config["feature_file"])

    if dataset == "retail":
        tasks = [1, 2]
        datasets = 1
        features = df.copy()
        if len(columns_to_norm) > 0:
            normalized_data = StandardScaler().fit_transform(features[columns_to_norm])
            features[columns_to_norm] = normalized_data

        confounders = features[columns_to_use].values
        treatment = features["treatment_flg"].values

    else:
        tasks = [3]
        datasets = 5
        features = pd.read_csv(config["feature_file"])  # "movielens_features.csv")
        treatment = features.values[:, 0].astype(int)
        confounders = features.values[:, 1:]

    for dat in range(datasets):
        for task in tasks:
            for k in [5, 20]:

                dataset_ = dataset + str(dat)

                v = dataset_ + "_" + str(k) + "_" + str(task)

                results_file = config["benchmark_results_file"].replace("version", str(v))

                result = pd.DataFrame()
                for run in range(number_of_runs):

                    random.seed(run)
                    torch.manual_seed(run)

                    if dataset == "retail":
                        # extract the features and the labels
                        if task == 1:
                            outcome = features["avg_money_after"].values
                        elif task == 2:
                            outcome = features["avg_money_change"].values
                        else:
                            outcome = (
                                pd.read_csv(
                                    config["output_file"].replace("run", str(dat))
                                )
                                .squeeze()
                                .values
                            )

                    causalml_results = run_benchmark(
                        confounders, outcome, treatment, k, task, "S", random_seed=run
                    )
                    p = pd.Series(
                        list(causalml_results.round(4).mean().values.T) + ["S"]
                    )
                    result = pd.concat([result, p], axis=1)
                    print("S done")

                    causalml_results = run_benchmark(
                        confounders, outcome, treatment, k, task, "R", random_seed=run
                    )
                    p = pd.Series(list(causalml_results.round(4).mean().values) + ["R"])
                    result = pd.concat([result, p], axis=1)
                    print("R done")

                    causalml_results = run_benchmark(
                        confounders, outcome, treatment, k, task, "T", random_seed=run
                    )
                    p = pd.Series(
                        list(causalml_results.round(4).mean().values.T) + ["T"]
                    )
                    result = pd.concat([result, p], axis=1)
                    print("T done")

                    causalml_results = run_benchmark(
                        confounders, outcome, treatment, k, task, "D", random_seed=run
                    )
                    p = pd.Series(
                        list(causalml_results.round(4).mean().values.T) + ["D"]
                    )
                    result = pd.concat([result, p], axis=1)
                    print("D done")

                    causalml_results = run_benchmark(
                        confounders, outcome, treatment, k, task, "X", random_seed=run
                    )
                    p = pd.Series(
                        list(causalml_results.round(4).mean().values.T) + ["X"]
                    )
                    result = pd.concat([result, p], axis=1)
                    print("X done")

                    causalml_results = run_benchmark(
                        confounders,
                        outcome,
                        treatment,
                        k,
                        task,
                        "Tree",
                        "XGB",
                        random_seed=run,
                    )
                    p = pd.Series(
                        list(causalml_results.round(4).mean().values) + ["Tree"]
                    )
                    result = pd.concat([result, p], axis=1)
                    print("trees done")

                    causalml_results = run_benchmark(
                        confounders,
                        outcome,
                        treatment,
                        k,
                        task,
                        "Dragon",
                        random_seed=run,
                    )
                    p = pd.Series(
                        list(causalml_results.round(4).mean().values.T) + ["Dragon"]
                    )
                    result = pd.concat([result, p], axis=1)
                    print("dragon done")
                    result.T.to_csv(results_file.replace("dml", "all"), index=False)

                    # CEVAE takes too long to run

                    causalml_results = run_benchmark(
                        confounders,
                        outcome,
                        treatment,
                        k,
                        task,
                        "CEVAE",
                        random_seed=run,
                    )
                    p = pd.Series(
                        list(causalml_results.round(4).mean().values.T) + ["CEVAE"]
                    )
                    result = pd.concat([result, p], axis=1)
                    print("CEVAE done")
                    result.T.to_csv(results_file.replace("dml", "all"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_RetailHero.json",
        help="The config of the dataset to run.",
    )

    args = parser.parse_args()
    config_file = args.config
    print(config_file)
    benchmarks(config_file)
