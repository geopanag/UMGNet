import os
import json
import random
import logging
import numpy as np
import pandas as pd
import torch
from data_preperation import download_and_prepare
from utils import outcome_regression_loss, run_umgnn, seed_everything
from benchmarks import benchmarks

DATA_NAME = 'retailhero'
CONFIG_PATH = "config_"+DATA_NAME+".json"

DATA_DIR = "../../data/"+DATA_NAME

USER_FEATURES = [
    "age", "F", "M", "U", 
    "first_issue_abs_time", 
    "first_redeem_abs_time", 
    "redeem_delay"
]

def load_config(config_path:str):
    with open(config_path, "r") as f:
        return json.load(f)

def setup_environment(config:dict):
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_DIR):
        download_and_prepare(DATA_NAME)

    os.chdir(DATA_DIR)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(config:dict, device:torch.device):
    edge_df = pd.read_csv(config["edge_index_file"])
    features_df = pd.read_csv(config["feature_file"])
    
    edge_index = torch.tensor(edge_df[["user", "product"]].values, dtype=torch.long).T.to(device)
    treatment = torch.tensor(features_df["treatment_flg"].values, dtype=torch.long).to(device)
    outcome_money = torch.tensor(features_df["avg_money_after"].values, dtype=torch.float).to(device)
    outcome_change = torch.tensor(features_df["avg_money_change"].values, dtype=torch.float).to(device)

    # Drop unused columns
    features_df = features_df.drop(["avg_money_before", "avg_count_before"], axis=1)

    return edge_df, features_df, edge_index, treatment, outcome_money, outcome_change

def build_input_tensors(features_df, device):
    xu = torch.tensor(features_df[USER_FEATURES].values, dtype=torch.float).to(device)
    return xu


def run_umgn(config : dict , 
            edge_df: pd.DataFrame, 
            edge_index : torch.tensor, 
            features_df : pd.DataFrame, 
            treatment: torch.tensor, 
            outcome_money: torch.tensor, 
            outcome_change: torch.tensor, 
            device : torch.device):
    
    xp = torch.eye(len(edge_df["product"].unique())).to(device)

    for k in [5, 20]:
        for task in [1, 2]:
            xu = build_input_tensors(features_df, device)
            outcome = outcome_money if task == 1 else outcome_change
            criterion = outcome_regression_loss

            version_tag = f"umgn_{config['lr']}_{config['n_hidden']}_{config['num_epochs']}_{config['dropout']}_{config['with_label_prop']}_{k}_{task}"
            model_file = config["model_file"].replace("version", version_tag)
            results_file = config["results_file_name"].replace("version", version_tag)

            results = []
            for run in range(config["number_of_runs"]):
                seed_everything(run)

                result = run_umgnn(
                    outcome=outcome,
                    treatment=treatment,
                    criterion=criterion,
                    xu=xu,
                    xp=xp,
                    edge_index=edge_index,
                    edge_index_df=edge_df,
                    task=task,
                    n_hidden=config["n_hidden"],
                    out_channels=config["out_channels"],
                    no_layers=config["no_layers"],
                    k=k,
                    run=run,
                    model_file=model_file,
                    num_users=treatment.shape[0],
                    num_products=len(edge_df["product"].unique()),
                    with_lp=config["with_label_prop"] == 1,
                    alpha=0.5,
                    l2_reg=config["l2_reg"],
                    dropout=config["dropout"],
                    lr=config["lr"],
                    num_epochs=config["num_epochs"],
                    early_thres=config["early_stopping_threshold"],
                    repr_balance=False,
                    device=device
                )
                results.append(result)

            pd.DataFrame(results).to_csv(results_file, index=False)



def main():
    config = load_config(CONFIG_PATH)
    device = setup_environment(config)
    benchmarks(CONFIG_PATH)
    edge_df, features_df, edge_index, treatment, outcome_money, outcome_change = load_data(config, device)
    run_umgn(config, edge_df, edge_index, features_df, treatment, outcome_money, outcome_change, device)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
