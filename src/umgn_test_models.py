import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.optim import Adam
import torch.nn.functional as F

import json

import torch
import importlib
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import SAGEConv, Linear

from utils import uplift_score, make_outcome_feature
import random

import argparse

parser = argparse.ArgumentParser(description="Run main.")
parser.add_argument('--config', type=str, default='config_GCNConv')
args = parser.parse_args()

from utils import  uplift_score, make_outcome_feature,binary_treatment_loss, outcome_regression_loss
import random


from models import BipartiteSAGE2mod, UserMP



class GNNModel(torch.nn.Module):
    def __init__(self, nfeat: int, nproduct: int, hidden_channels: int, out_ch: int, num_layers: int,
                 dropout_rate: float = 0, model_class: str = 'SAGEConv', **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        self.user_embed = nn.Linear(nfeat, hidden_channels)
        self.item_embed = nn.Linear(nproduct, hidden_channels)

        self.convs = torch.nn.ModuleList()

        try:
            self.model_class = getattr(importlib.import_module(f"torch_geometric.nn"), f"{model_class}")
        except AttributeError:
            # model layer is custom and not from PyG
            self.model_class = getattr(importlib.import_module(f"custom_layers.{model_class.lower()}"), f"{model_class}")

        if model_class == 'SAGEConv':
            kwargs = {
                'in_channels': (-1, -1),
                'out_channels': hidden_channels,
                'activation': 'ReLU'
            }

        self.activation = getattr(importlib.import_module(f"torch.nn"), f'{kwargs["activation"]}') \
            if 'activation' in kwargs else nn.Identity
        self.activation = self.activation()
        kwargs.pop('activation', None)

        for _ in range(num_layers):
            self.convs.append(self.model_class(**kwargs))

        # self.lin_hetero = Linear(hidden_channels, out_ch)

        self.num_layers = num_layers

        self.hidden_common1 = Linear(hidden_channels + num_layers * hidden_channels, hidden_channels)
        self.hidden_common2 = Linear(hidden_channels, hidden_channels)

        self.hidden_control = Linear(hidden_channels, int(hidden_channels / 2))
        self.hidden_treatment = Linear(hidden_channels, int(hidden_channels / 2))

        self.out_control = Linear(int(hidden_channels / 2), out_ch)
        self.out_treatment = Linear(int(hidden_channels / 2), out_ch)

        # self.lin = Linear(hidden_channels, out_ch)
        self.dropout = nn.Dropout(dropout_rate)
        # self.bn_hidden = nn.BatchNorm1d(hidden_channels)

        # self.bn_out = nn.BatchNorm1d(nfeat + hidden_channels + hidden_channels)

    def forward(self, xu: torch.tensor, xp: torch.tensor, edge_index: torch._tensor):
        out = []
        xu = self.user_embed(xu)
        xp = self.item_embed(xp)

        out.append(xu)

        embeddings = torch.cat((xu, xp), dim=0)

        for i in range(self.num_layers):
            embeddings = self.activation(self.convs[i](embeddings, edge_index))
            # embeddings = self.dropout(embeddings)
            # embeddings = self.bn_hidden(embeddings)

            out.append(embeddings[:xu.shape[0]])

        out = torch.cat(out, dim=1)

        hidden = self.dropout(self.activation(self.hidden_common1(out)))
        hidden = self.dropout(self.activation(self.hidden_common2(hidden)))

        # separate treatment and control
        hidden_1t0 = self.dropout(self.activation(self.hidden_control(hidden)))
        hidden_1t1 = self.dropout(self.activation(self.hidden_treatment(hidden)))

        out_2t0 = self.activation(self.out_control(hidden_1t0))
        out_2t1 = self.activation(self.out_treatment(hidden_1t1))

        return out_2t1, out_2t0, hidden_1t1, hidden_1t0



def outcome_regression_loss(t_true: torch.tensor, y_treatment_pred: torch.tensor, y_control_pred: torch.tensor,
                            y_true: torch.tensor):
    """
    Compute binary cross entropy for treatment and control output layers using treatment vector for masking 
    """
    # torch.where(t_true==1, y_treatment_pred, y_control_pred)
    loss0 = torch.mean((1. - t_true) * F.mse_loss(y_control_pred.squeeze(), y_true, reduction='none'))
    loss1 = torch.mean(t_true * F.mse_loss(y_treatment_pred.squeeze(), y_true, reduction='none'))

    return loss0 + loss1




def run_umgnn(outcome, treatment, criterion, xu, xp, edge_index, edge_index_df, task, n_hidden, out_channels, no_layers,
             k, run, num_users, num_products, with_lp, alpha, l2_reg, dropout, lr, num_epochs, early_thres,
             repr_balance, device, model_file, model_class='SAGEConv',  model_config=None, validation_fraction=5):
    
    # ------ K fold split
    if model_config is None:
        model_config = {}
        
    kf = KFold(n_splits=abs(k), shuffle=True, random_state=run)
    result_fold = []
    if with_lp:   
        dummy_product_labels = torch.zeros([num_products,1]).to(device).squeeze()

    for train_indices, test_indices in kf.split(xu):
        test_indices, train_indices = train_indices, test_indices

        # split the test indices to test and validation 
        val_indices = train_indices[:int(len(train_indices) / validation_fraction)]
        subtrain_indices = train_indices[int(len(train_indices) / validation_fraction):]

        ## Keep the graph before the treatment and ONLY the edges of the the train nodes (i.e. after the treatment)
        # remove edge_index_df[ edge_index_df['user'].isin(train_indices)  if you dont want edges from train set
        edge_index_up_current = edge_index[:, edge_index_df[
                                                  edge_index_df['user'].isin(subtrain_indices) | edge_index_df[
                                                      'T'] == 0].index.values]
        # make unsupervised and add num_nodes for bipartite message passing
        edge_index_up_current[1] = edge_index_up_current[1] + num_users
        edge_index_up_current = torch.cat([edge_index_up_current, edge_index_up_current.flip(dims=[0])], dim=1)

        ###------------------------------------------------------------ Label propagation
        ## Each user will have an estimate of its neighbors train labels (but not its own), mainly to assist semi-supervised learning
        if with_lp:
            label_for_prop = make_outcome_feature(xu, train_indices,  outcome.type(torch.LongTensor) ).to(device)

            label_for_prop = torch.cat([label_for_prop, dummy_product_labels], dim=0)
            model = UserMP().to(device)
            label_for_prop = model(label_for_prop, edge_index_up_current)
            label_for_prop = model(label_for_prop, edge_index_up_current)
            #print(label_for_prop.shape)
            label_for_prop = label_for_prop[:num_users].detach().to(device)
            mean = torch.mean(label_for_prop)
            std_dev = torch.std(label_for_prop)

            # Standardize the vector
            label_for_prop = (label_for_prop - mean) / std_dev
            
            xu_ = torch.cat([xu, label_for_prop.unsqueeze(1) ],dim=1)
        else:
            xu_ = xu

        # ---------------------------------------------------------- Model and Optimizer
        # xu_ : user embeddings e.g. sex, age, coupon issue time etc.
        # xp: product embeddings one-hot encoding 
        # model = BipartiteSAGE2mod(xu_.shape[1], xp.shape[1], n_hidden, out_channels, no_layers, dropout).to(device)
        model = GNNModel(xu_.shape[1],
                         xp.shape[1],
                         n_hidden,
                         out_channels,
                         no_layers,
                         dropout,
                         model_class=model_class,
                         **model_config).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

        # init params
        out = model(xu_, xp, edge_index_up_current)

        train_losses = []
        val_losses = []
        best_val_loss = np.inf
        early_stopping = 0

        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()

            out_treatment, out_control, hidden_treatment, hidden_control = model(xu_, xp, edge_index_up_current)

            if task == 0:
                out_treatment = F.sigmoid(out_treatment)
                out_control = F.sigmoid(out_control)


            loss = criterion(treatment[subtrain_indices], out_treatment[subtrain_indices],
                             out_control[subtrain_indices],
                             outcome[subtrain_indices]) # target_labels are your binary labels

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
                    out_treatment, out_control, hidden_treatment, hidden_control = model(xu_, xp, edge_index_up_current)

                    if task == 0:
                        out_treatment = F.sigmoid(out_treatment)
                        out_control = F.sigmoid(out_control)

                    loss = criterion(treatment[val_indices], out_treatment[val_indices], out_control[val_indices],
                                     outcome[val_indices])  # + dist

                    val_loss = round(float(loss.item()), 3)
                    val_losses.append(val_loss)

                    # -------------------------------------------------------------------------------------------
                    # torch.save(model, model_file)
                    # -------------------------------------------------------------------------------------------
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

        print(f'loading {model_class}')
        model = torch.load(model_file).to(device)
        model.eval()

        out_treatment, out_control, hidden_treatment, hidden_control = model(xu_, xp, edge_index_up_current)

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

        mse = (uplift.mean() - (
                    outcome_test[treatment_test == 1].mean() - outcome_test[treatment_test == 0].mean())) ** 2
        print(f'mse {mse}')
        up40 = uplift_score(uplift, treatment_test, outcome_test, 0.4)
        print(f'up40 {up40}')
        up20 = uplift_score(uplift, treatment_test, outcome_test, 0.2)
        print(f'up20 {up20}')

        result_fold.append((up40, up20))

    return pd.DataFrame(result_fold).mean().values




def main():
    # ----------------- Load parameters
    with open(f'config_RetailHero.json', 'r') as config_file:
        config = json.load(config_file)

    path_to_data = config["path_to_data"]
    os.chdir(path_to_data)

   
    n_hidden = config["n_hidden"]
    no_layers = config["no_layers"]
    out_channels = config["out_channels"]
    num_epochs = config["num_epochs"]
    lr = config["lr"]
    results_file_name = config['results_file_name']
    model_file_name = config["model_file"]
    early_thres = config['early_stopping_threshold']
    l2_reg = config['l2_reg']
    #config["with_representation_balance"]==1
    with_lp = config['with_lp'] == 1
    number_of_runs = config['number_of_runs']
    alpha = 0.5
    repr_balance = False

    

    edge_index_df = pd.read_csv(config["edge_index_file"])
    features = pd.read_csv(config["user_feature_file"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edge_index = torch.tensor(edge_index_df[['user', 'product']].values).type(torch.LongTensor).T.to(device)

    features = features.drop(['avg_money_before', 'avg_count_before'], axis=1)

    columns_to_norm = ['age', 'first_issue_abs_time', 'first_redeem_abs_time', 'redeem_delay', 'degree_before',
                       'weighted_degree_before']
    if len(columns_to_norm) > 0:
        normalized_data = StandardScaler().fit_transform(features[columns_to_norm])
        features[columns_to_norm] = normalized_data


    num_products = len(edge_index_df['product'].unique())
    # extract the features and the labels
    treatment = torch.tensor(features['treatment_flg'].values).type(torch.LongTensor).to(device)
    outcome_original = torch.tensor(features['target'].values).type(torch.FloatTensor).to(device)
    outcome_money = torch.tensor(features['avg_money_after'].values).type(torch.FloatTensor).to(device)
    outcome_change = torch.tensor(features['avg_money_change'].values).type(torch.FloatTensor).to(device)

    # add always the product with the maximum index (it has only one edge) to facilitate the sparse message passing
    features = features[
        ['age', 'F', 'M', 'U', 'first_issue_abs_time', 'first_redeem_abs_time', 'redeem_delay']] #,'degree_before','weighted_degree_before'
    xu = torch.tensor(features.values).type(torch.FloatTensor).to(device)

    xp = torch.eye(num_products).to(device)
    
    
    model_configs = {}
    model_configs["GCNConv"] = {
                            "in_channels": -1,
                            "out_channels": 64
                            }
    model_configs["NGCF"] = {
                        "in_dim": 64,
                        "out_dim": 64,
                        "dropout": 0.1
                        }
    model_configs["LGConv"] = {}

    for k in [ 5, 20]:
        for task in [1, 2]:
            for model_class in ['NGCF','LGConv', "GCNConv"]:
                # for lr in [0.001,0.01]:
                torch.cuda.empty_cache()
                if lr == 0.001:
                    dropout = 0.2
                    n_hidden = 32
                else:
                    dropout = 0.4
                    n_hidden = 64

                v = "tgnn_oldfts_"+model_class+"_" + str(lr) + "_" + str(n_hidden) + "_" + str(num_epochs) + "_" + str(dropout) + "_" + str(with_lp) + "_" + str(k) + "_" + str(task)
                
                model_file = model_file_name.replace("version", str(v))
                results_file = results_file_name.replace("version", str(v))

                result_version = []
                for run in range(number_of_runs):
                    random.seed(run)
                    torch.manual_seed(run)

                    # extract the features and the labels
                    if task == 0:
                        outcome = outcome_original
                    elif task == 1:
                        outcome = outcome_money
                    elif task == 2:
                        outcome = outcome_change

                    criterion = outcome_regression_loss

                    num_users = int(treatment.shape[0])

                    result_fold = run_umgnn(outcome, treatment, criterion, xu, xp, edge_index, edge_index_df, task, n_hidden,
                                        out_channels, no_layers, k, run, num_users, num_products, with_lp,
                                        alpha, l2_reg, dropout, lr, num_epochs, early_thres, repr_balance, device,model_file, 
                                        model_class=model_class, model_config=model_configs[model_class])
                    result_version.append(result_fold)

                pd.DataFrame(result_version).to_csv(results_file, index=False)


if __name__ == '__main__':
    main()
