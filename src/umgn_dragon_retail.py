import pandas as pd 
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.optim import Adam
import torch.nn.functional as F

import json 
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from utils import  uplift_score, make_outcome_feature, outcome_regression_loss_dragnn
import random


from models import BipartiteDraGNN, UserMP




def run_dragnn(outcome: torch.tensor , treatment: torch.tensor, criterion: torch.nn.modules.loss._Loss, 
              xu: torch.tensor , xp: torch.tensor , edge_index: torch.tensor, edge_index_df: pd.DataFrame, 
              task: int, n_hidden: int, out_channels: int, no_layers: int, k: int, run: int,
              model_file: str, num_users: int, num_products:int , with_lp: bool, alpha: float, l2_reg: float, 
              dropout: float, lr: float, num_epochs: int, early_thres: int,repr_balance: bool, device:torch.device, 
              validation_fraction: int =5) -> np.ndarray:
    #------ K fold split
    kf = KFold(n_splits=abs(k), shuffle=True, random_state=run)
    result_fold = []
    if with_lp:
        if task==0:
            dummy_product_labels = torch.zeros([num_products,2]).to(device)
        else:
            dummy_product_labels = torch.zeros([num_products,1]).to(device)
        user_conv = UserMP(aggr='mean' , normed=False).to(device)
        user_conv.eval()


    for train_indices, test_indices in kf.split(xu):
        test_indices, train_indices = train_indices, test_indices

        # split the test indices to test and validation 
        val_indices = train_indices[:int(len(train_indices)/validation_fraction)]
        subtrain_indices = train_indices[int(len(train_indices)/validation_fraction):]

        ## Keep the graph before the treatment and ONLY the edges of the the train nodes (i.e. after the treatment)
        # remove edge_index_df[ edge_index_df['user'].isin(train_indices)  if you dont want edges from train set
        edge_index_up_current = edge_index[:, edge_index_df[ edge_index_df['user'].isin(subtrain_indices) | edge_index_df['T']==0 ].index.values]
        # make unsupervised and add num_nodes for bipartite message passing
        edge_index_up_current[1] = edge_index_up_current[1]+ num_users
        edge_index_up_current = torch.cat([edge_index_up_current,edge_index_up_current.flip(dims=[0])],dim=1)

        ###------------------------------------------------------------ Label propagation
        ## Each user will have an estimate of its neighbors train labels (but not its own), mainly to assist semi-supervised learning
        if with_lp:
            label_for_prop = make_outcome_feature(xu, train_indices,  outcome.type(torch.LongTensor) ).to(device)

            label_for_prop = torch.cat([label_for_prop, dummy_product_labels], dim=0)

            label_for_prop = user_conv(label_for_prop, edge_index_up_current)
            label_for_prop = user_conv(label_for_prop, edge_index_up_current)
            label_for_prop = label_for_prop[:num_users,:].detach().to(device)
            xu_ = torch.cat([xu, label_for_prop],dim=1)
        else:
            xu_ = xu

            
        #---------------------------------------------------------- Model and Optimizer
        # xu_ : user embeddings e.g. sex, age, coupon issue time etc.
        # xp : product embeddings one-hot encoding 
        model = BipartiteDraGNN(xu_.shape[1], xp.shape[1] , n_hidden, out_channels, no_layers, dropout).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay = l2_reg)

        # init params
        out_ = model( xu_ , xp , edge_index_up_current)

        train_losses = []
        val_losses = []
        best_val_loss = np.inf
        early_stopping = 0

        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()

            
            out_treatment, out_control, out_T, hidden_treatment, hidden_control = model(xu_, xp, edge_index_up_current)

            
            if task==0:
                out_treatment = F.sigmoid(out_treatment)
                out_control = F.sigmoid(out_control)
                
            #if repr_balance:
            #dist = alpha * wasserstein(hidden_treatment, hidden_control, cuda=True) 
            #dist = alpha*sinkhorn_loss(hidden_treatment , hidden_control)
            #sinkhorn = SinkhornDistance(eps=0.1, max_iter=100) 
            #dist = alpha* sinkhorn(hidden_treatment , hidden_control)
            #else:
            #    dist = 0

            loss = criterion(treatment[subtrain_indices], out_treatment[subtrain_indices], 
                        out_control[subtrain_indices], out_T[subtrain_indices], outcome[subtrain_indices]) + dist # target_labels are your binary labels

            #loss = criterion(out[train_indices], y[train_indices]) # target_labels are your binary labels
            loss.backward()
            optimizer.step()
            total_loss = float(loss.item())

            train_losses.append(total_loss ) 
            # test validation loss 
            if epoch%5==0:
                with torch.no_grad():
                    # no dropout hence no need to rerun
                    model.eval()
                    out_treatment, out_control, out_T, hidden_treatment, hidden_control = model(xu_, xp, edge_index_up_current)
                    
                    if task==0:
                        out_treatment = F.sigmoid(out_treatment)
                        out_control = F.sigmoid(out_control)
                        
                    loss = criterion(treatment[val_indices],out_treatment[val_indices], out_control[val_indices], out_T[val_indices], outcome[val_indices])# + dist

                    val_loss = round(float(loss.item()),3)
                    val_losses.append(val_loss)

                    #-------------------------------------------------------------------------------------------
                    #torch.save(model, model_file) 
                    #-------------------------------------------------------------------------------------------
                    if val_loss < best_val_loss:
                        early_stopping=0
                        best_val_loss = val_loss 
                        torch.save(model, model_file)
                    else:
                        early_stopping += 1
                        if early_stopping > early_thres:
                            print("early stopping..")
                            break

                    model.train()

                if epoch%10==0:
                    print(train_losses[-1])
                    print(val_losses[-10:])

        ###---------- Final evlauation
        model = torch.load(model_file).to(device)
        model.eval()

        out_treatment, out_control, out_T, hidden_treatment, hidden_control = model(xu_, xp, edge_index_up_current)

        if task==0:
            out_treatment = F.sigmoid(out_treatment)
            out_control = F.sigmoid(out_control)

        #------------------------ Evaluating
        treatment_test = treatment[test_indices].detach().cpu().numpy()
        outcome_test = outcome[test_indices].detach().cpu().numpy()
        out_treatment = out_treatment.detach().cpu().numpy()
        out_control = out_control.detach().cpu().numpy()

        uplift = out_treatment[test_indices] - out_control[test_indices]
        uplift = uplift.squeeze()

        mse = (uplift.mean() - (outcome_test[treatment_test==1].mean() - outcome_test[treatment_test==0].mean()))**2
        print(f'mse {mse}')
        up40 = uplift_score(uplift, treatment_test, outcome_test,0.4)
        print(f'up40 {up40}')
        up20 = uplift_score(uplift, treatment_test, outcome_test,0.2)
        print(f'up20 {up20}')

        
        result_fold.append((up40, up20))

    return pd.DataFrame(result_fold).mean().values






def main():
    #----------------- Load parameters
    with open('config_RetailHero.json', 'r') as config_file:
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
    with_lp = config['with_label_prop'] == 1
    number_of_runs = config['number_of_runs']
    dropout = config["dropout"]

    alpha = 0.5
    repr_balance = False

    edge_index_df = pd.read_csv(config["edge_index_file"])
    features = pd.read_csv(config["feature_file"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edge_index = torch.tensor(edge_index_df[['user','product']].values).type(torch.LongTensor).T.to(device)
       
    columns_to_norm = ['age','first_issue_abs_time','first_redeem_abs_time','redeem_delay','degree_before','weighted_degree_before'] 
    if len(columns_to_norm)>0:
        normalized_data = StandardScaler().fit_transform(features[columns_to_norm])
        features[columns_to_norm] = normalized_data

    # extract the features and the labels
    treatment =torch.tensor( features['treatment_flg'].values).type(torch.LongTensor).to(device)
    #outcome_original = torch.tensor(features['target'].values).type(torch.FloatTensor).to(device)
    outcome_money = torch.tensor(features['avg_money_after'].values).type(torch.FloatTensor).to(device)
    outcome_change = torch.tensor(features['avg_money_change'].values).type(torch.FloatTensor).to(device)

    # add always the product with the maximum index (it has only one edge) to facilitate the sparse message passing
    features = features.drop(['avg_money_before','avg_count_before'],axis=1)
    
    
    num_products = len(edge_index_df['product'].unique())
    xp = torch.eye(num_products).to(device)

    for k in [20, 5]:
        for task in [1,2]:
        
            features_tmp  = features[['age','F','M','U','first_issue_abs_time','first_redeem_abs_time','redeem_delay'] ]
                
            xu = torch.tensor(features_tmp.values).type(torch.FloatTensor).to(device)
            
            torch.cuda.empty_cache()
            
            v = "dragon_"+str(lr)+"_"+str(n_hidden)+"_"+str(num_epochs)+"_"+str(dropout)+"_"+str(with_lp)+"_"+str(k)+"_"+str(task)
            
            model_file = model_file_name.replace("version",str(v))
            results_file = results_file_name.replace("version",str(v))

            result_version = []
            for run in range(number_of_runs):  
                run+=5
                random.seed(run)
                torch.manual_seed(run)

                # extract the features and the labels
                if task == 1:
                    outcome = outcome_money
                elif task == 2:
                    outcome = outcome_change

                criterion = outcome_regression_loss_dragnn

                num_users = int(treatment.shape[0])

                result_fold = run_dragnn(outcome, treatment, criterion, xu, xp, edge_index, edge_index_df, task, n_hidden, out_channels, no_layers, k, run, model_file, num_users, num_products, with_lp, alpha, l2_reg, dropout, lr, num_epochs, early_thres,repr_balance, device)
                result_version.append(result_fold)

            pd.DataFrame(result_version).to_csv(results_file,index=False)



if __name__ == '__main__':
    main()
