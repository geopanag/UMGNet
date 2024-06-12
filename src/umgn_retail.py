import pandas as pd 
import os
import numpy as np
import json 
import torch
from sklearn.preprocessing import StandardScaler


from utils import  outcome_regression_loss, run_umgnn
import random





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
    dropout = config['dropout']

    alpha = 0.5
    repr_balance = False
    
    edge_index_df = pd.read_csv(config["edge_index_file"])
    features = pd.read_csv(config["user_feature_file"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_products = len(edge_index_df['product'].unique())
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
    
    xp = torch.eye(num_products).to(device)
    
    for k in [5, 20]:
        for task in [1,2]:
            features_tmp  = features[['age','F','M','U','first_issue_abs_time','first_redeem_abs_time','redeem_delay'] ]
            print( features_tmp.columns)
            xu = torch.tensor(features_tmp.values).type(torch.FloatTensor).to(device)

            torch.cuda.empty_cache()

            v = "umgn_"+str(lr)+"_"+str(n_hidden)+"_"+str(num_epochs)+"_"+str(dropout)+"_"+str(with_lp)+"_"+str(k)+"_"+str(task)

            model_file = model_file_name.replace("version",str(v))
            results_file = results_file_name.replace("version",str(v))

            result_version = []
            for run in range(number_of_runs):  
                np.random.seed(run)
                random.seed(run)
                torch.manual_seed(run)

                # extract the features and the labels
                if task == 1:
                    outcome = outcome_money
                elif task == 2:
                    outcome = outcome_change

                criterion = outcome_regression_loss

                num_users = int(treatment.shape[0])

                result_fold = run_umgnn(outcome, treatment, criterion,xu, xp, edge_index, edge_index_df, task, n_hidden, out_channels, no_layers, k, run, model_file, num_users, num_products, with_lp, alpha, l2_reg, dropout, lr, num_epochs, early_thres,repr_balance, device)
                result_version.append(result_fold)

            pd.DataFrame(result_version).to_csv(results_file,index=False)



if __name__ == '__main__':
    main()
