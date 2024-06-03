
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from sklearn.model_selection import KFold
import random
from cfr import CFR
import time

cfg = {
    "alpha": 10 ** 6,
    "lr": 1e-3,
    "wd": 0.5,
    "sig": 0.1,
    "epochs": 1000,
    "ipm_type": "mmd_lin",
    "repnet_num_layers": 3,
    "repnet_hidden_dim": 48,
    "repnet_out_dim": 48,
    "repnet_dropout": 0.145,
    "outnet_num_layers": 3,
    "outnet_hidden_dim": 32,
    "outnet_dropout": 0.145,
    "gamma": 0.97,
    "split_outnet": True,
}

class DataSet:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index, :], self.y[index], self.z[index,:]#, :

    
def test_cfr(X_train: np.ndarray ,y_train: np.ndarray, t_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, t_test: np.ndarray, device):
    batch_size = 256

    dataset = DataSet(X_train, y_train, t_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True, drop_last=True)
   
    print('training CFR')
    t = time.time()
    model = CFR(in_dim = X_train.shape[1], out_dim = 1, cfg = cfg).to(device)

    train_mse, ipm_result = model.fit(
        dataloader, X_train, y_train, t_train, X_test, y_test, t_test,  device
        #torch.tensor(X_train, dtype=torch.float32, device=device), 
        #torch.tensor(y_train, dtype=torch.float32, device=device), 
        #torch.tensor(t_train, dtype=torch.float32, device=device), 
        #torch.tensor(X_test, dtype=torch.float32, device=device), 
        #torch.tensor(y_test, dtype=torch.float32, device=device), 
        #torch.tensor(t_test, dtype=torch.float32, device=device),  device
    )
    print(time.time()-t)

    x_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
    #t_test = torch.tensor(t_test, dtype=torch.float32, device=device)
    # here
    N = len(x_test)
    t_idx = np.where(t_test == 1)[0]
    c_idx = np.where(t_test == 0)[0]

    _t0 = torch.FloatTensor([0 for _ in range(N)]).reshape([-1, 1])
    _t1 = torch.FloatTensor([1 for _ in range(N)]).reshape([-1, 1])

    # maybe batch by batch
    _cate_c_list = []
    _cate_t_list = []

    

    print("testing CFR")
    model.eval()

    for i in range(0, len(x_test), batch_size):
        x_test_batch = x_test[i:min(len(x_test),i+batch_size)]
        y_test_batch = y_test[i:min(len(x_test),i+batch_size) ]
        _t0_batch = _t0[i:min(len(x_test),i+batch_size)]
        _t1_batch = _t1[i:min(len(x_test),i+batch_size)]
        
        _cate_c = y_test_batch - model.forward(x_test_batch, _t0_batch)
        _cate_t = model.forward(x_test_batch, _t1_batch) - y_test_batch
        
        _cate_c_list.append(_cate_c.cpu().detach().numpy().reshape(-1, 1))
        _cate_t_list.append(_cate_t.cpu().detach().numpy().reshape(-1, 1)) 
        
    
    _cate_t = np.concatenate(_cate_t_list)
    _cate_c = np.concatenate(_cate_c_list)

    uplift = np.zeros((len(t_test),1))
    uplift[c_idx] = _cate_c[c_idx]
    uplift[t_idx] = _cate_t[t_idx]
    
    y_test = y_test.cpu().detach().numpy()
    #print(uplift[0:10])
    #print(t_test[0:10])
    #print(y_test[0:10])
    score40 = uplift_score(uplift.squeeze(), t_test.squeeze(), y_test, rate=0.4)
    
    score20 = uplift_score(uplift.squeeze(),t_test.squeeze(), y_test, rate = 0.2)
    print(f'uplift40 = {score40} uplift20 = {score20}')

    return score40, score20
        

    
def uplift_score(prediction, treatment, target, rate=0.2):
    """
    From https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
    """
    order = np.argsort(-prediction)
    # this does not work
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1] [:treatment_n].mean()
    #print(prediction[order])

    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    
    score = treatment_p - control_p
    return score




#path = "/home/georgios/Desktop/research/gnn_uplift/experiment/"
path = "/home/users/gpanagopoulos/experiments/causality/"

df = pd.read_csv(path+"data/RetailHero/user_features_v4.csv")
    
columns_to_use = [ 'age','F','M','U','first_issue_abs_time','first_redeem_abs_time','redeem_delay'] 
columns_to_norm = ['age','first_issue_abs_time','first_redeem_abs_time','redeem_delay']
tasks = [1,2]
dats = 1
features = df.copy()
if len(columns_to_norm)>0:
    normalized_data = StandardScaler().fit_transform(features[columns_to_norm])
    features[columns_to_norm] = normalized_data

confounders = features[columns_to_use].values
treatment = features['treatment_flg'].values

treatment = np.expand_dims(treatment, axis=1)

outcome = features['avg_money_change'].values

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu") 


task = [1]
number_of_runs = 5
for k in [5,20]:
    
    dataset_ = "retail"
    v = "benchmarks_cfr_"+dataset_+"_"+str(k)+"_"+str(task)
    
    causalml_dml_results_file = path+"results/causalml_dml_results_version".replace("version",str(v))
    result = pd.DataFrame()
    for run in range(number_of_runs): 

        random.seed(run)
        torch.manual_seed(run)

        kf = KFold(n_splits=k, shuffle=True, random_state=run)
        results = []
        print(f'k ={k} run = {run}')
        
        for train_indices, test_indices in kf.split(confounders):
            test_indices, train_indices = train_indices, test_indices

            
            up40, up20  = test_cfr(confounders[train_indices], outcome[train_indices] , treatment[train_indices,:], 
                                    confounders[test_indices], outcome[test_indices], treatment[test_indices,:], device)

            results.append((up40, up20))
        p = pd.Series( list(pd.DataFrame(results).round(4).mean().values.T) )
        result = pd.concat([result,p],axis=1)
        
        result.T.to_csv(causalml_dml_results_file.replace("dml","all"),index=False)
    