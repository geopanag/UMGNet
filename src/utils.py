
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import KFold
import torch
import numpy as np

import pandas as pd
import torch.nn.functional as F

from causalml.inference.meta import BaseXClassifier, BaseSClassifier, BaseTClassifier,BaseRClassifier, BaseDRRegressor, BaseXRegressor, BaseSRegressor, BaseTRegressor, BaseRRegressor
from causalml.inference.tf import DragonNet
from causalml.inference.tf.utils import regression_loss
from causalml.inference.tree import UpliftTreeClassifier
from causalml.inference.tree.causal.causaltree import CausalTreeRegressor
from causalml.inference.nn import CEVAE
from causalml.propensity import ElasticNetPropensityModel


from sklearn.linear_model import LogisticRegression
from typing import Callable


from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
from sklearn.cluster import KMeans

import sys



def al_lp(test_indices: np.ndarray, degree: np.ndarray, pred_uncertainty: np.ndarray, cluster_distance: np.ndarray, cluster_assigment: np.ndarray , 
          cluster_budget: dict, treatment: np.ndarray, sample_budget:int , a1:float, a2:float, a3:float):
    """
    Solve the active learning linear program with a greedy approach, sorting to maximize the objective function and adding samples that do not violate constraints
    """
    if len(pred_uncertainty)>0:    
        objective = a1*pred_uncertainty + a2*degree[test_indices] + a3*cluster_distance[test_indices]
    else:
        objective = a2*degree[test_indices] + a3*cluster_distance[test_indices]
    
    new_train_indices = []
    treatment_budget = np.ceil(sample_budget/2)

    # sorted indices contains the ABSOLUTE indices as stored in the test indices i.e. they point to xu, not relative indices that point to the test_indices vector
    sorted_indices = test_indices[np.argsort(objective)]

    # hence cluster_assignment and treatment are not subset to test_indices, but stay in dimension n which is where test indices point to

    # test the treatment and cluster budgets
    for node in sorted_indices:
        if (cluster_budget[cluster_assigment[node]]>=0):
            if treatment_budget-treatment[node]<0:
                # can not add more treated subjects
                continue

            cluster_budget[cluster_assigment[node]] -= 1
            treatment_budget = treatment_budget-treatment[node]
            new_train_indices.append(node)
        else:
            # can not add more subjects from this cluster
            continue 
    
    return new_train_indices

def binary_treatment_loss(t_true, t_pred):
    """
    Compute cross entropy for propensity score , from Dragonnet
    """
    t_pred = (t_pred + 0.001) / 1.002
    
    return torch.mean(F.binary_cross_entropy(t_pred.squeeze(), t_true))


def outcome_regression_loss_dragnn(t_true: torch.tensor,y_treatment_pred: torch.tensor, y_control_pred: torch.tensor, t_pred: torch.tensor, y_true: torch.tensor):
    """
    Compute binary cross entropy for treatment and control output layers using treatment vector for masking 
    """
    #torch.where(t_true==1, y_treatment_pred, y_control_pred)
    loss0 = torch.mean((1. - t_true) * F.mse_loss(y_control_pred.squeeze(), y_true, reduction='none')) 
    loss1 = torch.mean(t_true *  F.mse_loss(y_treatment_pred.squeeze(), y_true, reduction='none') )

    # make t_true as dtype float 

    lossT = binary_treatment_loss(t_true.float(), F.sigmoid(t_pred))

    return loss0 + loss1 + lossT


def binary_treatment_loss(t_true, t_pred):
    """
    Compute cross entropy for propensity score
    """
    t_pred = (t_pred + 0.001) / 1.002
    losst = torch.sum(F.binary_cross_entropy(t_pred.squeeze(), t_true))

    return losst


def outcome_regression_loss(t_true: torch.tensor,y_treatment_pred: torch.tensor, y_control_pred: torch.tensor, y_true: torch.tensor):
    """
    Compute binary cross entropy for treatment and control output layers using treatment vector for masking 
    """
    #torch.where(t_true==1, y_treatment_pred, y_control_pred)
    loss0 = torch.mean((1. - t_true) * F.mse_loss(y_control_pred.squeeze(), y_true, reduction='none')) 
    loss1 = torch.mean(t_true *  F.mse_loss(y_treatment_pred.squeeze(), y_true, reduction='none') )

    return loss0 + loss1



def cluster(features: np.ndarray, budget: int, n_clusters:int = 100):
    """
    Cluster the data points into n_clusters clusters and return the cluster assignments, budget per cluster and negative distance between each sample and the closest centroid
    """
    # Initialize and fit the k-means model
    relative_budget = budget/features.shape[0]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)

    # Get cluster labels for each data point
    cluster_assignments = kmeans.labels_

    # Get cluster centroids
    centroids = kmeans.cluster_centers_
    # Optionally, you can also get the inertia (sum of squared distances to the nearest centroid) of the clustering
    #inertia = kmeans.inertia_
    closest_centroids, distances = pairwise_distances_argmin_min(features, centroids)

    cluster_size = dict(Counter(cluster_assignments))
    cluster_budget = { cluster: np.ceil(cluster_size[cluster]*relative_budget) for cluster in cluster_size }
    return cluster_assignments, cluster_budget, -distances




def mc_dropout(model,xu_, xp, edge_index_up_current, test_indices, ensembles = 5):
    """ Function to get the monte-carlo samples and uncertainty estimates
    similar to https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch 
    """
    #model.train()
    dropout_predictions = []#np.empty((0, n_samples, n_classes))
    
    for i in range(ensembles):
        #predictions = np.empty((0, n_classes))
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
        pred_treatment, pred_control, hidden_treatment, hidden_control = model(xu_, xp, edge_index_up_current)
        pred_treatment = pred_treatment[test_indices].detach().cpu().numpy()
        pred_control = pred_control[test_indices].detach().cpu().numpy()
        uplift = pred_treatment - pred_control               
        
        dropout_predictions.append(uplift) 
        
    dropout_predictions = np.hstack(dropout_predictions)
    #print(dropout_predictions.shape)
    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(dropout_predictions, axis=1)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=1)  # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)
    
    return mean, variance, entropy

    
def make_outcome_feature(x, train_indices, y):
    """
    Make a feature to propagate with non zero value only for the train indices
    """
    mask = torch.ones(y.size(0), dtype=torch.bool)
    mask[train_indices] = 0
    y[mask] = 0
    return y
    


def uplift_score(prediction, treatment, target, rate=0.2):
    """
    From https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()
    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score



def test_causalml(confounders: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, k: int, 
                  task: int=0, causal_model_type: str = 'X' , model_out: str = "XGB",random_seed:int = 0) -> list:
    """
    Test the causalml model in a kfold cross validation.
    """
    results = []

    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    results = []
    for train_indices, test_indices in kf.split(confounders):
        test_indices, train_indices = train_indices, test_indices

        
        up40, up20  = causalml_run(confounders[train_indices], outcome[train_indices] , treatment[train_indices], 
                                        confounders[test_indices], outcome[test_indices], treatment[test_indices], task, causal_model_type, model_out)
        results.append((up40, up20))

    return pd.DataFrame(results)
    


def causalml_run(confounders_train: np.ndarray ,outcome_train: np.ndarray, treatment_train: np.ndarray, 
                 confounders_test: np.ndarray, outcome_test: np.ndarray, treatment_test: np.ndarray, 
                 task :int = 0, causal_model_type: str ='X', model_class: str ="XGB", model_regr :str="XGBR", total_uplift=False) -> tuple:
    """
    Run the causalml model on the train and test data using the given models for causal ml, propensity, outcome and effect. 
    """
    dic_mod = {"XGB":XGBClassifier, "LR":LogisticRegression, "XGBR":XGBRegressor}

    if causal_model_type == 'S':
        if task ==0:
            learner = BaseSClassifier(learner=dic_mod[model_class]())
        else:
            learner = BaseSRegressor(learner=dic_mod[model_regr]())

        learner.fit( X = confounders_train, y = outcome_train , treatment = treatment_train ) 
        
        if total_uplift:
            uplift=learner.predict(X = np.vstack([confounders_train,confounders_test]), treatment= np.hstack([treatment_train,treatment_test])).squeeze()
        else:
            uplift=learner.predict(X = confounders_test, treatment= treatment_test).squeeze()
        
    elif causal_model_type == 'T':
        if task==0:
            learner = BaseTClassifier(learner = dic_mod[model_class]())
        else:
            learner = BaseTRegressor(learner = dic_mod[model_regr]())

        learner.fit(X= confounders_train, y=outcome_train , treatment= treatment_train)  
        uplift=learner.predict(X = confounders_test, treatment = treatment_test).squeeze()
        if total_uplift:
            uplift=learner.predict(X = np.vstack([confounders_train,confounders_test]), treatment= np.hstack([treatment_train,treatment_test])).squeeze()
        else:
            uplift=learner.predict(X = confounders_test, treatment= treatment_test).squeeze()

    elif causal_model_type == 'X':
        propensity_model = ElasticNetPropensityModel() #dic_mod[model_class]()

        propensity_model.fit(X=confounders_train, y = treatment_train)
        p_train = propensity_model.predict(X=confounders_train)
        p_test = propensity_model.predict(X=confounders_test)

        #p_train = pd.Series(p_train[:, 0]).values
        #p_test  = pd.Series(p_test[:, 0]).values        

        if task==0:
            learner = BaseXClassifier(
                outcome_learner=dic_mod[model_class](),
                effect_learner=dic_mod[model_regr]())
        else:
            learner = BaseXRegressor(
                learner=dic_mod[model_regr]())
        
        learner.fit(X = confounders_train , y = outcome_train, treatment = treatment_train, p=p_train)  
        
        if total_uplift:
            uplift=learner.predict(X = np.vstack([confounders_train,confounders_test]), treatment= np.hstack([treatment_train,treatment_test]) , p=np.hstack([p_train,p_test])).squeeze()
        else:
            uplift=learner.predict(X = confounders_test, treatment= treatment_test, p=p_test).squeeze()
     
    elif causal_model_type == 'R':
        propensity_model = ElasticNetPropensityModel() #dic_mod[model_class]()

        propensity_model.fit(X=confounders_train, y = treatment_train)
        p_train = propensity_model.predict(X=confounders_train)
        p_test = propensity_model.predict(X=confounders_test)

        if task==0:
            learner = BaseRClassifier(
                outcome_learner=dic_mod[model_class](),
                effect_learner=dic_mod[model_regr]())
        else:
            learner = BaseRRegressor(
                learner=dic_mod[model_regr]())
        
        learner.fit(X = confounders_train , y = outcome_train, treatment = treatment_train, p=p_train)  
        
        if total_uplift:
            uplift=learner.predict(X = np.vstack([confounders_train,confounders_test]), treatment= np.hstack([treatment_train,treatment_test]) , p=np.hstack([p_train,p_test])).squeeze()
        else:
            uplift=learner.predict(X = confounders_test).squeeze()

    elif causal_model_type == 'D':
        propensity_model = ElasticNetPropensityModel()#dic_mod[model_class]()

        propensity_model.fit(X=confounders_train, y = treatment_train)
        p_train = propensity_model.predict(X=confounders_train)
        p_test = propensity_model.predict(X=confounders_test)

        if task==0:
            learner = BaseDRRegressor(
                learner=dic_mod[model_class](),
                treatment_effect_learner = dic_mod[model_regr]() )
        else:
            learner = BaseDRRegressor(
                learner=dic_mod[model_regr](),
                treatment_effect_learner = dic_mod[model_regr]())
            
        learner.fit(X = confounders_train , y = outcome_train, treatment = treatment_train, p=p_train)  

        if total_uplift:
            uplift=learner.predict(X = np.vstack([confounders_train,confounders_test]), treatment= np.hstack([treatment_train,treatment_test]) , p=np.hstack([p_train,p_test])).squeeze()
        else:
            uplift=learner.predict(X = confounders_test, treatment= treatment_test, p=p_test).squeeze()
    
    elif causal_model_type == 'Tree':
        if task ==0:
            learner = UpliftTreeClassifier(control_name="0")
        else:
            learner = CausalTreeRegressor(control_name="0")
        X_train = np.hstack(( treatment_train.reshape(-1, 1), confounders_train))
        X_test = np.hstack((treatment_test.reshape(-1, 1), confounders_test))
        learner.fit( X = X_train, treatment = treatment_train.astype(str), y=outcome_train)

        if total_uplift:
            uplift = learner.predict( X = np.vstack([X_train,X_test] ).squeeze())
            uplift = uplift.argmax(1)
        else:
            uplift = learner.predict( X = X_test).squeeze()
        

    elif causal_model_type == 'Dragon':
        if task==0:
            learner = DragonNet()
        else:
            learner = DragonNet(loss_func=regression_loss)
        learner.fit(X = confounders_train, treatment= treatment_train, y = outcome_train.astype(np.float32) )
        if total_uplift:
            uplift = learner.predict(X = np.vstack([confounders_train,confounders_test]) , treatment= np.hstack([treatment_train, treatment_test])) 
        else:
            uplift = learner.predict(X = confounders_test, treatment= treatment_test)      
        uplift = uplift[:,1] - uplift[:,0]

    elif causal_model_type == 'CEVAE':
        if task==0:
            learner = CEVAE()
        else:
            learner = CEVAE()
        learner.fit(X = torch.tensor(confounders_train, dtype=torch.float), treatment= torch.tensor(treatment_train, dtype=torch.float) , 
                    y = torch.tensor(outcome_train, dtype=torch.float))
        
        if total_uplift:
            uplift = learner.predict(X = np.vstack([confounders_train,confounders_test]) , treatment= np.hstack([treatment_train, treatment_test])) 
        else:
            uplift = learner.predict(X = confounders_test, treatment= treatment_test)                                                  
       

    if total_uplift:
        score40 = uplift_score(uplift, np.hstack([treatment_train,treatment_test]), np.hstack([outcome_train,outcome_test]), rate=0.4)#uplift_at_k(y_true = outcome_test, uplift=uplift, treatment= treatment_test, strategy='by_group', k=0.4)
        score20 = uplift_score(uplift, np.hstack([treatment_train,treatment_test]), np.hstack([outcome_train,outcome_test]), rate=0.2)#uplift_at_k(y_true = outcome_test, uplift=uplift, treatment= treatment_test, strategy='by_group', k=0.2)
                     
    else:
        score40 = uplift_score(uplift, treatment_test, outcome_test, rate=0.4)#uplift_at_k(y_true = outcome_test, uplift=uplift, treatment= treatment_test, strategy='by_group', k=0.4)
        score20 = uplift_score(uplift, treatment_test, outcome_test, rate=0.2)#uplift_at_k(y_true = outcome_test, uplift=uplift, treatment= treatment_test, strategy='by_group', k=0.2)
    return score40, score20



