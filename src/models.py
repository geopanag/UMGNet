import os
import numpy as np
import torch
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import torch.nn as nn
from torch import nn, optim, Tensor

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import MessagePassing


class BipartiteDraGNN(torch.nn.Module):
    def __init__(self, nfeat:int, nproduct:int , hidden_channels:int , out_channels: int, num_layers:int, dropout_rate:float =0):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        self.user_embed = nn.Linear(nfeat, hidden_channels )
        self.item_embed =  nn.Linear(nproduct, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv((-1,-1), hidden_channels))
            
        #self.lin_hetero = Linear(hidden_channels, out_channels)
        
        self.num_layers = num_layers

        self.hidden_common1 = nn.Linear(hidden_channels + num_layers*hidden_channels, hidden_channels)
        self.hidden_common2 = nn.Linear(hidden_channels, hidden_channels)

        self.hidden_control = nn.Linear(hidden_channels, int(hidden_channels/2))
        self.hidden_treatment = nn.Linear(hidden_channels, int(hidden_channels/2))
        self.hidden_T = nn.Linear(hidden_channels, int(hidden_channels/2))

        self.out_control = nn.Linear( int(hidden_channels/2), out_channels)
        self.out_treatment = nn.Linear( int(hidden_channels/2), out_channels)
        self.out_T = nn.Linear( int(hidden_channels/2), out_channels)

        #self.lin = Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        #self.bn_hidden = nn.BatchNorm1d(hidden_channels)
        
        #self.bn_out = nn.BatchNorm1d(nfeat + hidden_channels + hidden_channels)
        self.activation = nn.ReLU()


    def forward(self, xu: torch.tensor, xp:torch.tensor, edge_index:torch._tensor):
        out = [] 
        xu = self.user_embed(xu)
        xp = self.item_embed(xp)

        out.append(xu)

        embeddings = torch.cat((xu,xp), dim=0) 
        
        for i in range(self.num_layers):
            embeddings = self.activation(self.convs[i](embeddings, edge_index))
            #embeddings = self.dropout(embeddings)
            #embeddings = self.bn_hidden(embeddings)
            
            out.append(embeddings[:xu.shape[0]])            
        
        out = torch.cat( out, dim=1)
        
        hidden = self.dropout(self.activation(self.hidden_common1(out)))
        hidden = self.dropout(self.activation(self.hidden_common2(hidden)))
        
        # separate treatment and control 
        hidden_1t0 = self.dropout(self.activation(self.hidden_control(hidden)))
        hidden_1t1 = self.dropout(self.activation(self.hidden_treatment(hidden)))
        hidden_1T = self.dropout(self.activation(self.hidden_T(hidden)))

        out_2t0 = self.activation(self.out_control(hidden_1t0))
        out_2t1 = self.activation(self.out_treatment(hidden_1t1))
        out_2T = self.activation(self.out_T(hidden_1T))
        
        
        return out_2t1, out_2t0, out_2T, hidden_1t1, hidden_1t0
    


class UserMP(MessagePassing):
    def __init__(self, aggr='add',normed=False, **kwargs):
        super(UserMP, self).__init__(aggr=aggr,node_dim=-1, **kwargs)
        self.normed = normed

    def forward(self, x, edge_index):
        #size=(label_for_prop.shape[0],dummy_product_labels.shape[0]) 
        return self.propagate(edge_index,  x=x)
        
    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        #  Return only the aggregation of the neighbors not the node's feature 
        return aggr_out
    


class BipartiteSAGE2mod(torch.nn.Module):
    def __init__(self, nfeat:int, nproduct:int , hidden_channels:int , out_channels: int, num_layers:int, dropout_rate:float =0):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        self.user_embed = nn.Linear(nfeat, hidden_channels )
        self.item_embed =  nn.Linear(nproduct, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv((-1,-1), hidden_channels))
            
        #self.lin_hetero = Linear(hidden_channels, out_channels)
        
        self.num_layers = num_layers

        self.hidden_common1 = nn.Linear(hidden_channels + num_layers*hidden_channels, hidden_channels)
        self.hidden_common2 = nn.Linear(hidden_channels, hidden_channels)

        self.hidden_control = nn.Linear(hidden_channels, int(hidden_channels/2))
        self.hidden_treatment = nn.Linear(hidden_channels, int(hidden_channels/2))

        self.out_control = nn.Linear( int(hidden_channels/2), out_channels)
        self.out_treatment = nn.Linear( int(hidden_channels/2), out_channels)

        #self.lin = Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        #self.bn_hidden = nn.BatchNorm1d(hidden_channels)
        
        #self.bn_out = nn.BatchNorm1d(nfeat + hidden_channels + hidden_channels)
        self.activation = nn.ReLU()


    def forward(self, xu: torch.tensor, xp:torch.tensor, edge_index:torch._tensor):
        out = [] 
        xu = self.user_embed(xu)
        xp = self.item_embed(xp)

        out.append(xu)

        embeddings = torch.cat((xu,xp), dim=0) 
        
        for i in range(self.num_layers):
            embeddings = self.activation(self.convs[i](embeddings, edge_index))
            #embeddings = self.dropout(embeddings)
            #embeddings = self.bn_hidden(embeddings)
            
            out.append(embeddings[:xu.shape[0]])            
        
        out = torch.cat( out, dim=1)
        
        hidden = self.dropout(self.activation(self.hidden_common1(out)))
        hidden = self.dropout(self.activation(self.hidden_common2(hidden)))
        
        # separate treatment and control 
        hidden_1t0 = self.dropout(self.activation(self.hidden_control(hidden)))
        hidden_1t1 = self.dropout(self.activation(self.hidden_treatment(hidden)))

        out_2t0 = self.activation(self.out_control(hidden_1t0))
        out_2t1 = self.activation(self.out_treatment(hidden_1t1))
        
        
        return out_2t1, out_2t0, hidden_1t1, hidden_1t0
    


class GNN(torch.nn.Module):
    def __init__(self,nfeat: int , hidden1: int, edge_index: Tensor, class_no: int=2):
        super(GNN, self).__init__()

        self.A = edge_index
        hidden2 = 2*hidden1
        self.gcn1 = GCNConv(nfeat,hidden1)
        #self.bn1 = nn.BatchNorm1d(hidden1)
        #self.gcn2 = GCNConv(hidden1,hidden2)
        #self.bn2 = nn.BatchNorm1d(hidden3)
        self.fc = nn.Linear(nfeat+hidden1,class_no)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor):
        out = []
        out.append(x)
        x = self.relu(self.gcn1( x, self.A))
        #out.append(x)
        #x = self.relu(self.gcn2( x, self.A))
        out.append(x)
        out = torch.cat(out, dim=1)
        return self.fc(out)
    



class MLP(torch.nn.Module):
    def __init__(self,n_feat, hidden1,class_no=2):
        super(MLP, self).__init__()
        self.fc = nn.Linear(n_feat,hidden1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1,class_no)

    def forward(self,x):
        x = self.fc(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out
