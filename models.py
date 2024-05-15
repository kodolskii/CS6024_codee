# Building model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.optim.lr_scheduler import MultiStepLR

#CoGN
import numpy as np
from torch import nn
import os
from tqdm import tqdm
from ase.io import read
from itertools import repeat
import numpy as np
import json
from tqdm import tqdm
import math
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from pymatgen.core.periodic_table import Element
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import collate
import pandas as pd
import numpy as np
from torch_geometric.nn.models import MLP
from torch_scatter import scatter

class GCNN(nn.Module):
    def __init__(self, n_output=1, num_features_pro= 1024, output_dim=128, dropout=0.2):
        super(GCNN, self).__init__()

        print('GCNN Loaded')

        # for protein 1
        self.n_output = n_output
        self.pro1_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro1_fc1 = nn.Linear(num_features_pro, output_dim)

        # for protein 2
        self.pro2_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro2_fc1 = nn.Linear(num_features_pro, output_dim)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256 ,64)
        self.out = nn.Linear(64, self.n_output)

    def forward(self, pro1_data, pro2_data):

        #get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch


        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        
	# global pooling
        x = gep(x, pro1_batch)   

        # flatten
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)



        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(xt)

	# global pooling
        xt = gep(xt, pro2_batch)  

        # flatten
        xt = self.relu(self.pro2_fc1(xt))
        xt = self.dropout(xt)


	# Concatenation  
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out
        

net = GCNN()
print(net)

"""# GAT"""

class AttGNN(nn.Module):
    def __init__(self, n_output=1, num_features_pro= 1024, output_dim=128, dropout=0.2, heads = 1 ):
        super(AttGNN, self).__init__()

        print('AttGNN Loaded')

        self.hidden = 8
        self.heads = 1
        
        # for protein 1
        self.pro1_conv1 = GATConv(num_features_pro, self.hidden* 16, heads=self.heads, dropout=0.2)
        self.pro1_fc1 = nn.Linear(128, output_dim)


        # for protein 2
        self.pro2_conv1 = GATConv(num_features_pro, self.hidden*16, heads=self.heads, dropout=0.2)
        self.pro2_fc1 = nn.Linear(128, output_dim)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, n_output)
        


    def forward(self, pro1_data, pro2_data):

        # get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch
         
        
        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        
	# global pooling
        x = gep(x, pro1_batch)  
       
        # flatten
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)



        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(self.pro2_fc1(xt))
	
	# global pooling
        xt = gep(xt, pro2_batch)  

        # flatten
        xt = self.relu(xt)
        xt = self.dropout(xt)

	
	# Concatenation
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out

net_GAT = AttGNN()
print(net_GAT)

class CoGN_Model(nn.Module):
    class AtomEmbedding(nn.Module):
        def __init__(self, num_class, emb_dim):
            super(CoGN_Model.AtomEmbedding, self).__init__()
            self.embedding_layer = nn.Embedding(num_class, emb_dim)
        
        def forward(self, inputs):
            return self.embedding_layer(inputs)
    
    class GaussBasisExpansion(nn.Module):
        def __init__(self, mu, sigma, **kwargs):
            super().__init__(**kwargs)
            mu = torch.unsqueeze(torch.FloatTensor(mu), 0)
            self.register_buffer("mu", mu)
            sigma = torch.unsqueeze(torch.FloatTensor(sigma), 0)
            self.register_buffer("sigma", sigma)
        
        @classmethod
        def from_bounds(cls, n: int, low: float, high: float, variance: float = 1.0):
            mus = np.linspace(low, high, num=n + 1)
            var = np.diff(mus)
            mus = mus[1:]
            return cls(mus, np.sqrt(var * variance))
        
        @classmethod
        def from_bounds_log(cls, n: int, low: float, high: float, base: float = 32, variance: float = 1):
            mus = (np.logspace(0, 1, num=n + 1, base=base) - 1) / (base - 1) * (high - low) + low
            var = np.diff(mus)
            mus = mus[1:]
            return cls(mus, np.sqrt(var * variance))
        
        def forward(self, x, **kwargs):
            return torch.exp(-torch.pow(x - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
    
    class EdgeEmbedding(nn.Module):
        def __init__(self, bins_distance=32, max_distance=5.0, distance_log_base=1.0, gaussian_width_distance=1.0):
            super().__init__()
            if distance_log_base == 1.0:
                self.distance_embedding = CoGN_Model.GaussBasisExpansion.from_bounds(
                    bins_distance, 0.0, max_distance, variance=gaussian_width_distance
                )
            else:
                self.distance_embedding = CoGN_Model.GaussBasisExpansion.from_bounds_log(
                    bins_distance, 0.0, max_distance, base=distance_log_base, variance=gaussian_width_distance,
                )
        
        def forward(self, distance):
            d = torch.unsqueeze(distance, 1)
            return self.distance_embedding(d)
    
    class Block(nn.Module):
        def __init__(self, node_mlp, edge_mlp, global_mlp, block, aggregate_edges_local="sum"):
            super().__init__()
            self.aggregate_edges_local = aggregate_edges_local
            self.block = block
            
            if edge_mlp:
                self.edgemlp = nn.Sequential(
                    nn.Linear(edge_mlp["input_dim"], edge_mlp["hidden_dim_list"][0]),
                    nn.ReLU(),
                    nn.Linear(edge_mlp["hidden_dim_list"][0], edge_mlp["hidden_dim_list"][0])
                )
            
            if node_mlp:
                self.nodemlp = nn.Sequential(
                    nn.Linear(node_mlp["input_dim"], node_mlp["hidden_dim_list"][0]),
                    nn.ReLU(),
                    nn.Linear(node_mlp["hidden_dim_list"][0], node_mlp["input_dim"])
                )
            
            if global_mlp:
                self.globalmlp = nn.Sequential(
                    nn.Linear(global_mlp["input_dim"], global_mlp["hidden_dim_list"][0]),
                    nn.ReLU(),
                    nn.Linear(global_mlp["hidden_dim_list"][0], global_mlp["hidden_dim_list"][0])
                )
        
        def update_edges(self, x, edge_index, edge_attr):
            node_in = x[edge_index[0].to(int)]
            node_out = x[edge_index[1].to(int)]
            concat_feature = torch.cat((edge_attr, node_in, node_out), dim=-1)
            edge_attr_new = self.edgemlp(concat_feature)
            return edge_attr_new
        
        def update_nodes(self, x, edge_index, edge_attr):
            node_index = edge_index[0]
            aggregated_edge = scatter(edge_attr, node_index, dim=0, reduce=self.aggregate_edges_local)
            x = x + self.nodemlp(aggregated_edge)
            return x
        
        def update_global(self, x, edge_index):
            x_mean = torch.mean(x, dim=0)
            global_out = self.globalmlp(x_mean)
            return global_out
        
        def forward(self, x, edge_index, edge_attr):
            if self.block == "processing":
                edge_attr = self.update_edges(x, edge_index, edge_attr)
                x = self.update_nodes(x, edge_index, edge_attr)
                return x, edge_attr
            if self.block == "output":
                return self.update_global(x, edge_index)
    
    def __init__(self, node_class, emb_dim=128, num_layer=5, bins_distance=32, distance_cutoff=5):
        super().__init__()
        self.atom_embedding = CoGN_Model.AtomEmbedding(node_class, emb_dim)
        self.edge_embedding = CoGN_Model.EdgeEmbedding(
            bins_distance=bins_distance, max_distance=distance_cutoff
        )
        
        self.atom_mlp = nn.Linear(emb_dim, emb_dim)
        self.edge_mlp = nn.Linear(bins_distance, emb_dim)
        
        processing_block_cfg = {
            'edge_mlp': {'input_dim': emb_dim * 3, 'hidden_dim_list': [emb_dim] * 5, 'activation': 'silu'},
            'node_mlp': {'input_dim': emb_dim, 'hidden_dim_list': [emb_dim], 'activation': 'silu'},
            'global_mlp': None,
            'aggregate_edges_local': 'sum',
            "block": "processing"
        }
        
        output_block_cfg = {
            'edge_mlp': None,
            'node_mlp': None,
            'global_mlp': {'input_dim': emb_dim, 'hidden_dim_list': [1], 'activation': None},
            "block": "output"
        }
        
        self.num_layer = num_layer
        self.processing_layers = nn.ModuleList([
            CoGN_Model.Block(**processing_block_cfg) for _ in range(self.num_layer)
        ])
        self.output_layer = CoGN_Model.Block(**output_block_cfg)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, edge_index, edge_attr):
        node = self.atom_embedding(x)
        node = self.atom_mlp(node)
        
        edge = self.edge_embedding(edge_attr)
        edge = self.edge_mlp(edge)
        
        for i in range(self.num_layer):
            node, edge = self.processing_layers[i](node, edge_index, edge)
        
        out = self.output_layer(node, edge_index, edge)
        out = self.sigmoid(out)
        return out

# Instantiate and print the model
net_CoGN = CoGN_Model(node_class=100)  # Adjust `node_class` as necessary
print(net_CoGN)

