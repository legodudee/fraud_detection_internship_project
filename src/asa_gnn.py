import torch
import torch.nn as nn
from asa_gnn_layer import ASAGNNLayer
from adaptive_neighbour_sampling import AdaptiveNeighbourSampling

class ASAGNN(nn.Module):
    def __init__(self,in_features, num_layers):
        """ Adaptive Sampling and Aggregation GNN Model
        This model stacks multiple ASAGNN layers to perform node classification or regression tasks.
        """
        super(ASAGNN, self).__init__()
        self.k = num_layers
        self.layer = ASAGNNLayer(in_features)
        self.sampler = AdaptiveNeighbourSampling(similarity_threshold=0.5)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, adj_matrix, transaction_record, labels):
        """ Forward pass for the ASAGNN model
        adj_matrix: Adjacency matrix of the graph
        transaction_record: Feature matrix of the nodes
        labels: Labels for supervised learning
        """
        num_nodes = transaction_record.size(0)
        neighbour_dict = self.sampler.forward((adj_matrix, transaction_record, labels))
        h = transaction_record
        for i in range(self.k):
            for node_idx in range(num_nodes):
                h[node_idx] = self.layer.forward(h[node_idx], neighbour_dict)
        probabilities = self.softmax(h)
        return probabilities
