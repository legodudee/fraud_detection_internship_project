import torch
import torch.nn as nn
from attention_mechanism import AttentionMechanism
from adaptive_neighbour_aggregation import AdaptiveNeighbourAggregation

class ASAGNNLayer(nn.Module):
    def __init__(self,in_features):
        """ Adaptive Sampling and Aggregation GNN Layer
        This layer combines an attention mechanism with the adaptive neighbour aggreegation
        to update node features in a graph.
        """
        super(ASAGNNLayer, self).__init__()
        self.attention_mechanism = AttentionMechanism(in_features)
        self.adaptive_aggregator = AdaptiveNeighbourAggregation(in_features)
    
    def forward(self, h_target, neighbour_dict):
        """ Forward pass for the ASAGNN layer
        h_target: Feature vector of the target node
        neighbour_dict: Dictionary containing features of neighbor nodes
        """
        h_neighbors = torch.stack(list(neighbour_dict.values()), dim=0)
        h_aggregated = self.attention_mechanism.forward(h_target, h_neighbors)
        h_updated = self.adaptive_aggregator.forward(h_target, h_aggregated)
        
        return h_updated