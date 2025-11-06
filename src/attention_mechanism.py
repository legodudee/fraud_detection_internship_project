import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    def __init__(self, in_features):
        """ Attention Mechanism Module
        This module implements an attention mechanism to weigh the importance of neighbor nodes
        when aggregating their features for a target node.
        """
        super(AttentionMechanism, self).__init__()
        self.high_dimensional_mapping = nn.Linear(2 * in_features, 1, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        nn.init.xavier_uniform_(self.high_dimensional_mapping)

    def compute_attention_weights(self, h_target, h_neighbors):
        """Compute attention weights between target node and its neighbors."""
        
        n_neighbors = h_neighbors.size(0)
        h_target_repeated = h_target.unsqueeze(0).repeat(n_neighbors, 1)
        combined = torch.cat([h_target_repeated, h_neighbors], dim=1)
        e = self.high_dimensional_mapping(combined).squeeze()
        e_leaky = self.leaky_relu(e)
        attention_weights = e_leaky/torch.sum(e_leaky)
        
        return attention_weights
    
    def aggregate_neighbour_weighted_sum(self, h_neighbors, attention_weights):
        """Aggregate neighbour features together using weighted sum"""
        h_aggregated = torch.sum(attention_weights.unsqueeze(1) * h_neighbors, dim=0)
        return h_aggregated
    
    def forward(self, h_target, h_neighbors):
        """Forward pass to compute the attended representation of a target node."""
        attention_weights = self.compute_attention_weights(h_target, h_neighbors)
        h_aggregated = self.aggregate_neighbour_weighted_sum(h_neighbors, attention_weights)
        return h_aggregated

        

