import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveNeighbourAggregation(nn.Module):
    def __init__(self, in_features):
        super(AdaptiveNeighbourAggregation, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.W = nn.Linear(2 * in_features, in_features)
        self.relu = nn.ReLU()

    def neighbour_diversity_score(self, h_neighbours):
        """ Not sure about this function yet"""
        return
    
    def forward(self, h_v, h_neighbours):
        """ Compute the value for h_v during the k-th layer"""
        D_v = self.neighbour_diversity_score(h_neighbours)
        D_v_norm = self.softmax(D_v)
        gate_value = self.sigmoid(-D_v_norm)
        gated_neighbors = gate_value * h_neighbours
        concat_features = torch.cat([h_v, gated_neighbors], dim=-1)
        h_v_updated = self.relu(self.W(concat_features))

        return h_v_updated