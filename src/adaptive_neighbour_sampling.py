import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveNeighbourSampling(nn.Module):
    def __init__(self, similarity_threshold):
        """
        Adaptive Neibour Sampling Module
        This module implement a sampling strategy from ASA-GNN that:
        1. Filter a noisy neighbour based on cosine similarity and edge weight
        2. Over-sample for fraudulent node to enrich information
        """
        super(AdaptiveNeighbourSampling, self).__init__()
        self.similarity_threshold = similarity_threshold 

    
    def cosine_similarity(self, target_node, neighbor_nodes):
        """Compute cosine similarity between two sets of node features."""
        if target_node.dim() == 1:
            target_node = target_node.unsqueeze(0)
    
    
        target_norm = F.normalize(target_node, p=2, dim=-1)
        neighbors_norm = F.normalize(neighbor_nodes, p=2, dim=-1)
        
        return torch.mm(neighbors_norm, target_norm.t()).squeeze()

    def get_edge_weight(self,adjacency_matrix,node_idx,neighbor_indices):
        """Get edge weight from adjacency matrix. Will need to improve something here"""
        weight = adjacency_matrix[node_idx,neighbor_indices]
        return weight
    
    def compute_sampling_prob(self,current_node, neighbour_node, edge_weights):
        """Compute sampling probability based on similarity and edge weight."""
        similarity = self.cosine_similarity(current_node, neighbour_node)
        weighted_similarity = similarity * edge_weights
        prob = weighted_similarity / torch.sum(weighted_similarity)
        return prob
    
    def topk_sampling(self, probabilities, k):
        """Select top-k indices based on probabilities."""
        if len(probabilities) <= k:
            return torch.arange(len(probabilities))
        else:
            _, topk_indices = torch.topk(probabilities, k)
            return topk_indices
    
    def sample_neighbors(self,
                        adj_matrix: torch.Tensor,
                        current_transaction: torch.Tensor,
                        transaction_record: torch.Tensor,
                        labels: torch.Tensor,
                        node_idx: int) -> torch.Tensor:
        """
        Complete adaptive neighbor sampling for a single node
        """
        
        neighbor_mask = adj_matrix[node_idx] > 0
        neighbor_indices = torch.where(neighbor_mask)[0]
        
        if len(neighbor_indices) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # Get neighbor features and edge weights
        neighbor_features = transaction_record[neighbor_indices]
        edge_weights = self.get_edge_weight(adj_matrix, node_idx, neighbor_indices)
        
        # Compute selection probabilities
        probabilities = self.compute_sampling_prob(
            current_transaction, neighbor_features, edge_weights
        )
        
        # Top-k sampling to filter noisy neighbors
        selected_local_indices = self.topk_sampling(probabilities, self.sample_size)
        selected_neighbors = neighbor_indices[selected_local_indices]
        
        return selected_neighbors
     
    def forward(self, adjacency_matrix, transaction_record, labels):
        num_nodes = transaction_record.size(0)
        neighbor_dict = {}
        for node_idx in range(num_nodes):
            neighbors = self.sample_neighbors(
                adjacency_matrix,
                transaction_record[node_idx],
                transaction_record,
                labels,
                node_idx
            )
            neighbor_dict[node_idx] = neighbors
        
        return neighbor_dict