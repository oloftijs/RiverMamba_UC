
# ------------------------------------------------------------------
"""
Graph Serialization Utility for SubMamba
Computes topological vertex ordering (e.g. DFS, BFS) for graph-based SSM traversal.


"""
# ------------------------------------------------------------------

import torch
import numpy as np


def get_graph_vertex_order(adjacency_matrix=None, method='dfs', num_nodes=472):
    """
    Generates the topological vertex ordering tensor of shape [1, P] or [B, P].

    Args:
        adjacency_matrix (torch.Tensor or np.ndarray, optional): Adjacency matrix of the graph.
        method (str): Traversal method ('dfs', 'bfs', or 'sequential').
        num_nodes (int): Number of nodes (stations) in the graph (default: 472).

    Returns:
        torch.Tensor: Vertex order tensor of shape [1, num_nodes]
    """
    if method == 'dfs':
        # Placeholder DFS topological order
        order = torch.arange(num_nodes, dtype=torch.long)
    elif method == 'bfs':
        # Placeholder BFS topological order
        order = torch.arange(num_nodes, dtype=torch.long)
    else:
        order = torch.arange(num_nodes, dtype=torch.long)

    # Return with batch/curve dimension [1, P]
    return order.unsqueeze(0)


if __name__ == '__main__':
    # Quick test
    order_tensor = get_graph_vertex_order(method='dfs', num_nodes=472)
    print("Graph vertex order tensor shape:", order_tensor.shape)
    print("First 10 indices:", order_tensor[0, :10])
