import torch


class SortEdgeIndexTransform(object):
    def __call__(self, data):
        edge_index = data.edge_index

        # Get the source and target nodes from the edge_index
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        # First sort by source nodes, then by target nodes for the same source
        sorted_indices = torch.argsort(source_nodes)  # Sort by source node first
        source_sorted = edge_index[:, sorted_indices]  # Apply sorting to edge_index

        # For nodes with the same source, sort by target node
        source_nodes_sorted = source_sorted[0]  # Already sorted by source
        target_nodes_sorted = source_sorted[1]
        final_sorted_indices = torch.argsort(target_nodes_sorted[source_nodes_sorted == source_nodes_sorted])

        # Apply final sorting to the edge_index
        sorted_edge_index = edge_index[:, final_sorted_indices]

        data.edge_index = sorted_edge_index
        return data