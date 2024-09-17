import itertools

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch_geometric.nn import WLConv
from torchmetrics.functional import cosine_similarity


def cluster_and_visualize(embeddings, n_clusters=5, filename='clustered_embeddings.png'):
    # Step 1: Clustering using KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Step 2: Project embeddings to 2D space using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Step 3: Visualization and save to file
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.colorbar(scatter)
    plt.title('2D Projection of Graph Embeddings with Clusters')

    # Save the figure instead of showing it
    plt.savefig(filename)
    print(f"Image saved as {filename}")


def project_and_visualize(embeddings, filename='projected_embeddings.png'):
    # Step 1: Project embeddings to 2D space using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Step 2: Visualization and save to file
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], cmap='viridis', s=50)
    plt.colorbar(scatter)
    plt.title('2D Projection of Graph Embeddings')

    # Save the figure instead of showing it
    plt.savefig(filename)
    print(f"Image saved as {filename}")

def visualize_2d_embeddings(embeddings, filename='projected_embeddings.png'):
    # Step 1: Visualization and save to file (assuming embeddings are already 2D)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], cmap='viridis', s=50)
    plt.colorbar(scatter)
    plt.title('2D Visualization of Graph Embeddings')

    # Save the figure instead of showing it
    plt.savefig(filename)
    print(f"Image saved as {filename}")


def visualize_2d_embeddings_color(embeddings, regression_targets, filename='projected_embeddings.png'):
    """
    Visualizes the 2D embeddings with points colored according to the regression targets.

    Parameters:
    - embeddings: A 2D numpy array or tensor of shape (n_samples, 2).
    - regression_targets: The regression targets used to color the points.
    - filename: The filename for saving the plot.
    """
    # Step 1: Visualization and save to file (assuming embeddings are already 2D)
    plt.figure(figsize=(8, 6))

    # Create the scatter plot, with colors determined by the regression targets
    #embeddings = (embeddings - embeddings.mean())/embeddings.std()
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=regression_targets, cmap='viridis', s=50)

    # Add a colorbar to indicate the mapping of colors to target values
    plt.colorbar(scatter, label='Regression Target')

    # Add a title
    plt.title('2D Visualization of Graph Embeddings')

    # Save the figure instead of showing it
    plt.savefig(filename)
    print(f"Image saved as {filename}")


def plot_regression_target_distribution(regression_targets, filename='regression_target_distribution.png'):
    """
    Plots the distribution of regression targets.

    Parameters:
    - regression_targets: A 1D array or list of regression target values.
    - filename: The filename for saving the plot.
    """
    # Create the histogram for the distribution of the regression targets
    plt.figure(figsize=(8, 6))
    plt.hist(regression_targets, bins=30, color='blue', edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel('Regression Target Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Regression Targets')

    # Save the figure instead of showing it
    plt.savefig(filename)
    print(f"Regression target distribution saved as {filename}")
    plt.show()

# Function to get graph embeddings
def get_graph_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            emb = model(data.x, data.edge_index, data.batch)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def prune_model_randomly(model, pruning_percentage):
    """
    Prune a certain percentage of the model's weights by randomly setting them to zero.

    Parameters:
    model (torch.nn.Module): The neural network model.
    pruning_percentage (float): The percentage of weights to prune (between 0 and 100).
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Flatten the parameter tensor and get its total number of elements
            param_data = param.data.cpu().numpy()
            total_elements = param_data.size

            # Determine the number of elements to prune
            num_to_prune = int(total_elements * pruning_percentage / 100)

            # Randomly choose indices to set to 0
            indices = np.random.choice(total_elements, num_to_prune, replace=False)

            # Set the selected indices to 0
            flat_param_data = param_data.flatten()
            flat_param_data[indices] = 0

            # Reshape the flattened data back to its original shape
            pruned_param_data = flat_param_data.reshape(param_data.shape)

            # Assign the pruned weights back to the model's parameter
            param.data = torch.from_numpy(pruned_param_data).to(param.device)

            print(f"Pruned {num_to_prune} weights in parameter '{name}'.")


def map_features_to_integers(x):
    """
    Maps each unique feature vector in x to a unique integer.

    Args:
    x (torch.Tensor): A matrix where each row is a node feature vector.

    Returns:
    torch.Tensor: A vector of integers where each integer represents a unique node feature vector.
    """
    # Convert tensor to numpy for easier manipulation
    x_np = x.cpu().numpy()  # Convert to numpy array if necessary (ensure it's on CPU)

    # Find unique feature vectors and their corresponding indices
    unique_vectors, inverse_indices = np.unique(x_np, axis=0, return_inverse=True)

    # `inverse_indices` contains the integer mapping for each node's feature vector
    return torch.tensor(inverse_indices, dtype=torch.long)

def apply_wlconv(embeddings_by_graph, num_iterations=3):
    """
    Apply 3 iterations of WLConv on each graph's embeddings.
    """
    wl_conv = WLConv()  # WLConv layer from torch_geometric
    for graph_embeddings in embeddings_by_graph:
        edge_index = graph_embeddings['edge_index']

        #x = graph_embeddings['x']  # Or use 'x' to start with initial node features
        x = map_features_to_integers(graph_embeddings['x']).long().to(edge_index.get_device()) # Integer encoding for unique feature vectors
        #x = torch.ones(graph_embeddings['x'].shape[0]).long().to(edge_index.get_device())

        # Apply 3 iterations of WLConv
        for iteration in range(1, num_iterations+1):
            x = wl_conv(x, edge_index)  # Update node embeddings using WLConv

            # Store the updated embeddings after WLConv iterations
            graph_embeddings["wlconv_embeddings_{}".format(iteration)] = x

    return embeddings_by_graph

def compute_intra_inter_cluster_similarity(graph_embeddings, depth=3):
    """
    Compute intra-cluster and inter-cluster similarity using WL colors as cluster labels.
    """
    wl_colors = graph_embeddings["wlconv_embeddings_{}".format(depth)]
    node_embeddings = graph_embeddings["conv{}".format(depth)]  # Assume WL colors are stored in x (or wherever the color labels are)

    # Get unique cluster labels (WL colors)
    unique_labels = torch.unique(wl_colors)

    # Intra-cluster similarity: similarity between embeddings of nodes with the same WL color
    intra_cluster_similarities = []
    for label in unique_labels:
        indices = (wl_colors == label).nonzero(as_tuple=True)[0]  # Indices of nodes with this color
        if len(indices) > 1:  # Only compute if there's more than one node in the cluster
            pairs = list(itertools.combinations(indices, 2))  # All pairs within the cluster
            similarities = [
                cosine_similarity(node_embeddings[i].unsqueeze(0), node_embeddings[j].unsqueeze(0)).item()
                for i, j in pairs
            ]
            intra_cluster_similarities.extend(similarities)

    intra_cluster_similarity = sum(intra_cluster_similarities) / len(intra_cluster_similarities) if intra_cluster_similarities else 0

    # Inter-cluster similarity: similarity between embeddings of nodes with different WL colors
    inter_cluster_similarities = []
    for label_1, label_2 in itertools.combinations(unique_labels, 2):  # All pairs of clusters
        indices_1 = (wl_colors == label_1).nonzero(as_tuple=True)[0]
        indices_2 = (wl_colors == label_2).nonzero(as_tuple=True)[0]
        similarities = [
            cosine_similarity(node_embeddings[i].unsqueeze(0), node_embeddings[j].unsqueeze(0)).item()
            for i in indices_1 for j in indices_2
        ]
        inter_cluster_similarities.extend(similarities)

    inter_cluster_similarity = sum(inter_cluster_similarities) / len(inter_cluster_similarities) if inter_cluster_similarities else 0

    return intra_cluster_similarity, inter_cluster_similarity, unique_labels.shape[0]

# Helper function to replace NaN with None for JSON compatibility
def replace_nan_with_none(d):
    """Recursively replace NaN values with None (null in JSON) in a dictionary."""
    if isinstance(d, dict):
        return {k: replace_nan_with_none(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_nan_with_none(v) for v in d]
    elif isinstance(d, float) and np.isnan(d):
        return None
    return d

# Helper function to replace None (null in JSON) with NaN in the dictionary
def replace_none_with_nan(d):
    """Recursively replace None values with NaN in a dictionary."""
    if isinstance(d, dict):
        return {k: replace_none_with_nan(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_none_with_nan(v) for v in d]
    elif d is None:
        return np.nan
    return d