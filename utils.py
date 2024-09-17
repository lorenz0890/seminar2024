import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


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