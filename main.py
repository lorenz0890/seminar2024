import json

import numpy as np
import torch
from sklearn.preprocessing import KBinsDiscretizer
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9, TUDataset
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.transforms import NormalizeFeatures

from models import *
from hooks import *
from utils import get_graph_embeddings, cluster_and_visualize, project_and_visualize, visualize_2d_embeddings, \
    prune_model_randomly, visualize_2d_embeddings_color, plot_regression_target_distribution, apply_wlconv, \
    compute_intra_inter_cluster_similarity, replace_nan_with_none, replace_none_with_nan
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error

from sklearn.svm import SVR

import matplotlib.pyplot as plt

from visualization import plot_similarity


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)


def main():
    # Load the MUTAG dataset from TUDataset
    # TODO increase atent space dimensionality
    #dataset_name = 'KKI' #'MUTAG', 'PTC_FM', 'PTC_MR', 'KKI', 'ENZYMES', 'PROTEINS'
    #model_name = 'GCN' #'GCN', 'GIN'
    for dataset_name in ['MUTAG', 'KKI', 'PTC_FM', 'PTC_MR']:
        for model_name in ['GIN', 'GCN']:
            print(dataset_name, model_name)
            run_experiments = True
            if run_experiments:

                dataset = TUDataset(root='./data/TUDataset', name=dataset_name, transform=NormalizeFeatures())
                # Shuffle and split dataset into training and test sets (80% train, 20% test)
                train_size = int(0.8 * len(dataset))
                test_size = len(dataset) - train_size
                train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

                # Set up data loaders for batching
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                # Set device (use GPU if available)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Initialize GIN model
                if model_name == 'GIN':
                    model = GIN(input_dim=dataset.num_features, hidden_dim=32, output_dim=dataset.num_classes)  # 32, 128
                if model_name == 'GCN':
                    model = GCN(input_dim=dataset.num_features, hidden_dim=32, output_dim=dataset.num_classes)
                # model = MLP(input_dim=dataset.num_features, hidden_dim=32, output_dim=2)
                model = model.to(device)

                # Set up optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                results = {
                    1 : {'intra' : [], 'inter' : [], 'cluster' : []},
                    2: {'intra': [], 'inter': [], 'cluster' : []},
                    3: {'intra': [], 'inter': [], 'cluster' : []},
                    'test_acc' : [],
                    'train_acc': [],
                }
                # Training loop
                for epoch in range(0, 20):  # Train for 200 epochs
                    loss = -1
                    if epoch > 0: #get pre trining values
                        loss = train(model, train_loader, optimizer, device)
                    train_acc = test(model, train_loader, device)
                    test_acc = test(model, test_loader, device)

                    if True: #epoch % 10 == 0 or epoch == 0:
                        # Store embeddings for each graph in the dataset
                        embeddings_by_graph = []
                        hook_handles = []
                        register_hooks(model, hook_handles)
                        model.eval()  # Set the model to evaluation mode
                        loader = DataLoader(dataset, batch_size=1, shuffle=False)
                        with torch.no_grad():
                            for data in loader:
                                data = data.to(device)
                                # Run the forward pass through the model (hooks will store embeddings)
                                _ = model(data.x, data.edge_index, data.batch)

                                # Store the embeddings after each convolution layer for the current graph
                                graph_embeddings = {'conv1': hook_fn.embeddings[0],
                                                    'conv2': hook_fn.embeddings[1],
                                                    'conv3': hook_fn.embeddings[2],
                                                    'edge_index' : data.edge_index,
                                                    'x' : data.x
                                                    }
                                embeddings_by_graph.append(graph_embeddings)

                                # Clear the embeddings list for the next graph
                                hook_fn.embeddings.clear()
                        print(len(embeddings_by_graph))
                        apply_wlconv(embeddings_by_graph, num_iterations=3)

                        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                        for depth in range(1,4):
                            intra_similarity, inter_similarity, cluster = [], [], []
                            for graph_embeddings in embeddings_by_graph:
                                similarities = compute_intra_inter_cluster_similarity(graph_embeddings, depth=depth)
                                intra_similarity.append(similarities[0])
                                inter_similarity.append(similarities[1])
                                cluster.append(similarities[2])

                            #print(depth, np.mean(intra_similarity), np.mean(inter_similarity))
                            results[depth]['intra'].append(np.mean(intra_similarity))
                            results[depth]['inter'].append(np.mean(np.mean(inter_similarity)))
                            results[depth]['cluster'].append(np.mean(np.mean(cluster)))
                            #colors = torch.unique(graph_embeddings["wlconv_embeddings_{}".format(depth)]).shape[0]
                            print(f'Depth: {depth:03d}, Intra: {np.mean(intra_similarity):.4f}, Inter: {np.mean(inter_similarity):.4f}, Cluster: {np.mean(cluster):.4f}')
                        results['test_acc'].append(test_acc)
                        results['train_acc'].append(train_acc)
                        remove_hooks(hook_handles)

                # Call the function to save the plot to disk
                results_cleaned = replace_nan_with_none(results)
                with open("results_{}_{}.json".format(model_name, dataset_name), 'w') as f:
                    json.dump(results_cleaned, f, indent=4)
            else:
                with open("results_{}_{}.json".format(model_name, dataset_name), 'r') as f:
                    loaded_results = json.load(f)
                results = replace_none_with_nan(loaded_results)
                print(results.keys())
                f = lambda k: k if k == 'test_acc' else int(k)
                results = {f(k) : v for k, v in results.items()}


            plot_similarity(results, filename="similarity_plot_{}_{}.png".format(model_name, dataset_name))
if __name__ == '__main__':
    main()
