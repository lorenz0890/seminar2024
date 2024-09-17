import torch
from sklearn.preprocessing import KBinsDiscretizer
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9, TUDataset
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.transforms import NormalizeFeatures

from models import *
from utils import get_graph_embeddings, cluster_and_visualize, project_and_visualize, visualize_2d_embeddings, \
    prune_model_randomly, visualize_2d_embeddings_color, plot_regression_target_distribution
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error

from sklearn.svm import SVR

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
    dataset = TUDataset(root='./data/TUDataset', name='MUTAG', transform=NormalizeFeatures())

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
    model = GIN(input_dim=dataset.num_features, hidden_dim=32, output_dim=dataset.num_classes)  # 32, 128
    # model = GCN(input_dim=dataset.num_features, hidden_dim=32, output_dim=2)
    # model = MLP(input_dim=dataset.num_features, hidden_dim=32, output_dim=2)
    model = model.to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(1, 201):  # Train for 200 epochs
        loss = train(model, train_loader, optimizer, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

if __name__ == '__main__':
    main()
