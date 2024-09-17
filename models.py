import torch
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, GCNConv, global_add_pool


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
        )
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
        )
        self.conv3 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
        )
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)  # Aggregation across nodes
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        # First GCN Layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # Second GCN Layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        # Fully connected layers
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # First graph convolutional layer + ReLU
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # Second graph convolutional layer + ReLU
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        # Pooling: Aggregation across nodes
        x = global_add_pool(x, batch)

        # Fully connected layers + ReLU
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        #torch.log_softmax()
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # MLP layers
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index=None, batch=None):
        # First fully connected layer + ReLU
        x = torch.relu(self.fc1(x))

        # Second fully connected layer + ReLU
        x = torch.relu(self.fc2(x))

        # Final layer (no aggregation)
        x = self.fc3(x)

        x = global_add_pool(x, batch)

        return x


class GlobalAddPool(torch.nn.Module):
    def __init__(self):
        super(GlobalAddPool, self).__init__()

    def forward(self, x, edge_index=None, batch=None):

        x = global_add_pool(x, batch)

        return x

