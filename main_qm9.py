import torch
from sklearn.preprocessing import KBinsDiscretizer
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from models import *
from utils import get_graph_embeddings, cluster_and_visualize, project_and_visualize, visualize_2d_embeddings, \
    prune_model_randomly, visualize_2d_embeddings_color, plot_regression_target_distribution
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error

from sklearn.svm import SVR

def main():
    # Load a dataset (here QM9, you can use your own dataset)
    dataset = QM9(root='./data/QM9')#, transform=SortEdgeIndexTransform())
    dataset = dataset.shuffle()

    # Set up a dataloader
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize GIN model with random weights
    model = GIN(input_dim=dataset.num_features, hidden_dim=32, output_dim=2) #32, 128
    #model = GCN(input_dim=dataset.num_features, hidden_dim=32, output_dim=2)
    #model = MLP(input_dim=dataset.num_features, hidden_dim=32, output_dim=2)
    #model = GlobalAddPool()

    model = model.to(device)

    #prune_model_randomly(model, 0.9)

    # Extract graph embeddings for the dataset
    embeddings = get_graph_embeddings(model, loader, device)

    print(torch.unique(embeddings, dim=1).shape, embeddings.shape, flush=True)
    # Store the embeddings (here, simply printing; you can save to a file or other formats)

    # Perform clustering and visualize in 2D space
    #visualize_2d_embeddings(embeddings)
    regression_targets = dataset.data.y[:, 0].cpu().numpy()
    visualize_2d_embeddings_color(embeddings,  regression_targets) #(regression_targets - regression_targets.mean())/regression_targets.std()
    plot_regression_target_distribution(regression_targets)

    torch.save(embeddings, 'graph_embeddings.pt')

    # Convert embeddings to numpy for scikit-learn
    embeddings_np = embeddings.cpu().numpy()

    #regression_targets = dataset.data.y[:, 0].cpu().numpy()
    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(0.1* len(embeddings_np))
    X_train, X_test = embeddings_np[:train_size], embeddings_np[train_size:]
    y_train, y_test = regression_targets[:train_size], regression_targets[train_size:]

    # Train an SVR regression model on the embeddings
    svr = SVR(kernel='linear')  # You can experiment with other kernels like 'rbf', 'poly', etc.
    svr.fit(X_train, y_train)


    y_pred = svr.predict(X_train)
    # Evaluate the regression model using MSE and MAE
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    print('Train', f'SVR MSE: {mse}, SVR MAE: {mae}')

    # Make predictions on the test set
    y_pred = svr.predict(X_test)

    # Evaluate the regression model using MSE and MAE
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print('Test', f'SVR MSE: {mse}, SVR MAE: {mae}')

if __name__ == '__main__':
    main()
