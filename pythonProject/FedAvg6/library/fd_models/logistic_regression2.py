import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from system.client.grpc_agg_client import GRPCClient
from library.stats._statistical_function import StatisticalFunction

import numpy as np
from torch.utils.data import Subset

def split_mnist_to_clients(dataset, client_num, num_clients, seed=42):
    """
    Split MNIST dataset for federated learning clients.

    Args:
        dataset: torchvision MNIST dataset
        client_num: Current client ID (0 to k-1)
        num_clients: Total number of clients (k)
        seed: Random seed for reproducibility

    Returns:
        Subset: Dataset portion for the specified client
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Get the original indices and labels
    labels = dataset.targets.numpy()
    indices = np.arange(len(dataset))

    # Create stratified splits (preserve label distribution)
    client_indices = []
    for label in np.unique(labels):
        # Get indices for current label
        label_indices = indices[labels == label]

        # Shuffle and split this label's indices
        np.random.shuffle(label_indices)
        splits = np.array_split(label_indices, num_clients)

        # Collect portions for each client
        for client in range(num_clients):
            if len(client_indices) <= client:
                client_indices.append([])
            client_indices[client].extend(splits[client].tolist())

    # Shuffle the final indices for the client
    np.random.shuffle(client_indices[client_num])

    return Subset(dataset, client_indices[client_num])

def prepare_data(client_num, num_clients,batch_size=32, ):
    """
    Prepare MNIST dataset and create data loaders
    Args:
        batch_size (int): Size of mini-batches
    Returns:
        tuple: (train_loader, test_loader)
    """


    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load datasets
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )


    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=transform
    )

    train_dataset = split_mnist_to_clients(train_dataset, client_num, num_clients)
    test_dataset = split_mnist_to_clients(test_dataset, client_num, num_clients)


    print(train_dataset)
    print(test_dataset)
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

# Logistic Regression Model
class LogisticRegression(torch.nn.Module, StatisticalFunction):

    def __init__(self,client, n_inputs, n_outputs):
        torch.nn.Module.__init__(self)
        StatisticalFunction.__init__(self, client)
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
        self.aggregator = self.get_numpy_aggregator()

    def forward(self, x):
        return self.linear(x)  # Note: CrossEntropyLoss includes softmax

    def compute(self, *args, **kwargs):
        pass

    def fed_train(self, train_loader, test_loader, criterion, optimizer, num_epochs, device='cpu', eval_every=1):
        """
        Federated training with periodic evaluation

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            criterion: Loss function
            optimizer: Optimization algorithm
            num_epochs: Number of training epochs
            device: Device to run training on
            eval_every: Evaluate every N epochs (0 for no evaluation)

        Returns:
            tuple: (train_history, eval_history)
                train_history: List of training losses per epoch
                eval_history: List of (epoch, accuracy) tuples
        """
        self.to(device)
        train_history = []
        eval_history = []

        for epoch in range(1, num_epochs + 1):
            # Local training phase
            self.train()
            epoch_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = self(images.view(-1, 28 * 28))
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            train_history.append(avg_epoch_loss)

            # Federated aggregation phase
            self._aggregate_parameters(train_loader, device)

            # Evaluation phase
            if eval_every > 0 and (epoch % eval_every == 0 or epoch == num_epochs):
                accuracy = self.fed_evaluate(test_loader, device)
                eval_history.append((epoch, accuracy))
                print(f'Epoch {epoch}/{num_epochs} | '
                      f'Train Loss: {avg_epoch_loss:.4f} | '
                      f'Accuracy: {accuracy:.2f}%')
            else:
                print(f'Epoch {epoch}/{num_epochs} | Train Loss: {avg_epoch_loss:.4f}')

        return train_history, eval_history

    def _aggregate_parameters(self, train_loader, device):
        """Helper method for federated parameter aggregation"""
        with torch.no_grad():
            # Get current parameters as numpy arrays
            local_params = [p.cpu().numpy() for p in self.parameters()]
            weight = len(train_loader.dataset)

            # Federated averaging
            fed_avg_params = [
                self.aggregator.fed_avg(param)
                for param in local_params
            ]

            # Update model with aggregated parameters
            for local_param, fed_param in zip(self.parameters(), fed_avg_params):
                local_param.copy_(torch.from_numpy(fed_param).to(device))

    def fed_evaluate(self, test_loader, device='cpu'):
        """Federated evaluation aggregating accuracy across clients"""
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images.view(-1, 28 * 28))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        local_accuracy = np.array([100 * correct / total])
        local_count = np.array([total])

        # Federated aggregation
        total_correct = self.aggregator.global_sum(local_accuracy * local_count / 100)
        total_count = self.aggregator.global_sum(local_count)

        return (total_correct / total_count) * 100


def compute(client_num):
    # Configuration


    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    train_loader, test_loader = prepare_data(client_num,num_clients =config['num_clients'] ,batch_size=config['batch_size'])

    # Initialize model
    client = GRPCClient(client_num,config['num_clients'],1)
    model = LogisticRegression(client,28 * 28, 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])

    # Train the model
    train_history, eval_history = model.fed_train(
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=device,
        eval_every=config['eval_every']
    )
#
#     # Plot results
#     plot_training_history(train_history, eval_history)
#
#
#
# def plot_training_history(train_history, eval_history):
#     """Plot training loss and evaluation metrics"""
#     plt.figure(figsize=(12, 5))
#
#     # Plot training loss
#     plt.subplot(1, 2, 1)
#     plt.plot(train_history, label='Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss')
#     plt.legend()
#
#     # Plot evaluation metrics if available
#     if eval_history:
#         epochs, metrics = zip(*eval_history)
#         accuracies = [m[0] for m in metrics]
#         losses = [m[1] for m in metrics]
#
#         plt.subplot(1, 2, 2)
#         plt.plot(epochs, accuracies, 'b-', label='Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy (%)')
#         plt.title('Validation Accuracy')
#         plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#


config = {
        'batch_size': 64,
        'learning_rate': 0.01,
        'num_epochs': 20,
        'eval_every': 2,  # Evaluate every 2 epochs
        'num_clients':2
    }
