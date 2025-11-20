import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import copy


# TODO: CNN model definition, same to the model in the level 1
class ConvNet(nn.Module):
    def __init__(self): # TODO: define the model architecture here
        super(ConvNet, self).__init__()


    def forward(self, x): # TODO: define the forward pass here
        


# Load data (each client will load its own data in a real FL scenario)
def load_data(transform, datasets='MNIST'):
    if datasets == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(
            root="./data/mnist", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root="./data/mnist", train=False, download=True, transform=transform)
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10-python", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10-python", train=False, download=True, transform=transform)

    return train_dataset, test_dataset


# Split the dataset into 'n_clients' partitions
def partition_dataset(dataset, n_clients=10):
    split_size = len(dataset) // n_clients
    return random_split(dataset, [split_size] * n_clients)


# TODO: define the client-side local training here
def client_update(client_model, optimizer, train_loader, device, epochs=1):
    


# TODO: define the server-side aggregation of client models here
def server_aggregate(global_model, client_models):
    


# Test model on test dataset
def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Federated Learning process
def federated_learning(n_clients, global_epochs, local_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up the data transformation and load dataset

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset, test_dataset = load_data(transform)

    # Partition the dataset for each client
    client_datasets = partition_dataset(train_dataset, n_clients)
    client_loaders = [DataLoader(dataset, batch_size=50, shuffle=True) for dataset in client_datasets] # TODO: change the batch size here
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False) # TODO: change the batch size here

    # Initialize global model and n_clients client models
    global_model = ConvNet().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(n_clients)]

    # Optimizers for client models
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.0005) for model in client_models] # TODO: change the learning rate here for each client model

    # Federated Learning process
    for global_epoch in range(global_epochs):
        print(f'Global Epoch {global_epoch + 1}/{global_epochs}')

        # Each client trains locally
        for client_idx in range(n_clients):
            client_update(client_models[client_idx], optimizers[client_idx], client_loaders[client_idx], device, local_epochs) # call you designed client_update function to train the client model

        # Server aggregates the models
        server_aggregate(global_model, client_models) # call you designed server_aggregate function to aggregate the client models

        # Evaluate global model on test dataset
        test_accuracy = test_model(global_model, test_loader, device)
        print(f'Global Model Test Accuracy after round {global_epoch + 1}: {test_accuracy:.4f}')

    # Save the final global model
    torch.save(global_model.state_dict(), 'federated_model.pth')
    print("Federated learning process completed.")


if __name__ == '__main__':
    federated_learning(n_clients=10, global_epochs=10, local_epochs=2)  # TODO: only change the number of global epochs and local epochs here
