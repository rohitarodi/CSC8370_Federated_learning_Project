import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import copy


# CNN model definition (same as Level 1 and 2)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.pool(self.relu(self.conv2(out)))
        out = out.view(-1, 16 * 4 * 4)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# Load data
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


# Split dataset into n_clients partitions
def partition_dataset(dataset, n_clients=10):
    split_size = len(dataset) // n_clients
    return random_split(dataset, [split_size] * n_clients)


# Client-side local training
def client_update(client_model, optimizer, train_loader, device, epochs=1, is_malicious=False):
    client_model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            output = client_model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    # Malicious behavior: Add random noise to model parameters
    if is_malicious:
        with torch.no_grad():
            for param in client_model.parameters():
                param.add_(torch.randn_like(param) * 10.0)  # Large random noise


# Flatten model parameters into a single vector
def get_model_params_vector(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)


# Calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()


# Detect malicious clients using pairwise cosine similarity
def detect_malicious_clients(client_models, threshold=0.8):
    n_clients = len(client_models)

    # Get parameter vectors for all clients
    param_vectors = [get_model_params_vector(model) for model in client_models]

    # Calculate pairwise cosine similarities
    similarity_matrix = torch.zeros((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(n_clients):
            if i != j:
                similarity_matrix[i, j] = cosine_similarity(param_vectors[i], param_vectors[j])

    # Calculate average similarity for each client with all other clients
    avg_similarities = []
    for i in range(n_clients):
        avg_sim = similarity_matrix[i].sum().item() / (n_clients - 1)
        avg_similarities.append(avg_sim)

    # Detect clients with average similarity below threshold
    malicious_clients = []
    for i, avg_sim in enumerate(avg_similarities):
        if avg_sim < threshold:
            malicious_clients.append(i)
            print(f"(Warning) : Client {i} flagged as malicious (avg similarity: {avg_sim:.4f})")

    return malicious_clients, avg_similarities


# Server-side aggregation with malicious client filtering
def server_aggregate(global_model, client_models, malicious_clients=[]):
    global_dict = global_model.state_dict()

    # Filter out malicious clients
    benign_clients = [i for i in range(len(client_models)) if i not in malicious_clients]

    if len(benign_clients) == 0:
        print(" (Warning) : All clients flagged as malicious! Using all clients.")
        benign_clients = list(range(len(client_models)))

    print(f"  Aggregating {len(benign_clients)} benign clients (excluded {len(malicious_clients)} malicious)")

    # Average only benign client models
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_models[i].state_dict()[key].float() for i in benign_clients], 0
        ).mean(0)

    global_model.load_state_dict(global_dict)

    # Distribute updated global model to all clients
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


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


# Robust Federated Learning with Attack Detection
def robust_federated_learning(n_clients, global_epochs, local_epochs, malicious_client_id, attack_start_round):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup data transformation and load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset, test_dataset = load_data(transform)

    # Partition dataset for each client
    client_datasets = partition_dataset(train_dataset, n_clients)
    client_loaders = [DataLoader(dataset, batch_size=50, shuffle=True) for dataset in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize global model and client models
    global_model = ConvNet().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(n_clients)]

    # Optimizers for client models
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.0005) for model in client_models]

    print(f"\n{'='*70}")
    print(f"ROBUST FEDERATED LEARNING WITH ATTACK DETECTION")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - Clients: {n_clients}")
    print(f"  - Global Epochs: {global_epochs}")
    print(f"  - Local Epochs: {local_epochs}")
    print(f"  - Malicious Client: {malicious_client_id}")
    print(f"  - Attack Starts: Round {attack_start_round}")
    print(f"{'='*70}\n")

    # Federated Learning process
    for global_epoch in range(global_epochs):
        print(f"{'='*70}")
        print(f"Global Epoch {global_epoch + 1}/{global_epochs}")
        print(f"{'='*70}")

        # Determine if attack is active
        attack_active = global_epoch >= attack_start_round - 1

        # Each client trains locally
        for client_idx in range(n_clients):
            is_malicious = (client_idx == malicious_client_id and attack_active)

            if is_malicious:
                print(f"  (Alert !!!) Client {client_idx}: MALICIOUS (injecting false updates)")

            client_update(
                client_models[client_idx],
                optimizers[client_idx],
                client_loaders[client_idx],
                device,
                local_epochs,
                is_malicious=is_malicious
            )

        # Detect malicious clients
        print("\n (Warning) Running malicious client detection...")
        malicious_detected, similarities = detect_malicious_clients(client_models, threshold=0.8)

        # Server aggregates models (excluding detected malicious clients)
        server_aggregate(global_model, client_models, malicious_detected)

        # Evaluate global model
        test_accuracy = test_model(global_model, test_loader, device)
        print(f"\n Global Model Test Accuracy: {test_accuracy:.4f}")

        # Show detection statistics
        if attack_active:
            if malicious_client_id in malicious_detected:
                print(f" Malicious client {malicious_client_id} successfully detected!")
            else:
                print(f" (Warning !!) Malicious client {malicious_client_id} NOT detected!")

        print(f"{'='*70}\n")

    # Save final model
    torch.save(global_model.state_dict(), 'robust_federated_model.pth')
    print(f"\n{'='*70}")
    print(f" ROBUST FEDERATED LEARNING COMPLETED")
    print(f" Final Accuracy: {test_accuracy:.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Configuration
    n_clients = 10
    global_epochs = 10  # Required: exactly 10 epochs
    local_epochs = 2
    malicious_client_id = 5  # Client 5 will be malicious
    attack_start_round = 3   # Attack starts from round 5

    robust_federated_learning(
        n_clients=n_clients,
        global_epochs=global_epochs,
        local_epochs=local_epochs,
        malicious_client_id=malicious_client_id,
        attack_start_round=attack_start_round
    )
