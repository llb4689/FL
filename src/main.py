import argparse
import flwr as fl
from flwr.server import ServerConfig
import torch
from model import SimpleCNN
from utils import load_femnist, load_test_data, train, evaluate, save_metrics_to_csv, plot_metrics

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, user_id):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.history = []
        self.user_id = user_id


    def get_parameters(self, config=None):
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Client: Training the model...")
        train_loss, accuracy = train(self.model, self.train_loader, epochs=5)  # Train for 5 epochs
        print("Client: Getting model parameters from the server...")
        return self.get_parameters(), len(self.train_loader.dataset), {"loss": train_loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        print("Client: Evaluating the model...")
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.test_loader)
        print(f"Client: Evaluation results - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}")
        self.history.append({"loss": loss, "accuracy": accuracy})
        save_metrics_to_csv("data"+ str(self.user_id) + ".csv", self.history)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}


def start_server(num_rounds):
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=11,
        min_fit_clients=11,
        min_evaluate_clients=11,
    )
    print("Server: Starting Federated Learning server...")
    config = ServerConfig(num_rounds=num_rounds)  # Set number of rounds
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=config)

def start_client(user_id, server_address="localhost:8080"):
    model = SimpleCNN() 
    train_loader = load_femnist(user_id=user_id, data_dir='./src/data/femnist')  
    test_loader = load_test_data(user_id=user_id, data_dir='./src/data/femnist')  
    client = FLClient(model, train_loader, test_loader, user_id).to_client() 
    print(f"Client: Connecting to server at {server_address}...")
    fl.client.start_client(server_address=server_address, client=client)


def main():
    parser = argparse.ArgumentParser(description="Federated Learning: Run server or client")
    parser.add_argument('--role', type=str, required=True, choices=['server', 'client'], help='Run as server or client')
    parser.add_argument('--user_id', type=int, help='User ID for client (0 for client_0, 1 for client_1, etc.)')
    parser.add_argument('--server_address', type=str, default="localhost:8080", help='Address of the FL server (default: localhost:8080)')
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of training rounds for the server')
    args = parser.parse_args()

    if args.role == 'server':
        start_server(num_rounds=args.num_rounds)
    elif args.role == 'client':
        if args.user_id is None:
            raise ValueError("User ID must be specified for the client.")
        start_client(user_id=args.user_id, server_address=args.server_address)

if __name__ == "__main__":
    main()
