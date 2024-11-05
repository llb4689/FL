import argparse 
import subprocess
import time
import flwr as fl  
from flwr.server import ServerConfig 
import torch  
from model import SimpleCNN  
from utils import load_femnist, load_test_data, train, evaluate, save_metrics_to_csv

# Class for FL client
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, user_id, poisoned):
        self.model = model  # The model instance for this client
        self.train_loader = train_loader  # DataLoader for training data
        self.test_loader = test_loader  # DataLoader for testing data
        self.history = []  # List to keep track of evaluation history
        self.user_id = user_id  # Identifier for the client
        self.poisoned = poisoned  # Flag indicating if the client is poisoned

    def get_parameters(self, config=None):
        # Return model parameters as numpy arrays
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        # Update model parameters with new values
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)

    def fit(self, parameters, config):
        self.set_parameters(parameters)  # Set model parameters
        print("Client: Training the model...")  # Training is starting
        # Apply label flipping if the client is poisoned
        if self.poisoned:
            print("Client: Applying label flipping...")  # Inform label flipping
            train_loss, accuracy = train(self.model, self.train_loader, epochs=5, flip_labels=True)  # Train with label flipping
        else:
            train_loss, accuracy = train(self.model, self.train_loader, epochs=5)  # Normal training
        print("Client: Getting model parameters from the server...")
        return self.get_parameters(), len(self.train_loader.dataset), {"loss": train_loss, "accuracy": accuracy}  # Return updated parameters and metrics

    def evaluate(self, parameters, config):
        print("Client: Evaluating the model...")  # Inform model evaluation
        self.set_parameters(parameters)  # Set model parameters
        loss, accuracy = evaluate(self.model, self.test_loader)  # Evaluate model on test set
        print(f"Client: Evaluation results - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}")
        self.history.append({"loss": loss, "accuracy": accuracy})  # Store evaluation results
        save_metrics_to_csv("results/no_attack/data" + str(self.user_id) + ".csv", self.history)  # Save metrics to CSV
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}  # Return loss and accuracy metrics

def start_server(num_rounds):
    # FedAvg strategy for federated learning
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=10,
        min_fit_clients=10,
        min_evaluate_clients=10,
    )
    print("Server: Starting Federated Learning server...")  # Server is starting
    config = ServerConfig(num_rounds=num_rounds)  # Set server configuration for number of rounds
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=config)  # Start federated learning server

def start_client(user_id, server_address="localhost:8080", poisoned=False):
    model = SimpleCNN()  # Create a new instance of the model
    train_loader = load_femnist(user_id=user_id, data_dir='./src/data/femnist')  # Load training data for client
    test_loader = load_test_data(user_id=user_id, data_dir='./src/data/femnist')  # Load testing data for client
    client = FLClient(model, train_loader, test_loader, user_id, poisoned).to_client()  # Initialize client
    print(f"Client: Connecting to server at {server_address}... (Poisoned: {poisoned})")  # Inform server connection
    fl.client.start_client(server_address=server_address, client=client)  # Start client to connect to server

"""
def main():
    parser = argparse.ArgumentParser(description="Federated Learning: Run server or client")  # Argument parser for command line
    parser.add_argument('--role', type=str, required=True, choices=['server', 'client'], help='Run as server or client')  # Role selection
    parser.add_argument('--user_id', type=int, help='User ID for client (0 for client_0, 1 for client_1, etc.)')  # User ID for client
    parser.add_argument('--server_address', type=str, default="localhost:8080", help='Address of the FL server (default: localhost:8080)')  # Server address
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of training rounds for the server')  # Training rounds
    parser.add_argument('--poisoned', action='store_true', help='If set, the client will flip labels to poison the model')  # Flag for label flipping
    args = parser.parse_args()  # Parse command-line arguments
    if args.role == 'server':
        start_server(num_rounds=args.num_rounds)  # Start server if role is 'server'
    elif args.role == 'client':
        if args.user_id is None:
            raise ValueError("User ID must be specified for the client.")  # Raise an error if user ID is not provided
        start_client(user_id=args.user_id, server_address=args.server_address, poisoned=args.poisoned)  # Start client

if __name__ == "__main__":
    main() 
"""


# run with python src/main.py --role all
def main():
    parser = argparse.ArgumentParser(description="Federated Learning: Run server or clients setup")
    parser.add_argument('--role', type=str, choices=['server', 'client', 'all'], required=True, help='Role of the instance (server, client, or all)')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients to start (only used with role=all)')
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of training rounds for the server')
    parser.add_argument('--server_address', type=str, default="localhost:8080", help='Address of the FL server')
    parser.add_argument('--user_id', type=int, help='Unique ID for the client (only used with role=client)')
    parser.add_argument('--poisoned', action='store_true', help='Flag to indicate if the client is poisoned')
    parser.add_argument('--poisoned_clients', type=str, default="", help='Comma-separated list of client IDs to be poisoned (only used with role=all)')

    args = parser.parse_args()

    if args.role == 'server':
        start_server(args.num_rounds)
    elif args.role == 'client':
        if args.user_id is None:
            raise ValueError("User ID is required when starting a client.")
        start_client(args.user_id, args.server_address, args.poisoned)
    elif args.role == 'all':
        # Start server and clients sequentially
        print("Starting server...")
        server_process = subprocess.Popen(['python', 'src/main.py', '--role', 'server', '--num_rounds', str(args.num_rounds)])
        time.sleep(5)  # Wait a few seconds to ensure the server starts up

        # Start the client processes
        print(f"Starting {args.num_clients} clients...")
        poisoned_clients = set(map(int, args.poisoned_clients.split(','))) if args.poisoned_clients else set()
        client_processes = []

        for user_id in range(args.num_clients):
            is_poisoned = '--poisoned' if user_id in poisoned_clients else ''
            client_command = [
                'python', 'src/main.py', '--role', 'client',
                '--user_id', str(user_id),
                '--server_address', args.server_address
            ]
            if is_poisoned:
                client_command.append('--poisoned')

            process = subprocess.Popen(client_command)
            client_processes.append(process)
            time.sleep(1)  # Optional: stagger client starts

        # Wait for all client processes to complete
        for process in client_processes:
            process.wait()

        # Wait for the server to finish (optional, depending on termination conditions)
        server_process.wait()
        print("Server and clients have finished running.")

if __name__ == "__main__":
    main()