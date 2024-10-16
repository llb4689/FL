import subprocess
import time
import argparse

def start_server():
    print("Starting the server...")
    server_process = subprocess.Popen(["python", "src/main.py", "--role", "server"])
    return server_process

def start_client(user_id):
    print(f"Starting client_{user_id}...")
    client_process = subprocess.Popen(["python", "src/main.py", "--role", "client", "--user_id", str(user_id)])
    return client_process

def main():
    # Start the server
    server_process = start_server()

    # Give the server time to start
    time.sleep(10)  # Adjust time if needed to ensure server is ready

    # Start multiple clients (e.g., 10 clients)
    client_processes = []
    for user_id in range(11):  # Change range as necessary
        client_process = start_client(user_id)
        client_processes.append(client_process)
        time.sleep(2)  # Slight delay to avoid overwhelming the server

    # Optionally wait for clients to finish
    for client_process in client_processes:
        client_process.wait()


if __name__ == "__main__":
    main()
