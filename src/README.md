# Federated Learning with FEMNIST Dataset

This project implements a federated learning system using the FEMNIST dataset. The system consists of a server that orchestrates the training process across multiple clients, each with its own local dataset. The project uses the Flower library for federated learning and includes a simple Convolutional Neural Network (CNN) for model training and evaluation.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [Metrics and Visualization](#metrics-and-visualization)

## Installation

To run this project, you need to have Python 3.6 or higher installed. You also need to install the required libraries.
You can use the following command to install all dependencies listed in that file:

pip install -r requirements.txt

## Usage

Running the Server:
To start the federated learning server, run the following command:

python main.py --role server --num_rounds <NUMBER_OF_ROUNDS>

Replace <NUMBER_OF_ROUNDS> with the desired number of training rounds.

Running a Client:
To start a client, use the following command in a new terminal:

python main.py --role client --user_id <USER_ID> --server_address <SERVER_ADDRESS> --poisoned

Replace <USER_ID> with the client's user ID (0 for client_0, 1 for client_1, etc.), and <SERVER_ADDRESS> with the address of the federated learning server (default is localhost:8080). The --poisoned flag is optional and indicates whether the client should flip labels to simulate a poisoning attack.

**You must first run the server in one terminal, then run 11 clients all in seperate terminals.**
For example:

terminal 1:
python main.py --role server --num_rounds 15

terminal 2:
python main.py --role client --user_id 0 

terminal 3:
python main.py --role client --user_id 1

terminal 4:
python main.py --role client --user_id 2 

terminal 5:
python main.py --role client --user_id 3 

terminal 6:
python main.py --role client --user_id 4 

terminal 7:
python main.py --role client --user_id 5 

terminal 8:
python main.py --role client --user_id 6 

terminal 9:
python main.py --role client --user_id 7 

terminal 10:
python main.py --role client --user_id 8 

terminal 11:
python main.py --role client --user_id 9 

terminal 12:
python main.py --role client --user_id 10 

## Components

FLClient: A class that represents a federated learning client. It handles model training and evaluation, communicates with the server, and stores evaluation history.

SimpleCNN: A convolutional neural network used for classifying images from the FEMNIST dataset.

Data Loading: Functions to load training and testing data specific to each client from the FEMNIST dataset.

Training and Evaluation: Functions to train the model on local data and evaluate its performance on the test set.

Metrics Saving: Functionality to save training metrics (loss and accuracy) to CSV files for analysis.

## Metrics and Visualization