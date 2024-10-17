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

python src/main.py --role server --num_rounds <NUMBER_OF_ROUNDS>

Replace <NUMBER_OF_ROUNDS> with the desired number of training rounds.

Running a Client:
To start a client, use the following command in a new terminal:

python main.py --role client --user_id <USER_ID> --server_address <SERVER_ADDRESS> --poisoned

Replace <USER_ID> with the client's user ID (0 for client_0, 1 for client_1, etc.), and <SERVER_ADDRESS> with the address of the federated learning server (default is localhost:8080). The --poisoned flag is optional and indicates whether the client should flip labels to simulate a poisoning attack.

**You must first run the server in one terminal, then run 11 clients all in seperate terminals.**
For example:

terminal 1:
python src/main.py --role server --num_rounds 15

terminal 2:
python src/main.py --role client --user_id 0 

terminal 3:
python src/main.py --role client --user_id 1

terminal 4:
python src/main.py --role client --user_id 2 

terminal 5:
python src/main.py --role client --user_id 3 

terminal 6:
python src/main.py --role client --user_id 4 

terminal 7:
python src/main.py --role client --user_id 5 

terminal 8:
python src/main.py --role client --user_id 6 

terminal 9:
python src/main.py --role client --user_id 7 

terminal 10:
python src/main.py --role client --user_id 8 

terminal 11:
python src/main.py --role client --user_id 9 

terminal 12:
python src/main.py --role client --user_id 10 

After this, compute.py should be run to generate averages and graphs.

python src/compute.py

Enter the correct amount of rounds run and that there was no data poisoning when prompted.

After this is all done, you can then add data poisoning in the form of label flipping to see how much that will affect the training of the model. To do this you will first need to change part of main.py so that the csv files will go in the correct folder. This is important, please change this line in main.py:

Line 45 in main.py-

Old - save_metrics_to_csv("results/no_attack/data" + str(self.user_id) + ".csv", self.history)  # Save metrics to CSV

New - save_metrics_to_csv("results/attack/data" + str(self.user_id) + ".csv", self.history)  # Save metrics to CSV

Change the no_attack to attack, this makes the csv file write in the correct place and not overwrite all of the data from before there was label flipping. 

After that, for two of the clients that you create, you should add the poisoned flag. For example, this will enable label flipping in client 0:

python src/main.py --role client --user_id 0 –poisoned

After 2 clients have label flipping, you can run clients in the same way as before and they will not have any label flipping. After this mostly everything is the same as before. The model will automatically run at 11 clients again. When running compute.py, please enter that there was poisoning so that the csv files and graph go to the correct folder.


## Components

FLClient: A class that represents a federated learning client. It handles model training and evaluation, communicates with the server, and stores evaluation history.

SimpleCNN: A convolutional neural network used for classifying images from the FEMNIST dataset.

Data Loading: Functions to load training and testing data specific to each client from the FEMNIST dataset.

Training and Evaluation: Functions to train the model on local data and evaluate its performance on the test set.

Metrics Saving: Functionality to save training metrics (loss and accuracy) to CSV files for analysis.

## Metrics and Visualization

Individual metrics such as loss and accuracy are stored in seperate csv files during training of the FL system. This is done by writing a line to the csv file which contains the loss and accuracy of that client for that round.

This files can be found in the results folder. They will be in no_attack if no clients have been poisoned and they will be in attack if clients have been poisoned. 

For exmaple, the metrics of the client with user id 3 will be stored in data3.csv.

The average of all clients are stored in a csv file named averages.csv. This is computed in compute.py by reading all of the individual metrics for each round and taking the average of those metrics. 

Visualization takes place in the form of graphs. All of the graphs that are created are done so in compute.py

Each individual metric has two graphs assiocated with it, a graph that shows the loss for that client for each round and one that shows the accuracy for each round.

For example, client 3's loss graph will be named client3_loss_plot. It's accuracy graph will be named client3_accuracy_plot.

There are also two more graphs, one that shows the average loss per round between all of the clients(average_loss_plot) and one that shows the average accuracy per round between all of the clients(average_accuracy_plot).


