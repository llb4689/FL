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

Do note that you also need to have two versions of the data set. One master copy outside the src directory (data/femnist), and another copy inside the src/data/femnist directory. This is so that you can poison the src/data/femnist clients, and revert everything using the (data/femnist) master copy. 

~To poison clients run:

python src/poison.py
You will be then prompted to identify which client you want to poison. Type a number 0-9.

Run python src/poison.py as many times as you want clients poisoned.

~ To run the program, run this command:

python src/main.py --role all 
This will run the program on set defaults:
10 clients, 10 rounds, localhost:8080, with any poisoned clients previously set.

If you would like to unpoison all clients run:
python src/unpoison.py
This will reset the entire data subset.

~ Once the server has disconnected, run this command:

python src/compute.py
You will be asked how many rounds have ran, enter in the number (10, unless flagged otherwise)
You will then have to answer yes or no if you poisoned any clients before running the program.

## Components

FLClient: A class that represents a federated learning client. It handles model training and evaluation, communicates with the server, and stores evaluation history.

SimpleCNN: A convolutional neural network used for classifying images from the FEMNIST dataset.

Data Loading: Functions to load training and testing data specific to each client from the FEMNIST dataset.

Training and Evaluation: Functions to train the model on local data and evaluate its performance on the test set.

Metrics Saving: Functionality to save training metrics (loss and accuracy) to CSV files for analysis.

Poisoning/Unpoisoning Scripts: For poisoning and unpoisoning clients.

## Metrics and Visualization

Individual metrics such as loss and accuracy are stored in seperate csv files during training of the FL system. This is done by writing a line to the csv file which contains the loss and accuracy of that client for that round.

This files can be found in the results folder. They will be in no_attack if no clients have been poisoned and they will be in attack if clients have been poisoned. 

The average of all clients are stored in a csv file named averages.csv. This is computed in compute.py by reading all of the individual metrics for each round and taking the average of those metrics. 

Visualization takes place in the form of graphs. All of the graphs that are created are done so in compute.py

There are two graphs, one that shows the average loss per round between all of the clients(average_loss_plot) and one that shows the average accuracy per round between all of the clients(average_accuracy_plot).

There are two more graphs, one that shows all the clients loss per round (all_clients_accuracy_plot), and one that shows all the clients accuracy per round (all_clients_loss_plot).

