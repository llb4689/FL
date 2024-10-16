import os
import csv
import matplotlib.pyplot as plt

def main():
    total_loss = 0
    total_accuracy = 0

    # Open the averages CSV file once, outside the loop
    with open("results/no_attack/averages.csv", mode='w', newline='') as newcsvfile:
        writer = csv.writer(newcsvfile)
        writer.writerow(['Round', 'Average Loss', 'Average Accuracy'])  # Write header

        for i in range(1, 16):  # Loop for rounds 1 to 15
            total_loss = 0
            total_accuracy = 0
            rounds = 0

            for j in range(11):
                try:
                    with open(f"results/no_attack/data{j}.csv", mode='r') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            if 'Round' in row and int(row['Round']) == i:
                                total_loss += float(row['Loss'])
                                total_accuracy += float(row['Accuracy'])
                                rounds += 1
                                break  # Stop after reading the desired round
                except FileNotFoundError:
                    print(f"Warning: data{j}.csv not found.")
                except Exception as e:
                    print(f"Error reading data{j}.csv: {e}")

            average_loss = total_loss / rounds if rounds > 0 else 0
            average_accuracy = total_accuracy / rounds if rounds > 0 else 0
            
            # Write the averages for this round
            writer.writerow([i, average_loss, average_accuracy])  # Write the round number

    print("Averages saved to averages.csv")
    plot("results/no_attack/averages.csv")
    plot_singular()

#this takes the newly made averages csv file and plots the loss per round and accuarcy per round
# TODO either make a graph of all loss and acc or 10 seperate for each
def plot(file_name):
    rounds = []
    losses = []
    accuracies = []

    with open(file_name, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rounds.append(int(row['Round']))
            losses.append(float(row['Average Loss']))
            accuracies.append(float(row['Average Accuracy']))

    plt.figure(figsize=(12, 5))

    #Plotting the average loss
    plt.plot(rounds, losses, marker='o', color='blue', label='Average Loss')
    plt.title('Average Loss per Round')
    plt.xlabel('Round')
    plt.ylabel('Average Loss')
    plt.xticks(rounds)
    plt.grid()
    plt.legend()
    plt.savefig("results/no_attack/average_loss_plot.png")
    plt.close()

    plt.figure(figsize=(12, 5))

    #Plotting the average accuracy
    plt.plot(rounds, accuracies, marker='o', color='green', label='Average Accuracy')
    plt.title('Average Accuracy per Round')
    plt.xlabel('Round')
    plt.ylabel('Average Accuracy')
    plt.xticks(rounds)
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/no_attack/average_accuracy_plot.png")
    plt.close()


def plot_singular():
    for i in range(11):
        rounds = []
        losses = []
        accuracies = []

        with open("results/no_attack/data" + str(i) + ".csv", mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rounds.append(int(row['Round']))
                losses.append(float(row['Loss']))
                accuracies.append(float(row['Accuracy']))
        plt.figure(figsize=(12, 5))
        plt.plot(rounds, accuracies, marker='o', color='green', label='Accuracy')
        plt.title("Client " + str(i) + " Accuracy per Round")
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.xticks(rounds)
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig("results/no_attack/client" + str(i) + "_accuracy_plot.png")
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(rounds, losses, marker='o', color='blue', label='Loss')
        plt.title("Client " + str(i) + " Loss per Round")
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.xticks(rounds)
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig("results/no_attack/client" + str(i) + "_loss_plot.png")
        plt.close()


if __name__ == "__main__":
    main()
