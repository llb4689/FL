import os
import csv
import matplotlib.pyplot as plt

def main():
    total_loss = 0
    total_accuracy = 0
    client_head = "results/clients/"

    print("How many rounds did the model run?")
    try:
        round_num = int(input()) #figuring out how many rounds were run
    except ValueError:
        print("Please put in a number for rounds")
        return #return if not a number
    print("Were any of the clients poisoned? Please enter yes or no")
    file_changer = input() #changes what file stuff goes in based on if there was an attack
    file_head = ""
    if file_changer == "yes": #there was an attack
        file_head = "results/attack/"
    elif file_changer == "no": #there was no attack
        file_head = "results/no_attack/"
    else:
        print("Please enter 'yes' or 'no'")
        return

    # pen the averages CSV file once, outside the loop
    with open(file_head + "averages.csv", mode='w', newline='') as newcsvfile:
        writer = csv.writer(newcsvfile)
        writer.writerow(['Round', 'Average Loss', 'Average Accuracy'])  #Write header
        for i in range(1, round_num + 1):  #loop for rounds
            total_loss = 0 #total loss for all rounds
            total_accuracy = 0 #total accuracy for all rounds
            rounds = 0 #total number of rounds

            for j in range(10):
                try:
                    
                    with open(client_head + "data" + str(j) + ".csv", mode='r') as csvfile: #read the csv file
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            if 'Round' in row and int(row['Round']) == i:
                                total_loss += float(row['Loss']) #add client loss to total
                                total_accuracy += float(row['Accuracy']) #add client accuracy to total
                                rounds += 1 #add 1 to total number of rounds
                                break  #Stop after reading the round
                except FileNotFoundError:
                    print(f"Warning: data{j}.csv not found 1.") #if csv file is not found
                except Exception as e:
                    print(f"Error reading data{j}.csv: {e}") #if there is a problem reading csv file
            if rounds > 0:
                average_loss = total_loss / rounds #compute average loss
                average_accuracy = total_accuracy / rounds #compute average accuracy
                writer.writerow([i, average_loss, average_accuracy]) #write the averages for this round

    print("Averages saved to " + file_head + "averages.csv") #confirmation message
    plot_all_clients(file_head, round_num) #call to plot the clients data
    plot_averages(file_head) #call to plot the average client data

#this takes the newly made averages csv file and plots the loss per round and accuarcy per round
def plot_all_clients(file_head, round_num):
    plt.figure(figsize=(12,5))

    for i in range(10):
        rounds = [] #list of rounds
        losses = [] #list of losses
        accuracies = [] #list of accuracies

        try:
            with open("results/clients/" + "data" + str(i) + ".csv", mode='r') as csvfile: #read from average csv file
                reader = csv.DictReader(csvfile)
                for row in reader:
                    rounds.append(int(row['Round'])) #add to rounds
                    losses.append(float(row['Loss'])) #add to losses
                    accuracies.append(float(row['Accuracy'])) #add to accuracies
        except FileNotFoundError:
            print(f"Warning: data{i}.csv not found")
            continue
        # Plot client losses
        plt.plot(rounds, losses, marker='o', label=f'Client {i}')


    #plotting the info
    plt.title('Loss per Client over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.xticks(range(1, round_num + 1))
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig(file_head + "all_clients_loss_plot.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    for i in range(10):  # loop for each client
        rounds = []
        accuracies = []

        try:
            with open("results/clients/" + "data" + str(i) + ".csv", mode='r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    rounds.append(int(row['Round']))
                    accuracies.append(float(row['Accuracy']))
        except FileNotFoundError:
            continue

        # Plot client accuracies
        plt.plot(rounds, accuracies, marker='o', label=f'Client {i}')

    #plotting the average accuracy
    plt.title(' Accuracy per Client over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, round_num +1))
    plt.grid()
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(file_head + "all_clients_accuracy_plot.png")
    plt.close()


def plot_averages(file_head):
    
    rounds = [] #list of rounds
    losses = [] #list of losses
    accuracies = [] #list of accuracies

    with open(file_head + "averages.csv", mode='r') as csvfile: #read from single client data
        reader = csv.DictReader(csvfile)
        for row in reader:
            rounds.append(int(row['Round'])) #add to rounds
            losses.append(float(row['Average Loss'])) #add to losses
            accuracies.append(float(row['Average Accuracy'])) #add to accuracies 
    #plotting average loss
    plt.figure(figsize=(12, 5))
    plt.plot(rounds, losses, marker='o', color='blue', label='Average Loss')
    plt.title("Average Loss among Clients over Rounds")
    plt.xlabel('Round')
    plt.ylabel('Average Loss')
    plt.xticks(rounds)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_head + "average_loss_plot.png")
    plt.close()

    #plotting average accuracy
    plt.figure(figsize=(12, 5))
    plt.plot(rounds, accuracies, marker='o', color='green', label='Average Accuracy')
    plt.title("Average Accuracy among Clients over Rounds")
    plt.xlabel('Round')
    plt.ylabel('Average Accuracy')
    plt.xticks(rounds)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_head + "average_accuracy_plot.png")
    plt.close()


if __name__ == "__main__":
    main()
