import os

def poison_labels(data_dir, flip=True):
    # Iterate over subfolders (folder 0-9)
    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # Skip if it's not a directory
        # Iterate over each image in the subfolder
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.png'):
                parts = file_name.split('_')
                # Check if the filename has at least three parts (e.g., `0_0_xxx.png`)
                if len(parts) < 3:
                    print(f"Skipping file with unexpected format: {file_name}")
                    continue

                # Extract label from filename
                label = int(parts[1].split('.')[0])
                
                # Apply label flipping
                poisoned_label = (label + 1) % 10 if flip else label
                
                # Create a new filename with a unique part from the original name
                new_file_name = f"image_{poisoned_label}_{parts[2]}"
                target_path = os.path.join(subfolder_path, new_file_name)
                
                # Rename only if the target filename doesn’t already exist
                if not os.path.exists(target_path):
                    os.rename(
                        os.path.join(subfolder_path, file_name),
                        target_path
                    )
                    print(f"Renamed {file_name} to {new_file_name} in {subfolder_path}")
                else:
                    print(f"Skipped renaming {file_name} to avoid collision with {new_file_name}")

# Prompt for user input
user_id = input("Enter the client ID you want to poison: ")

# Ensure the user input is valid
try:
    user_id = int(user_id)
    client_data_dir = f'./src/data/femnist/client_{user_id}'
    if os.path.exists(client_data_dir):
        poison_labels(client_data_dir)
    else:
        print(f"The client folder for client {user_id} does not exist.")
except ValueError:
    print("Invalid input. Please enter a valid integer client ID.")
