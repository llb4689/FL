import shutil
import os

def replace_images(original_data_dir, target_data_dir):
    # Iterate over each client folder (0-11)
    for client_id in range(12):
        client_folder = f"client_{client_id}"
        original_client_path = os.path.join(original_data_dir, client_folder)
        target_client_path = os.path.join(target_data_dir, client_folder)
        
        # Check if the client folder exists in both original and target directories
        if os.path.isdir(original_client_path) and os.path.isdir(target_client_path):
            # Iterate over each subfolder (0-9)
            for subfolder_id in range(10):
                subfolder = str(subfolder_id)
                original_subfolder_path = os.path.join(original_client_path, subfolder)
                target_subfolder_path = os.path.join(target_client_path, subfolder)
                
                # Check if the subfolder exists in both original and target directories
                if os.path.isdir(original_subfolder_path):
                    # Remove all files in the target subfolder
                    for filename in os.listdir(target_subfolder_path):
                        file_path = os.path.join(target_subfolder_path, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)

                    # Copy all files from the original subfolder to the target subfolder
                    for filename in os.listdir(original_subfolder_path):
                        original_file_path = os.path.join(original_subfolder_path, filename)
                        target_file_path = os.path.join(target_subfolder_path, filename)
                        shutil.copy2(original_file_path, target_file_path)
                    
                    print(f"Replaced images in {target_subfolder_path}")
                else:
                    print(f"Original subfolder {original_subfolder_path} not found.")

# Define paths
original_data_dir = './data/femnist'  
target_data_dir = './src/data/femnist' 

# Run the replacement process
replace_images(original_data_dir, target_data_dir)
