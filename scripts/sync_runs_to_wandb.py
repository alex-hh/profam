"""For use on e.g. JADE, where nodes don't have internet access."""
import os
import subprocess
from datetime import datetime, timedelta


# Define time windows for recent updates
time_window_5_hours = timedelta(hours=5)
time_window_24_hours = timedelta(hours=24)
current_time = datetime.now()

# Command to list details of train.log files in all folders
ls_command = 'ls -l */train.log'

# Run the command and capture the output
output = subprocess.check_output(ls_command, shell=True).decode('utf-8')

# Split the output by lines to process each file
lines = output.strip().split('\n')

# Iterate through each line
for line in lines:
    parts = line.split()
    
    # Example line: -rw-r--r-- 1 axh06-dxt03 dxt03 25380 Sep 15 02:16 2024-09-15_02-15-03/train.log
    # File modification date is stored in parts[5:8]
    
    # Reconstruct the modification time
    month = parts[5]
    day = parts[6]
    time_or_year = parts[7]  # Could be the time or the year depending on the file's age
    file_path = parts[8]
    
    # Combine date info to create a datetime object
    file_mod_time_str = f"{month} {day} {time_or_year}"
    print("Inferred file modification time:", file_mod_time_str, line)
    
    try:
        file_mod_time = datetime.strptime(file_mod_time_str, '%b %d %H:%M')  # If time is given
        file_mod_time = file_mod_time.replace(year=current_time.year)  # Ensure current year
    except ValueError:
        file_mod_time = datetime.strptime(file_mod_time_str, '%b %d %Y')  # If year is given
    
    # Extract the folder name from the file path
    folder_name = os.path.dirname(file_path)
    
    # Check if the file was modified within the last 5 hours
    if current_time - file_mod_time <= time_window_5_hours:
        # Run wandb sync for the folder
        sync_command = f"wandb sync {folder_name} --sync-all"
        print(f"Running: {sync_command}")
        subprocess.run(sync_command, shell=True)
    elif current_time - file_mod_time > time_window_24_hours:
        # If the file hasn't been updated in the last 24 hours, ask the user if they want to delete the folder
        delete = input(
            f"Folder '{folder_name}' has not been updated in the last 24 hours."
            f"Do you want to delete it? (y/n): "
        ).strip().lower()
        if delete == 'y':
            # Delete the folder
            delete_command = f"rm -rf {folder_name}"
            print(f"Deleting folder: {folder_name}")
            subprocess.run(delete_command, shell=True)
        else:
            print(f"Folder '{folder_name}' was not deleted.")
