"""For use on e.g. JADE, where nodes don't have internet access."""
import os
import subprocess
from datetime import datetime, timedelta

import wandb

# Define time windows for recent updates
time_window_5_hours = timedelta(hours=5)
time_window_24_hours = timedelta(hours=24)
current_time = datetime.now()

# Command to list details of train.log files in all folders
ls_command = "ls -l */wandb/**/*.wandb"

# Run the command and capture the output
output = subprocess.check_output(ls_command, shell=True).decode("utf-8")

# Split the output by lines to process each file
lines = output.strip().split("\n")


# First loop: Process all files and store the most recent modification time for each folder
folder_mod_times = {}
for line in lines:
    parts = line.split()
    month, day, time_or_year = parts[5:8]
    file_path = parts[8]

    file_mod_time_str = f"{month} {day} {time_or_year}"
    try:
        file_mod_time = datetime.strptime(file_mod_time_str, "%b %d %H:%M")
        file_mod_time = file_mod_time.replace(year=current_time.year)
    except ValueError:
        file_mod_time = datetime.strptime(file_mod_time_str, "%b %d %Y")

    folder_name = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

    if (
        folder_name not in folder_mod_times
        or file_mod_time > folder_mod_times[folder_name]
    ):
        folder_mod_times[folder_name] = file_mod_time

# Second loop: Process only the most recent file for each folder
for folder_name, file_mod_time in folder_mod_times.items():
    print(f"Processing folder: {folder_name}")
    print(f"Last modified: {file_mod_time}")

    # The rest of the processing for the most recent file in each folder will go here

    # Check if the file was modified within the last 5 hours
    if current_time - file_mod_time <= time_window_5_hours:
        # Run wandb sync for the folder
        # wandb.sync(folder_name)
        sync_command = ["wandb", "sync", "--sync-all"]
        subprocess.run(sync_command, cwd=folder_name, shell=False)
    elif current_time - file_mod_time > time_window_24_hours:
        # If the file hasn't been updated in the last 24 hours, ask the user if they want to delete the folder
        delete = (
            input(
                f"Folder '{folder_name}' has not been updated in the last 24 hours."
                f"Do you want to delete it? (y/n): "
            )
            .strip()
            .lower()
        )
        if delete == "y":
            # Delete the folder
            delete_command = f"rm -rf {folder_name}"
            print(f"Deleting folder: {folder_name}")
            subprocess.run(delete_command, shell=True)
        else:
            print(f"Folder '{folder_name}' was not deleted.")
