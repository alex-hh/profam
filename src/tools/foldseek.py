import os
import subprocess
import time

import pandas as pd
import requests

# Base URL for the API
FOLDSEEK_API_URL = "https://search.foldseek.com/api/"


def convert_pdbs_to_3di(pdb_files, output_file):
    cmd = ["foldseek", "structureto3didescriptor"] + pdb_files + [output_file]
    filenames = [os.path.basename(f) for f in pdb_files]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Foldseek stdout: {result.stdout}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Foldseek execution failed: {e}", flush=True)
        print(f"Foldseek input: {cmd}", flush=True)
        print(f"Foldseek stderr: {e.stderr}", flush=True)
        raise
    mapping = pd.read_csv(
        output_file, sep="\t", names=["pdb", "aa", "3di", "ca"]
    ).set_index("pdb")
    return mapping.loc[filenames]["3di"].tolist()


# Step 1: Submit tickets
# Step 1: Submit a file to get a ticket
def submit_file(file_path, mode="3diaa", databases=None):
    url = f"{FOLDSEEK_API_URL}/ticket"

    # If databases are not provided, use a default list
    if databases is None:
        databases = [
            "afdb50",
            "afdb-swissprot",
            "afdb-proteome",
            "bfmd",
            "cath50",
            "mgnify_esm30",
            "pdb100",
            "gmgcl_id",
        ]

    # Prepare the form data
    form_data = {"q": open(file_path, "rb"), "mode": mode}

    # Add multiple databases[] parameters
    for db in databases:
        form_data[f"database[]"] = db

    # Send the POST request with the file and parameters
    response = requests.post(url, files=form_data)

    if response.status_code == 200:
        # Parse the JSON response to get ticket information
        tickets = response.json()  # Should be an array of ticket objects
        return tickets
    else:
        raise Exception(
            f"Failed to submit file. Status code: {response.status_code}, Response: {response.text}"
        )


# Step 2: Poll ticket status
def poll_ticket_status(ticket_id):
    url = f"{FOLDSEEK_API_URL}/ticket/{ticket_id}"

    while True:
        response = requests.get(url)
        if response.status_code == 200:
            ticket_info = response.json()
            status = ticket_info["status"]
            print(f"Ticket {ticket_id} status: {status}")

            if status == "COMPLETE":
                return ticket_info
            elif status in ["ERROR", "UNKNOWN"]:
                raise Exception(f"Ticket {ticket_id} encountered an error: {status}")
        else:
            raise Exception(
                f"Failed to get status for ticket {ticket_id}. Status code: {response.status_code}, Response: {response.text}"
            )

        # Wait for a while before polling again
        time.sleep(5)


# Step 3: Download results
def download_result(ticket_id):
    url = f"{FOLDSEEK_API_URL}/result/download/{ticket_id}"

    response = requests.get(url)

    if response.status_code == 200:
        # Save the results to a file
        with open(f"{ticket_id}_result.txt", "wb") as file:
            file.write(response.content)
        print(f"Results for ticket {ticket_id} downloaded successfully.")
    else:
        raise Exception(
            f"Failed to download result for ticket {ticket_id}. Status code: {response.status_code}, Response: {response.text}"
        )


# Main function to handle the entire workflow
def process_file_submission(file_path):
    """
        Use this command to get a submit a file in PDB or mmCIF format to the Foldseek search server. Replace the ‘PATH_TO_FILE’ string with the path to the file.
    curl -X POST -F q=@PATH_TO_FILE -F 'mode=3diaa' -F 'database[]=afdb50' -F 'database[]=afdb-swissprot' -F 'database[]=afdb-proteome' -F 'database[]=bfmd' -F 'database[]=cath50' -F 'database[]=mgnify_esm30' -F 'database[]=pdb100' -F 'database[]=gmgcl_id' https://search.foldseek.com/api/ticket
    """
    # Step 1: Submit the file and get ticket information
    print("Submitting file to Foldseek...")
    tickets = submit_file(file_path)

    # Step 2: Poll status for each submitted ticket
    for ticket in tickets:
        ticket_id = ticket["id"]
        print(f"Polling status for ticket {ticket_id}...")
        poll_ticket_status(ticket_id)

    # Step 3: Download results for each completed ticket
    for ticket in tickets:
        ticket_id = ticket["id"]
        print(f"Downloading result for ticket {ticket_id}...")
        download_result(ticket_id)
