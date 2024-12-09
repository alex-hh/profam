import json
import os
import subprocess


def run_foldtoken_on_pdbs(pdb_folder, save_vqid_path, level=8):

    """
    Extract vector foldtoken vqid from PDB files located in a specified folder.

    This function processes each PDB file in the given folder to compute vqid at a specified codebook size.
    The vqid and sequence of each PDB file are then saved.

    Parameters:
    - pdb_folder (str): Path to the folder containing PDB files.
    - save_vqid_path (str): Path where the output .jsonl file will be saved.
    - level (int, optional): codebook size, e.g. 2^8, defaults to 8.

    Raises:
    - subprocess.CalledProcessError: If the script execution fails.

    Returns:
    - None: Outputs are saved directly to files and any errors are printed to the standard error stream.

    """

    cmd = [
        "python",
        "src/tools/foldtoken/extract_vq_ids.py",
        "--path_in",
        str(pdb_folder),
        "--save_vqid_path",
        save_vqid_path,
        "--level",
        str(level),
    ]

    # Set the PYTHONPATH environment variable to include the foldtoken directory
    env = os.environ.copy()
    env["PYTHONPATH"] = "src/tools/foldtoken"

    try:

        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=env
        )
        print(f"ExtractVQIDs output saved to: {save_vqid_path}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"ExtractVQIDs execution failed: {e}", flush=True)
        print(f"ExtractVQIDs stderr: {e.stderr}", flush=True)
        raise


def read_foldtoken_output(foldtoken_output_file):
    """
    Read the output of the foldtoken script.

    This function reads the output of the foldtoken script and returns the sequence and vqid for each PDB file.

    Parameters:
    - foldtoken_outdir (str): Path to the folder containing the foldtoken output.

    Returns:
    - foldtoken_seqs (list): List of sequences.
    - foldtoken_vqid (list): List of vqid values.

    """

    data_dict = {
        key: value
        for line in open(foldtoken_output_file, "r")
        for key, value in json.loads(line).items()
    }
    labels = list(data_dict.keys())
    foldtoken_seqs = [entry["seq"] for entry in data_dict.values()]
    foldtoken_vqid = [",".join(map(str, entry["vqid"])) for entry in data_dict.values()]

    return labels, foldtoken_seqs, foldtoken_vqid


# Example of how to use the modified function
# run_extract_vq_ids("/SAN/orengolab/plm_embeds/profam/tmp_folder/106-106", "./N128_vqid.jsonl", 8)
