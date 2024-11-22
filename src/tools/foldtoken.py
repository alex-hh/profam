import subprocess
import os

def run_foldtoken_on_pdbs(pdb_folder, save_vqid_path, level=8):
    # Set up the command to run the Python script

    cmd = [
        "python", "src/tools/foldtoken/extract_vq_ids.py",
        "--path_in", str(pdb_folder),
        "--save_vqid_path", save_vqid_path,
        "--level", str(level)
    ]

    # Set the PYTHONPATH environment variable to include the foldtoken directory
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/tools/foldtoken'

    try:
        # Run the command with the updated environment
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print(f"ExtractVQIDs output saved to: {save_vqid_path}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"ExtractVQIDs execution failed: {e}", flush=True)
        print(f"ExtractVQIDs stderr: {e.stderr}", flush=True)
        raise

# Example of how to use the modified function
# run_extract_vq_ids("/SAN/orengolab/plm_embeds/profam/tmp_folder/106-106", "./N128_vqid.jsonl", 8)