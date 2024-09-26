import os
import subprocess


def run_foldmason_on_pdbs(filelist, output_dir, tmp_dir):
    cmd = (
        ["foldmason", "easy-msa"]
        + filelist
        + [os.path.join(output_dir, "result"), tmp_dir]
    )

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"FoldMason stdout: {result.stdout}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"FoldMason execution failed: {e}", flush=True)
        print(f"FoldMason stderr: {e.stderr}", flush=True)
        raise
