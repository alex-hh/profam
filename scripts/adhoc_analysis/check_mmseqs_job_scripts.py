import glob

"""
check that end of all job script files looks something like this:

Completed processing dataset: afdb_s50_single
Removed 3253 sequences from training set
Completed removing similar sequences from training sets
Wed 23 Apr 07:16:49 BST 2025
"""

job_script_pattern = "../qsub_logs/mmseqs/mmseqs_splitV2.o*"

job_scripts = glob.glob(job_script_pattern)

for job_script in job_scripts:
    with open(job_script, "r") as f:
        lines = f.readlines()[-4:]

    if not lines[0].startswith("Completed processing dataset:"):
        print(f"\n\nJob script {job_script} does not end with the expected message:")
        for line in lines:
            print(line.strip())
    elif not lines[1].startswith("Removed"):
        print(f"\n\nJob script {job_script} does not end with the expected message:")
        for line in lines:
            print(line.strip())
    elif not lines[2].startswith(
        "Completed removing similar sequences from training sets"
    ):
        print(f"\n\nJob script {job_script} does not end with the expected message:")
        for line in lines:
            print(line.strip())
    elif not lines[3].strip().endswith("BST 2025"):
        print(f"\n\nJob script {job_script} does not end with the expected message:")
        for line in lines:
            print(line.strip())
