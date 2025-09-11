"""
Takes fasta files generated from sampling and the corresponding target sequence fasta
as the query and aligns them outputting a .a3m file for each.
these are then used as synthetic MSAs for ColabFold evaluations.
"""
import os
import sys
import glob
import shutil
import tempfile
import subprocess

def run(cmd):
    print(f"[cmd] {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            # Some mmseqs versions log to stderr even on success
            print(result.stderr.strip())
        return result
    except subprocess.CalledProcessError as e:
        print("Command failed:")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        raise


if __name__=="__main__":

    # Inputs (adjust if needed)
    queries = glob.glob("../CASP16/target_fastas/*.fasta")
    sampled_fasta_dir = "../CASP16/ProFam_synthetic_msas"
    threads = int(os.environ.get("MMSEQS_THREADS", "24"))
    hhfilter_binary = os.environ.get("HHFILTER_BINARY", "/mnt/disk2/msa_pairformer/hhsuite/hhfilter")

    for query in queries:
        query_id = query.split("/")[-1].split(".")[0]
        sampled_fasta_path = os.path.join(sampled_fasta_dir, f"{query_id}_generated.fasta")
        output_path = os.path.join(sampled_fasta_dir, f"{query_id}.a3m")
        if os.path.exists(output_path):
            print(f"Skipping {query_id} as output file already exists")
            continue

        if not os.path.exists(sampled_fasta_path):
            print(f"Skipping {query_id} as sampled fasta file does not exist")
            continue

        # Absolute paths for robustness
        query_fa = os.path.abspath(query)
        target_fa = os.path.abspath(sampled_fasta_path)
        output_path = os.path.abspath(output_path)

        # Temporary work directory for mmseqs dbs
        workdir = tempfile.mkdtemp(prefix=f"mmseqs_{query_id}_")
        try:
            qdb = os.path.join(workdir, "queryDB")
            tdb = os.path.join(workdir, "targetDB")
            rdb = os.path.join(workdir, "resultDB")
            tmp = os.path.join(workdir, "tmp")

            os.makedirs(tmp, exist_ok=True)

            # 1) Create databases
            run(f"mmseqs createdb {query_fa} {qdb}")
            run(f"mmseqs createdb {target_fa} {tdb}")

            # 2) Search (prefilter + align) targets against the single query sequence
            #    -a 1 stores backtrace needed for MSA reconstruction
            run(
                f"mmseqs search {qdb} {tdb} {rdb} {tmp} -a 1 --threads {threads}"
            )

            # 3) Convert search results to A3M MSA anchored to the query
            #    (Default output is A3M if file ends with .a3m)
            run(f"mmseqs result2msa {qdb} {tdb} {rdb} {output_path}")

            # 4) Filter redundancy/coverage with hhfilter
            #    Remove sequences >98% identical and with <5% coverage
            #    Keep final output path the same by writing to a temp file first
            filtered_out = os.path.join(workdir, f"{query_id}.filtered.a3m")
            run(
                f"{hhfilter_binary} -i {output_path} -o {filtered_out} -id 98 -cov 5 -M a3m"
            )
            shutil.move(filtered_out, output_path)

            print(f"Wrote filtered A3M to: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {query_id}: {e}")
        finally:
            # Clean up temporary directory
            shutil.rmtree(workdir, ignore_errors=True)
        if os.path.exists(filtered_out + ".dbtype"):
            os.remove(filtered_out + ".dbtype")
        if os.path.exists(filtered_out + ".index"):
            os.remove(filtered_out + ".index")