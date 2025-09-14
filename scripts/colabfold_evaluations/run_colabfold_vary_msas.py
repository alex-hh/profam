import glob
import os
import sys
import shutil
from src.structure.superimposition import rmsd, tm_score, lddt
from src.data.objects import Protein
import pandas as pd
import numpy as np

def get_metrics(pdb_path_1, pdb_path_2):
    # rmsd and tm score
    prot1 = Protein.from_pdb(pdb_path_1, bfactor_is_plddt=True)
    prot2 = Protein.from_pdb(pdb_path_2, bfactor_is_plddt=True)
    metrics = {
        "seq_len_1": len(prot1),
        "seq_len_2": len(prot2),
        "mean_plddt_1": float(np.mean(prot1.plddt)) if prot1.plddt is not None else float("nan"),
        "mean_plddt_2": float(np.mean(prot2.plddt)) if prot2.plddt is not None else float("nan"),
    }
    try:
        metrics["tm_score"] = float(tm_score(prot1, prot2))
    except Exception:
        metrics["tm_score"] = float("nan")
    try:
        metrics["rmsd"] = float(rmsd(prot2, prot1)) if len(prot1) == len(prot2) else float("nan")
    except Exception:
        metrics["rmsd"] = float("nan")
    try:
        metrics["lddt"] = float(lddt(prot1, prot2))
    except Exception:
        metrics["lddt"] = float("nan")
    return metrics


def _list_pdbs(dir_path):
    return [p for p in glob.glob(os.path.join(dir_path, "*.pdb")) if os.path.isfile(p)]


def _pick_best_pred_pdb(pred_dir):
    pdbs = _list_pdbs(pred_dir)
    if len(pdbs) == 0:
        return None, float("nan")
    best_path = None
    best_plddt = -1.0
    for p in pdbs:
        try:
            prot = Protein.from_pdb(p, bfactor_is_plddt=True)
            mean_plddt = float(np.mean(prot.plddt)) if prot.plddt is not None else float("nan")
            # Prefer higher PLDDT; NaN treated as -inf equivalent
            score = mean_plddt if not np.isnan(mean_plddt) else -1.0
            if best_path is None or score > best_plddt:
                best_path = p
                best_plddt = score
        except Exception:
            continue
    return best_path, best_plddt if best_path is not None else float("nan")

if __name__=="__main__":
    previous_path = os.environ["PATH"]
    os.environ["PATH"] = f"/mnt/disk2/colabfold_2025_09/localcolabfold/colabfold-conda/bin:{previous_path}"
    input_fasta_pattern = "../CASP16/target_fastas/*.fasta"
    input_fastas = glob.glob(input_fasta_pattern)
    rows = []
    for input_fasta in input_fastas:
        casp_id = input_fasta.split("/")[-1].split(".")[0]
        try:
            ground_truth_pdb_path = glob.glob(f"../CASP16/pdbs/{casp_id}*.pdb")[0]
        except IndexError:
            print(f"No ground-truth PDB found for {casp_id}; skipping")
            continue

        output_dir_with = f"../CASP16/colabfold_outputs/{casp_id}/{casp_id}_with_MSA"
        if not os.path.exists(output_dir_with) or len(os.listdir(output_dir_with)) == 0:
            command = f"colabfold_batch {input_fasta} ../CASP16/colabfold_outputs/{casp_id}/{casp_id}_with_MSA"
            os.system(command)
        else:
            print(f"Skipping {input_fasta} (with MSA) as output directory already exists")

        output_dir_no = f"../CASP16/colabfold_outputs/{casp_id}/{casp_id}_no_MSA"
        if not os.path.exists(output_dir_no) or len(os.listdir(output_dir_no)) == 0:
            command = f"colabfold_batch {input_fasta} ../CASP16/colabfold_outputs/{casp_id}/{casp_id}_no_MSA --msa-mode single_sequence"
            os.system(command)
        else:
            print(f"Skipping {input_fasta} (no MSA) as output directory already exists")


        output_dir_with_synthetic = f"../CASP16/colabfold_outputs/{casp_id}/{casp_id}_with_synthetic_MSA"
        if not os.path.exists(output_dir_with_synthetic) or len(os.listdir(output_dir_with_synthetic)) == 0:
            a3m_path = f"../CASP16/ProFam_synthetic_msas/{casp_id}.a3m"
            if not os.path.exists(a3m_path):
                print(f"Skipping {input_fasta} (with synthetic MSA) as a3m file does not exist")
                continue
            command = f"colabfold_batch {a3m_path} {output_dir_with_synthetic} --num-models 1"
            os.system(command)
        else:
            print(f"Skipping {input_fasta} (with synthetic MSA) as output directory already exists")

        output_dir_with_synthetic_1200 = f"../CASP16/colabfold_outputs/{casp_id}/{casp_id}_with_synthetic_MSA_1200"
        if not os.path.exists(output_dir_with_synthetic_1200) or len(os.listdir(output_dir_with_synthetic_1200)) == 0:
            a3m_path = f"../CASP16/ProFam_synthetic_msas_1200/{casp_id}.a3m"
            if not os.path.exists(a3m_path):
                print(f"Skipping {input_fasta} (with synthetic MSA 1200) as a3m file does not exist")
                continue
            command = f"colabfold_batch {a3m_path} {output_dir_with_synthetic_1200} --num-models 1"
            os.system(command)
        else:
            print(f"Skipping {input_fasta} (with synthetic MSA) as output directory already exists")


        output_dir_with_synthetic_poet = f"../CASP16/colabfold_outputs/{casp_id}/{casp_id}_with_synthetic_MSA_poet"
        if not os.path.exists(output_dir_with_synthetic_poet) or len(os.listdir(output_dir_with_synthetic_poet)) == 0:
            a3m_path = f"../CASP16/poet_casp16_synthetic_msas/{casp_id}.a3m"
            if not os.path.exists(a3m_path):
                print(f"Skipping {input_fasta} (with synthetic MSA poet) as a3m file does not exist")
                continue
            command = f"colabfold_batch {a3m_path} {output_dir_with_synthetic_poet} --num-models 1"
            os.system(command)
        else:
            print(f"Skipping {input_fasta} (with synthetic MSA poet) as output directory already exists")


        output_dir_with_random = f"../CASP16/colabfold_outputs/{casp_id}/{casp_id}_with_random_MSA"
        if not os.path.exists(output_dir_with_random) or len(os.listdir(output_dir_with_random)) == 0:
            a3m_path = f"../CASP16/random_synthetic_msas/{casp_id}.a3m"
            if not os.path.exists(a3m_path):
                print(f"Skipping {input_fasta} (with random MSA) as a3m file does not exist")
                continue
            command = f"colabfold_batch {a3m_path} {output_dir_with_random} --num-models 1"
            os.system(command)
        else:
            print(f"Skipping {input_fasta} (with random MSA) as output directory already exists")

        best_with_pdb, best_with_plddt = _pick_best_pred_pdb(output_dir_with) if os.path.exists(output_dir_with) else (None, float("nan"))
        best_no_pdb, best_no_plddt = _pick_best_pred_pdb(output_dir_no) if os.path.exists(output_dir_no) else (None, float("nan"))
        best_with_synthetic_pdb, best_with_synthetic_plddt = _pick_best_pred_pdb(output_dir_with_synthetic) if os.path.exists(output_dir_with_synthetic) else (None, float("nan"))
        best_with_random_pdb, best_with_random_plddt = _pick_best_pred_pdb(output_dir_with_random) if os.path.exists(output_dir_with_random) else (None, float("nan"))
        best_with_synthetic_1200_pdb, best_with_synthetic_1200_plddt = _pick_best_pred_pdb(output_dir_with_synthetic_1200) if os.path.exists(output_dir_with_synthetic_1200) else (None, float("nan"))
        best_with_synthetic_poet_pdb, best_with_synthetic_poet_plddt = _pick_best_pred_pdb(output_dir_with_synthetic_poet) if os.path.exists(output_dir_with_synthetic_poet) else (None, float("nan"))

        # Compute metrics
        tm_with = rmsd_with = float("nan")
        tm_no = rmsd_no = float("nan")
        tm_with_synthetic = rmsd_with_synthetic = float("nan")
        gt_len = pred_with_len = pred_no_len = pred_with_synthetic_len = None
        if best_with_pdb is not None:
            m = get_metrics(best_with_pdb, ground_truth_pdb_path)
            tm_with = m.get("tm_score", float("nan"))
            rmsd_with = m.get("rmsd", float("nan"))
            pred_with_len = m.get("seq_len_1")
            gt_len = m.get("seq_len_2")
            lddt_with = m.get("lddt", float("nan"))
        if best_no_pdb is not None:
            m = get_metrics(best_no_pdb, ground_truth_pdb_path)
            tm_no = m.get("tm_score", float("nan"))
            rmsd_no = m.get("rmsd", float("nan"))
            pred_no_len = m.get("seq_len_1")
            gt_len = gt_len if gt_len is not None else m.get("seq_len_2")
            lddt_no = m.get("lddt", float("nan"))
        if best_with_synthetic_pdb is not None:
            m = get_metrics(best_with_synthetic_pdb, ground_truth_pdb_path)
            tm_with_synthetic = m.get("tm_score", float("nan"))
            rmsd_with_synthetic = m.get("rmsd", float("nan"))
            pred_with_synthetic_len = m.get("seq_len_1")
            gt_len = gt_len if gt_len is not None else m.get("seq_len_2")
            lddt_with_synthetic = m.get("lddt", float("nan"))
        if best_with_random_pdb is not None:
            m = get_metrics(best_with_random_pdb, ground_truth_pdb_path)
            tm_with_random = m.get("tm_score", float("nan"))
            rmsd_with_random = m.get("rmsd", float("nan"))
            pred_with_random_len = m.get("seq_len_1")
            gt_len = gt_len if gt_len is not None else m.get("seq_len_2")
            lddt_with_random = m.get("lddt", float("nan"))  
        if best_with_synthetic_1200_pdb is not None:
            m = get_metrics(best_with_synthetic_1200_pdb, ground_truth_pdb_path)
            tm_with_synthetic_1200 = m.get("tm_score", float("nan"))
            rmsd_with_synthetic_1200 = m.get("rmsd", float("nan"))
            pred_with_synthetic_1200_len = m.get("seq_len_1")
            gt_len = gt_len if gt_len is not None else m.get("seq_len_2")
            lddt_with_synthetic_1200 = m.get("lddt", float("nan"))
        if best_with_synthetic_poet_pdb is not None:
            m = get_metrics(best_with_synthetic_poet_pdb, ground_truth_pdb_path)
            tm_with_synthetic_poet = m.get("tm_score", float("nan"))
            rmsd_with_synthetic_poet = m.get("rmsd", float("nan"))
            pred_with_synthetic_poet_len = m.get("seq_len_1")
            gt_len = gt_len if gt_len is not None else m.get("seq_len_2")
            lddt_with_synthetic_poet = m.get("lddt", float("nan"))
        
        rows.append({
            "casp_id": casp_id,
            "gt_pdb": ground_truth_pdb_path,
            "pred_with_msa_pdb": best_with_pdb or "",
            "pred_with_msa_mean_plddt": best_with_plddt if not np.isnan(best_with_plddt) else "",
            "pred_no_msa_mean_plddt": best_no_plddt if not np.isnan(best_no_plddt) else "",
            "pred_with_synthetic_mean_plddt": best_with_synthetic_plddt if not np.isnan(best_with_synthetic_plddt) else "",
            "pred_with_random_mean_plddt": best_with_random_plddt if not np.isnan(best_with_random_plddt) else "",
            "pred_with_synthetic_1200_mean_plddt": best_with_synthetic_1200_plddt if not np.isnan(best_with_synthetic_1200_plddt) else "",
            "pred_with_synthetic_poet_mean_plddt": best_with_synthetic_poet_plddt if not np.isnan(best_with_synthetic_poet_plddt) else "",
            "tm_with_msa": tm_with if not np.isnan(tm_with) else "",
            "tm_no_msa": tm_no if not np.isnan(tm_no) else "",
            "tm_with_synthetic": tm_with_synthetic if not np.isnan(tm_with_synthetic) else "",
            "tm_with_random": tm_with_random if not np.isnan(tm_with_random) else "",
            "tm_with_synthetic_1200": tm_with_synthetic_1200 if not np.isnan(tm_with_synthetic_1200) else "",
            "tm_with_synthetic_poet": tm_with_synthetic_poet if not np.isnan(tm_with_synthetic_poet) else "",
            "rmsd_with_msa": rmsd_with if not np.isnan(rmsd_with) else "",
            "rmsd_no_msa": rmsd_no if not np.isnan(rmsd_no) else "",
            "rmsd_with_synthetic": rmsd_with_synthetic if not np.isnan(rmsd_with_synthetic) else "",
            "rmsd_with_random": rmsd_with_random if not np.isnan(rmsd_with_random) else "",
            "rmsd_with_synthetic_1200": rmsd_with_synthetic_1200 if not np.isnan(rmsd_with_synthetic_1200) else "",
            "rmsd_with_synthetic_poet": rmsd_with_synthetic_poet if not np.isnan(rmsd_with_synthetic_poet) else "",
            "lddt_with_msa": lddt_with if not np.isnan(lddt_with) else "",
            "lddt_no_msa": lddt_no if not np.isnan(lddt_no) else "",
            "lddt_with_synthetic": lddt_with_synthetic if not np.isnan(lddt_with_synthetic) else "",
            "lddt_with_random": lddt_with_random if not np.isnan(lddt_with_random) else "",
            "lddt_with_synthetic_1200": lddt_with_synthetic_1200 if not np.isnan(lddt_with_synthetic_1200) else "",
            "lddt_with_synthetic_poet": lddt_with_synthetic_poet if not np.isnan(lddt_with_synthetic_poet) else "",
            "gt_len": gt_len if gt_len is not None else "",
            "pred_with_len": pred_with_len if pred_with_len is not None else "",
            
        })

    metrics_csv = "../CASP16/colabfold_outputs/metrics_vary_msas.csv"
    df = pd.DataFrame(rows)
    df.to_csv(metrics_csv, index=False)
    os.environ["PATH"] = previous_path