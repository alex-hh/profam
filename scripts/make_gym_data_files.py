import argparse
import glob
import os


def main(args):
    msa_files = glob.glob(os.path.join(args.data_dir, "ProteinGym/DMS_msa_files/*.a2m"))
    val_msas = [
        "BLAT_ECOLX_full_11-26-2021_b02.a2m",
        "CALM1_HUMAN_full_11-26-2021_b03.a2m",
        "DYR_ECOLI_2023-08-07_b01.a2m",
        "DYR_ECOLI_full_11-26-2021_b08.a2m",
        "DLG4_RAT_full_11-26-2021_b03.a2m",
        "REV_HV1H2_full_theta0.99_04-29-2022_b09.a2m",
        "TAT_HV1BR_full_theta0.99_04-29-2022_b09.a2m",
        "RL40A_YEAST_full_11-26-2021_b01.a2m",
        "P53_HUMAN_full_04-29-2022_b09.a2m",
        "P53_HUMAN_full_11-26-2021_b09.a2m",
    ]
    train_msas = [
        os.path.join("ProteinGym/DMS_msa_files", os.path.basename(f))
        for f in msa_files
        if os.path.basename(f) not in val_msas
    ]
    val_msas = [
        os.path.join("ProteinGym/DMS_msa_files", os.path.basename(f))
        for f in msa_files
        if os.path.basename(f) in val_msas
    ]
    with open(os.path.join(args.data_dir, "ProteinGym/msa_files_no_val.txt"), "w") as f:
        f.write("\n".join(train_msas))
    with open(os.path.join(args.data_dir, "ProteinGym/msa_files_val.txt"), "w") as f:
        f.write("\n".join(val_msas))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    main(args)
