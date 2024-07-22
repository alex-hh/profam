"""
Creates fasta files for the foldseek clusters
converts them into parquet files.

first create a dictionary for the clusters
then iterate through the file.
"""
import pickle
def make_cluster_dictionary(cluster_path):
    line_counter = 0
    cluster_dict = {}
    with open(cluster_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            entry_id = line[0]
            rep_id = line[1]
            tax_id = line[2]
            if rep_id not in cluster_dict:
                cluster_dict[rep_id] = []
            cluster_dict[rep_id].append(entry_id)
            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for cluster dictionary")
    return cluster_dict

def get_sequence_from_profam_db(uniprot_id, cursor):
    cursor.execute('SELECT sequence FROM sequences WHERE sequence_id = ?', (uniprot_id,))
    result = cursor.fetchone()
    return result[0] if result else None

def create_foldseek_fastas(db_path, cluster_dict, save_dir):
    for cluster_id, members in cluster_dict.items():


if __name__ == "__main__":
    print("Creating foldseek dataset")
    cluster_path = "../data/foldseek/1-AFDBClusters-entryId_repId_taxId.tsv"
    uniprot_db_path = "/SAN/orengolab/cath_plm/profam_db/profam.db"
    save_dir = "../data/foldseek/"
    uniref_fasta_file = "../"
    cluster_dict = make_cluster_dictionary(cluster_path)
    print("Number of clusters:", len(cluster_dict))
    print("Saving cluster dictionary")
    with open(save_dir + "foldseek_cluster_dict.pkl", "wb") as f:
        pickle.dump(cluster_dict, f)


