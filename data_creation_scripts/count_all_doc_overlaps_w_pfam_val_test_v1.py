import os
import pandas as pd

class BaseOverlapCounter:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def count_overlaps(self):
        """
        returns a pd.DataFrame where each row corresponds
        to one document and the columns are the counts for 
        each of the pfam families in the val/test set
        """
        pass

class FastaOverlapCounter(BaseOverlapCounter):
    def __init__(self, fasta_dir):
        super().__init__(data_dir)
        self.fasta_dir = fasta_dir
    
    def get_fam_id_from_docpath(self, doc_path):
        pass

    def up_id_from_line(self, fasta_line):
        """
        returns the uniprot ID
        from the identifier line of a fasta file
        """
        pass

    def get_up_ids_from_fasta_lines(self, fasta_lines):
        up_ids = []
        for line in fasta_lines:
            if line.startswith(">"):
                up_ids.append(self.up_id_from_line(line))
        return up_ids

    def get_fam_id_up_ids(self):
        # return dict {fam_id: [up_ids]}
        pass

class ECOverlapCounter(FastaOverlapCounter):
    def __init__(
            self, 
            fasta_dir="../data/ec/ec_fastas"
            ):
        super().__init__(fasta_dir)
    
    def get_fam_id_from_docpath(self, doc_path):
        return doc_path.split("/")[-1].split(".")[0]
    
    def up_id_from_line(self, fasta_line):
        return fasta_line.split("|")[1]




class TEDOverlapCounter(FastaOverlapCounter):
    def __init__(
            self, 
            fasta_dir="ted/ted_s50_by_sfam"
            ):
        super().__init__(fasta_dir)
    
    def get_fam_id_from_docpath(self, doc_path):
        return doc_path.split("/")[-1].split(".")[0]
    
    def up_id_from_line(self, fasta_line):
        return fasta_line.split("|")[1]


class FoldseekOverlapCounter(BaseOverlapCounter):
    def __init__(
            data_dir, 
            foldseek_cluster_index_file="../visualise_families/1-AFDBClusters-entryId_repId_taxId.tsv",
            ):
        self.foldseek_cluster_index_file = foldseek_cluster_index_file
        super().__init__(data_dir)
    
    def get_fam_id_up_ids(self):
        id_clust_tax = pd.read_csv(
                self.foldseek_cluster_index_file, 
                sep="\t", 
                header=None, 
                names=["id", "clust", "tax"]
                )
        # refactor so to dict {clust: [up_ids]}
        clust_to_up_ids = {}
        for index, row in id_clust_tax.iterrows():
            if row["clust"] not in clust_to_up_ids:
                clust_to_up_ids[row["clust"]] = []
            clust_to_up_ids[row["clust"]].append(row["id"])
        return clust_to_up_ids

    
    

if __name__ == "__main__":
    base_data_dir = "../data"
    pfam_val_test_csv = "data/val_test/pfam/pfam_val_test_accessions_w_unip_accs.csv"
    