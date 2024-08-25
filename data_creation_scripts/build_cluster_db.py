"""
Create foldseek database.
We'll probably still load cluster_dict into memory. But the db will allow us to directly load all zipfiles / ids for a cluster.
"""
import argparse
import time
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()
af50_path = "/SAN/orengolab/cath_plm/ProFam/data/afdb/5-allmembers-repId-entryId-cluFlag-taxId.tsv"


class Protein(Base):
    __tablename__ = 'protein_metadata'
    uniprot_id = Column(String, primary_key=True, nullable=False)
    foldseek_cluster_id = Column(String, nullable=False)
    af50_cluster_id = Column(String, nullable=False)
    # is_foldseek_representative = Column(Boolean, default=False)  # these can be inferred by comparing foldseek_cluster_id and uniprot_id
    # is_af50_representative = Column(Boolean, default=False)
    zip_filename = Column(String, nullable=False)


def make_af50_dictionary(clusters_to_include=None):
    line_counter = 0
    af50_dict = {}
    with open(af50_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            rep_id = line[0]
            entry_id = line[1]
            # 1: clustered in AFDB50, 2: clustered in AFDB clusters, 3/4: removed (fragments/singletons)
            clu_flag = int(line[2])  # 1
            # n.b. the 2s are duplicates of the other cluster dict
            # n.b. we don't include the representative in its own cluster atm
            if clu_flag == 1 and (clusters_to_include is None or rep_id in clusters_to_include):
                if rep_id not in af50_dict:
                    af50_dict[rep_id] = []
                af50_dict[rep_id].append(entry_id)
            line_counter += 1
    return af50_dict


def make_zip_dictionary():
    line_counter = 0
    af2zip = {}
    with open("/SAN/bioinf/afdb_domain/zipmaker/zip_index", "r") as f:
        for line in f:
            line = line.strip().split("\t")
            afdb_id = line[0]
            uniprot_id = afdb_id.split("-")[1]
            assert afdb_id == f"AF-{uniprot_id}-F1-model_v4"
            zip_file = line[2]
            af2zip[uniprot_id] = zip_file

            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for zip file dictionary")
    return af2zip


def make_cluster_dictionary(cluster_path):
    line_counter = 0
    cluster_dict = {}
    with open(cluster_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            entry_id = line[0]
            rep_id = line[1]
            if rep_id not in cluster_dict:
                cluster_dict[rep_id] = []
            cluster_dict[rep_id].append(entry_id)
            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for cluster dictionary")
    return cluster_dict


def make_cluster_db(
    minimum_foldseek_cluster_size=1,
):
    engine = create_engine('sqlite:////SAN/orengolab/cath_plm/ProFam/data/foldseek_clusters.db')
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    print("Creating foldseek dataset", flush=True)
    cluster_dict = make_cluster_dictionary("/SAN/orengolab/cath_plm/ProFam/data/afdb/1-AFDBClusters-entryId_repId_taxId.tsv")
    print("Number of clusters:", len(cluster_dict))
    af2zip = make_zip_dictionary()
    af50_dict = make_af50_dictionary()

    t1 = time.time()
    # TODO: track failures.
    for ix, cluster_id in enumerate(cluster_dict.keys()):
        members = cluster_dict[cluster_id]
        if len(members) >= minimum_foldseek_cluster_size:

            for member in members:
                try:
                    zip_filename = af2zip[member]
                    entry = {
                        "foldseek_cluster_id": cluster_id,
                        "af50_cluster_id": member,
                        "uniprot_id": member,
                        "zip_filename": zip_filename,
                    }
                except:
                    print("Error looking up", member)
                    entry = {
                        "foldseek_cluster_id": cluster_id,
                        "af50_cluster_id": member,
                        "uniprot_id": member,
                        "zip_filename": "",
                    }

                entry = Protein(**entry)
                session.add(entry)

                if member in af50_dict:
                    for af50_member in af50_dict[member]:
                        try:
                            assert not af50_member == member
                            try:
                                zip_filename = af2zip[af50_member]
                                entry = {
                                    "foldseek_cluster_id": cluster_id,
                                    "af50_cluster_id": member,
                                    "uniprot_id": af50_member,
                                    "zip_filename": zip_filename,
                                }
                            except:
                                print("Error looking up", af50_member)
                                entry = {
                                    "foldseek_cluster_id": cluster_id,
                                    "af50_cluster_id": member,
                                    "uniprot_id": af50_member,
                                    "zip_filename": "",
                                }

                            entry = Protein(**entry)
                            session.add(entry)
                        except:
                            print("Error looking up", af50_member)
                else:
                    print("No af50 members for", member)

        if ix % 1000 == 0:
            print(f"Processed {ix} clusters", flush=True)
            session.commit()

    session.commit()
    session.close()
    t2 = time.time()
    print("Built database in", t2 - t1, "seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum_foldseek_cluster_size", type=int, default=1)
    parser.add_argument("--parquet_ids", type=int, default=None, nargs="+")
    args = parser.parse_args()

    make_cluster_db(minimum_foldseek_cluster_size=args.minimum_foldseek_cluster_size)
