"""
Create foldseek database.
We'll probably still load cluster_dict into memory. But the db will allow us to directly load all zipfiles / ids for a cluster.

# TODO: save backbone coordinates into db?
"""
import argparse
import pickle
import os
import time
from src.constants import PROFAM_DATA_DIR
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()
af50_path = os.path.join(PROFAM_DATA_DIR, "afdb/5-allmembers-repId-entryId-cluFlag-taxId.tsv")


class Protein(Base):
    __tablename__ = 'protein_metadata'
    uniprot_id = Column(String, primary_key=True, nullable=False)
    foldseek_cluster_id = Column(String, nullable=False)
    af50_cluster_id = Column(String, nullable=False)
    # is_foldseek_representative = Column(Boolean, default=False)  # these can be inferred by comparing foldseek_cluster_id and uniprot_id
    # is_af50_representative = Column(Boolean, default=False)
    zip_filename = Column(String, nullable=False)


def add_entries(entries, session, existing_uniprot_ids = None):
    try:
        for entry in entries:
            session.add(entry)
        session.commit()
    except:
        session.rollback()
        if existing_uniprot_ids is not None:
            unique_entries = [entry for entry in entries if entry.uniprot_id not in existing_uniprot_ids]
            for entry in unique_entries:
                session.add(entry)
            session.commit()
        else:
            for entry in entries:
                try:
                    session.add(entry)
                    session.commit()
                except:
                    print("Error adding entry", entry, flush=True)
                    pass


def make_cluster_db(
    start_index=0,
    minimum_foldseek_cluster_size=1,
):
    t0 = time.time()
    print(f"Creating foldseek database, starting from cluster index {start_index}", flush=True)
    db_path = os.path.join(PROFAM_DATA_DIR, "foldseek_clusters.db")
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)  # create table if it doesn't exist

    Session = sessionmaker(bind=engine)
    session = Session()
    # Query the table
    # existing_uniprot_ids = set()
    # for entry in session.query(Protein).yield_per(100000):
    #     existing_uniprot_ids.add(entry.uniprot_id)

    # Print the entries
    # print(f"Current entries in the database: {len(existing_uniprot_ids)}")
    print("Creating foldseek dataset", flush=True)
    # cluster_dict = make_cluster_dictionary("/SAN/orengolab/cath_plm/ProFam/data/afdb/1-AFDBClusters-entryId_repId_taxId.tsv")
    with open(os.path.join(PROFAM_DATA_DIR, "afdb/foldseek_cluster_dict.pkl"), "rb") as f:
        cluster_dict = pickle.load(f)
    cluster_ids = sorted(list(cluster_dict.keys()))
    print("Number of clusters:", len(cluster_dict))

    zip_dict_path = os.path.join(PROFAM_DATA_DIR, "afdb", "af2zip.pkl")
    print("loading precomputed zip index")
    with open(zip_dict_path, "rb") as f:
        af2zip = pickle.load(f)

    af50_dict_path = os.path.join(PROFAM_DATA_DIR, "afdb", "af50_cluster_dict.pkl")
    print("loading af50 dictionary")
    with open(af50_dict_path, "rb") as f:
        af50_dict = pickle.load(f)

    t1 = time.time()
    print("Setup (dictionary loading) time:", t1 - t0, "seconds", flush=True)
    # TODO: track failures.
    entries = []
    for ix, cluster_id in enumerate(cluster_ids):
        if ix < start_index:
            continue
        members = cluster_dict.pop(cluster_id)
        # handling upserts:
        # https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#insert-on-conflict-upsert
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

                entries.append(Protein(**entry))
                # stmt = insert(Protein.__table__).values(**entry)
                # stmt = stmt.prefix_with("OR IGNORE")
                # stmt = stmt.on_conflict_do_nothing(index_elements=["uniprot_id"])
                # session.execute(stmt)

                if member in af50_dict:
                    for af50_member in af50_dict[member]:
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

                        entries.append(Protein(**entry))
                        # stmt = insert(Protein.__table__).values(**entry)
                        # stmt = stmt.on_conflict_do_nothing(index_elements=["uniprot_id"])
                        # session.execute(stmt)
                        # print("Error looking up", af50_member)

                else:
                    print("No af50 members for", member)

        if ix % 1000 == 0:
            print(f"Processed {ix} clusters", flush=True)
            add_entries(entries, session)
            entries = []
            # existing_uniprot_ids = existing_uniprot_ids.union(set([entry.uniprot_id for entry in entries]))

    add_entries(entries, session)
    session.close()
    t2 = time.time()
    print("Built database in", t2 - t1, "seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", default=0, type=int)
    parser.add_argument("--minimum_foldseek_cluster_size", type=int, default=1)
    args = parser.parse_args()

    make_cluster_db(minimum_foldseek_cluster_size=args.minimum_foldseek_cluster_size, start_index=args.start_index)
