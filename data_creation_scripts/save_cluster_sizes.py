"""Create a csv file with three columns: cluster_id,n_foldseek,n_af50."""
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()


class Protein(Base):
    __tablename__ = 'protein_metadata'
    uniprot_id = Column(String, primary_key=True, nullable=False)
    foldseek_cluster_id = Column(String, nullable=False)
    af50_cluster_id = Column(String, nullable=False)
    # is_foldseek_representative = Column(Boolean, default=False)  # these can be inferred by comparing foldseek_cluster_id and uniprot_id
    # is_af50_representative = Column(Boolean, default=False)
    zip_filename = Column(String, nullable=False)

# Create engine to connect to the database
engine = create_engine('sqlite:////SAN/orengolab/cath_plm/ProFam/data/foldseek_clusters.db')

# Create a session
Session = sessionmaker(bind=engine)

session = Session()

processed_clusters = set()

cluster_path = "/SAN/orengolab/cath_plm/ProFam/data/afdb/1-AFDBClusters-entryId_repId_tax_Id.tsv"
cluster_sizes = "/SAN/orengolab/cath_plm/ProFam/data/afdb/cluster_sizes.csv"

with open(cluster_path, "r") as f:
    with open(cluster_sizes, "w") as output_file:
        output_file.write("cluster_id,n_foldseek,n_af50\n")
        for line in f:
            line = line.strip().split("\t")
            entry_id = line[0]
            rep_id = line[1]
            if rep_id not in processed_clusters:
                all_members = session.query(Protein).filter(Protein.foldseek_cluster_id == rep_id).all()
                foldseek_members = [x for x in all_members if x.uniprot_id == x.af50_cluster_id]
                output_file.write(f"{rep_id},{len(foldseek_members)},{len(all_members)}\n")
                processed_clusters.add(rep_id)
