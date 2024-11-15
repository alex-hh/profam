"""It makes sense to test this out for e.g. a single parquet first."""
import argparse
import os
import pandas as pd
from bio_datasets import Features, ProteinAtomArrayFeature, ProteinStructureFeature, Value, Array1D, Dataset
from src.data.builders.hf_datasets import StructureDocumentMixin



def examples_generator(shards: list[str]):
    for shard in shards:
        df = pd.read_parquet(shard)
        for _, row in df.iterrows():
            # we can convert back to a backbone dictionary
            proteins = StructureDocumentMixin.build_document(row)
            # proteins needs to be a list of backbone dictionaries (compatible with ProteinAtomArrayFeature.encode_example)
            yield {"id": proteins.identifier, "identifiers": proteins.accessions, "proteins": proteins.to_atom_array()}


def main(args):
    shards = [args.data_path]
    if args.feature_type == "structure":
        proteins_feature = ProteinStructureFeature(load_as="biotite")
    else:
        assert args.feature_type == "array"
        proteins_feature = ProteinAtomArrayFeature.from_preset("afdb", backbone_only=True, load_as="biotite")
    features = Features(
        proteins=proteins_feature,
        identifiers=Array1D((None,), "string"),
        id=Value("string"),
    )
    ds = Dataset.from_generator(examples_generator, gen_kwargs={"shards": [os.path.join(args.data_path, "0.parquet")]}, features=features)
    ds.save_to_disk(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--feature_type", choices=["array", "structure"], default="array")
    args = parser.parse_args()
    main(args)
