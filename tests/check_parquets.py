import numpy as np
import pyarrow.parquet as pq

# Read metadata
parquet_file = pq.ParquetFile(
    "../data/pfam/train_test_split_parquets/val/val_000.parquet"
)
print(parquet_file.metadata)
for row in parquet_file.iter_batches(batch_size=1):
    df = row.to_pandas()
    d = df.iloc[0].to_dict()
    try:
        l = len(d["accessions"])
        if l > 100_000:
            print(l)
        assert isinstance(d["accessions"], np.ndarray)
        assert isinstance(d["sequences"], np.ndarray)
        assert isinstance(d["accessions"][0], str)
        assert isinstance(d["sequences"][0], str)
    except:
        for k, v in d.items():
            print(k)
            print(v)
            print()
