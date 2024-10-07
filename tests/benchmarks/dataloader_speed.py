import os
import time

import hydra
from hydra import compose, initialize_config_dir

from src.constants import BASEDIR

with initialize_config_dir(os.path.join(BASEDIR, "configs")):
    cfg = compose(
        config_name="train",
        overrides=[
            "experiment=foldseek_inverse_folding",
            "data=foldseek_interleaved",
            f"paths.root_dir={BASEDIR}",
        ],
    )
    tokenizer_cfg = compose(
        config_name="tokenizer/profam",
    )


tokenizer = hydra.utils.instantiate(tokenizer_cfg.tokenizer)
dm = hydra.utils.instantiate(cfg.data, tokenizer=tokenizer, _convert_="partial")
dm.setup()
train_loader = dm.train_dataloader()

t_prev = time.time()
for ix, batch in enumerate(train_loader):
    print(ix, time.time() - t_prev)
    t_prev = time.time()
