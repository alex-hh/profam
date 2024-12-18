from omegaconf import DictConfig

def check_config(cfg: DictConfig):
    assert (
        cfg.model.embed_residue_index == cfg.tokenizer.embed_residue_index, 
        "Config Contradiction: model.embed_residue_index and tokenizer.embed_residue_index must be the same"
    )
