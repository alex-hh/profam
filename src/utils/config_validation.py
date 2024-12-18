from omegaconf import DictConfig


def check_config(cfg: DictConfig):
    assert cfg.model.embed_residue_index == cfg.tokenizer.embed_residue_index
