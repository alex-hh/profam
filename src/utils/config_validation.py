from omegaconf import DictConfig


def check_config(cfg: DictConfig):
    assert cfg.model.embed_residue_index == cfg.tokenizer.embed_residue_index
    if "pack_to_max_tokens" in cfg.data and cfg.data.pack_to_max_tokens:
        assert (
            cfg.data.batch_size == 1
        ), "batch_size must be 1 when packing to max tokens"
        assert cfg.model.pass_res_pos_in_doc_as_position_ids or (
            cfg.model.pass_res_pos_in_seq_as_position_ids
        ), "sequence packing (pack_to_max_tokens=True) requires position_ids to be in forward pass"
