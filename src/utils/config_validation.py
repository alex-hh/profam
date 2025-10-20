from omegaconf import DictConfig


def check_config(cfg: DictConfig):
    return
    max_train_doc_len = 0
    for dataset in cfg.data.dataset_builders:
        if dataset not in cfg.data.val_dataset_batch_sizes:
            n_toks = cfg.data.dataset_builders[
                dataset
            ].preprocessor.cfg.max_tokens_per_example
            max_train_doc_len = max(max_train_doc_len, n_toks)

    if "proteingym" in cfg.data.dataset_builders:
        pg_max_tokens = cfg.data.dataset_builders.proteingym.max_tokens_per_example
        assert cfg.trainer.tokens_per_document >= pg_max_tokens
        if (
            cfg.data.dataset_builders.proteingym.max_context_seqs is None
            or cfg.data.dataset_builders.proteingym.max_context_seqs > 0
        ):
            assert max_train_doc_len >= pg_max_tokens
    if "max_position_embeddings" in cfg.model.config:
        assert cfg.model.config.max_position_embeddings >= max_train_doc_len
        if "proteingym" in cfg.data.dataset_builders:
            assert cfg.model.config.max_position_embeddings >= pg_max_tokens

    if "pack_to_max_tokens" in cfg.data and cfg.data.pack_to_max_tokens:
        assert (
            cfg.data.batch_size == 1
        ), "batch_size must be 1 when packing to max tokens"

        assert (
            "pass_res_pos_in_doc_as_position_ids" in cfg.model
            and cfg.model.pass_res_pos_in_doc_as_position_ids
        ), "sequence packing (pack_to_max_tokens=True) requires position_ids in forward"

        assert (
            "attn_implementation" in cfg.model.config
            and cfg.model.config.attn_implementation == "flash_attention_2"
        ), "sequence packing (pack_to_max_tokens=True) requires flash_attention_2"

        if (
            "tokens_per_document" in cfg.trainer
            and cfg.trainer.tokens_per_document is not None
        ):
            if (
                abs(cfg.trainer.tokens_per_document - cfg.data.pack_to_max_tokens)
                > 1000
            ):
                print(
                    f"Warning: tokens_per_document ({cfg.trainer.tokens_per_document})"
                    f"is significantly different from cfg.data.pack_to_max_tokens"
                    f"({cfg.data.pack_to_max_tokens})."
                    f"This may cause incorrect calculation of accumulate_grad_batches."
                )

    if "config" in cfg.model:
        if (
            "hidden_size" in cfg.model.config
            and "intermediate_size" in cfg.model.config
        ):
            # Typically intermediate_size should be ~4x hidden_size for LLaMA models
            assert (
                cfg.model.config.intermediate_size >= cfg.model.config.hidden_size
            ), "intermediate_size should be larger than hidden_size"

            ratio = cfg.model.config.intermediate_size / cfg.model.config.hidden_size
            if ratio < 2 or ratio > 8:
                print(
                    f"Warning: intermediate_size ({cfg.model.config.intermediate_size}) to "
                    f"hidden_size ({cfg.model.config.hidden_size}) ratio is {ratio:.1f}. "
                    f"Typical values are between 2 and 8."
                )

    # Check rope_theta and position embeddings consistency
    if (
        "rope_theta" in cfg.model.config
        and "max_position_embeddings" in cfg.model.config
    ):
        if (
            cfg.model.config.rope_theta < 10000
            and cfg.model.config.max_position_embeddings > 16384
        ):
            print(
                f"Warning: Small rope_theta ({cfg.model.config.rope_theta}) with large "
                f"max_position_embeddings ({cfg.model.config.max_position_embeddings}) may cause "
                f"poor performance at long sequence lengths"
            )
