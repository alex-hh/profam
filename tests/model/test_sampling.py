def test_seqpos_start(default_model, proteingym_batch):
    # TODO: somehow use a dummy pipeline for sampling tests
    # n.b. completion seq pos is b, n, l
    assert (
        proteingym_batch["completion_seq_pos"][:, :, 1]
        == default_model.model.start_seq_pos
    ).all()
    assert (
        proteingym_batch["seq_pos"][:, 2] == default_model.model.start_seq_pos
    ).all()
