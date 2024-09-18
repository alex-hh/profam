def test_seqpos_start(test_model, proteingym_batch):
    for n, p in test_model.named_parameters():
        print(n, p.shape)
    # TODO: somehow use a dummy pipeline for sampling tests
    # n.b. completion seq pos is b, n, l
    assert (
        proteingym_batch["completion_seq_pos"][:, :, 1]
        == test_model.model.start_seq_pos
    ).all()
    assert (proteingym_batch["seq_pos"][:, 2] == test_model.model.start_seq_pos).all()
