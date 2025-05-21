def test_seqpos_start(test_model, proteingym_batch):
    for n, p in test_model.named_parameters():
        print(n, p.shape)
    # TODO: somehow use a dummy pipeline for sampling tests
    # n.b. completion seq pos is b, n, l
    assert (
        proteingym_batch["completion_residue_index"][:, :, 1]
        == test_model.model.start_residue_index
    ).all()
    # assumption here is that first residue index represents first
    # position in the msa - not necessarily true if we use MSA pos
    # and do not include the WT
    assert (
        proteingym_batch["residue_index"][:, 2] == test_model.model.start_residue_index
    ).all()
