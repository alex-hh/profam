import hydra
import pytest
import torch
from hydra import compose, initialize

from src.constants import BASEDIR
from src.data.objects import ProteinDocument

try:
    import flash_attn

    FLASH_ATTN_INSTALLED = True
except ImportError:
    FLASH_ATTN_INSTALLED = False


@pytest.mark.skipif(not FLASH_ATTN_INSTALLED, reason="Flash Attention not installed")
def test_sequence_packing_consistency(profam_tokenizer):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping Flash Attention test")

    # 1. Setup Model with Flash Attention
    # We use a single model instance and toggle pass_res_pos_in_doc_as_position_ids
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "model.scheduler_name=inverse_sqrt",
                "model.lr=1e-3",
                "model.config.hidden_size=128",
                "model.config.intermediate_size=512",
                "model.config.num_attention_heads=2",
                "model.config.num_hidden_layers=5",
                "model.config.num_key_value_heads=2",
                "model.config.max_position_embeddings=8192",
                "model.config.scoring_max_tokens=10240",
                "model.config.attn_implementation=flash_attention_2",
                "model.pass_res_pos_in_doc_as_position_ids=True",
            ],
        )

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model, tokenizer=profam_tokenizer)
    model.eval()
    model.to("cuda")
    # Flash Attention 2 usually requires fp16 or bf16
    model.to(torch.bfloat16)

    # 2. Construct Data
    # Two different sequences
    seq1 = "ACDEFGHIKLMNPQRSTVWY"  # Length 20
    seq2 = "ACDEFGHIKL"  # Length 10

    doc1 = ProteinDocument(sequences=[seq1])
    doc2 = ProteinDocument(sequences=[seq2])

    # Tokenize
    # Encoded: [BOS] [RAW] ...seq... [SEP]
    tok1 = profam_tokenizer.encode(doc1, document_token="[RAW]")
    tok2 = profam_tokenizer.encode(doc2, document_token="[RAW]")

    id1 = torch.tensor(tok1["input_ids"], device="cuda")  # Length 23
    id2 = torch.tensor(tok2["input_ids"], device="cuda")  # Length 13

    len1 = len(id1)
    len2 = len(id2)

    # --- Case A: Batched (B=2) ---
    # We pass no position_ids (pass_res_pos_in_doc_as_position_ids=False)
    # We need to pad the second sequence to match the first
    pad_token_id = profam_tokenizer.pad_token_id

    # Pad id2 to length of id1
    padding = torch.full((len1 - len2,), pad_token_id, device="cuda", dtype=torch.long)
    id2_padded = torch.cat([id2, padding])

    batch_input_ids = torch.stack([id1, id2_padded], dim=0)  # [2, 23]
    batch_attention_mask = (batch_input_ids != pad_token_id).long()

    # Disable manual position IDs for batched case (let model generate default ones)
    model.pass_res_pos_in_doc_as_position_ids = False

    with torch.no_grad():
        out_batch = model(
            input_ids=batch_input_ids, attention_mask=batch_attention_mask
        )
        logits_batch = out_batch.logits

    # --- Case B: Packed (B=1) ---
    # Concatenate sequences
    # Enable manual position IDs (pass_res_pos_in_doc_as_position_ids=True)

    packed_input_ids = torch.cat([id1, id2], dim=0).unsqueeze(0)
    packed_attention_mask = torch.ones_like(packed_input_ids)

    model.pass_res_pos_in_doc_as_position_ids = True

    with torch.no_grad():
        out_packed = model(
            input_ids=packed_input_ids, attention_mask=packed_attention_mask
        )
        logits_packed = out_packed.logits

    # --- Verify Equality ---
    # Compare Seq 1
    # Batch: [0, :len1]
    # Packed: [0, :len1]
    diff1 = (logits_batch[0, :len1] - logits_packed[0, :len1]).abs().max()

    # Compare Seq 2
    # Batch: [1, :len2]
    # Packed: [0, len1:len1+len2]
    diff2 = (logits_batch[1, :len2] - logits_packed[0, len1 : len1 + len2]).abs().max()

    # Tolerances might need to be looser for float16/bfloat16
    assert diff1 < 1e-2, f"Sequence 1 outputs differ significantly: {diff1.item()}"
    assert diff2 < 1e-2, f"Sequence 2 outputs differ significantly: {diff2.item()}"

    # --- Case C: Packed WITHOUT corrected position IDs ---
    # This should yield DIFFERENT results for the second sequence (due to RoPE)

    model.pass_res_pos_in_doc_as_position_ids = False

    with torch.no_grad():
        out_packed_wrong = model(
            input_ids=packed_input_ids, attention_mask=packed_attention_mask
        )
        logits_packed_wrong = out_packed_wrong.logits

    # Compare Seq 2 with original batched Seq 2
    diff2_wrong = (
        (logits_batch[1, :len2] - logits_packed_wrong[0, len1 : len1 + len2])
        .abs()
        .max()
    )

    assert (
        diff2_wrong > 1e-2
    ), f"Sequence 2 outputs should differ when pos IDs are wrong, but diff was {diff2_wrong.item()}"
