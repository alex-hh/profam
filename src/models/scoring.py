import torch

from src.data.objects import ProteinDocument
from src.evaluators.models import ScoringModelForEvaluation
from src.models.inference import ProFamInferenceModel


class ProFamScorer(ProFamInferenceModel, ScoringModelForEvaluation):
    def __init__(
        self, *args, bos_token: str = "", scoring_max_tokens: int = 2048, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scoring_max_tokens = scoring_max_tokens
        self.bos_token = bos_token  # n.b. inverse folding prompt builder will add separator so empty string in that case

    def score_completions(
        self,
        protein_document: ProteinDocument,
        completions: ProteinDocument,
        document_is_prompt: bool = False,
    ):
        if document_is_prompt:
            raise NotImplementedError("We need to infer original sequence length...")
        else:
            prompt = self.prompt_builder(protein_document, self.model.tokenizer)
        # encode prompt, then encode completions
        encoded = self.model.tokenizer.encode(
            prompt,
            document_token=self.document_token,
            padding="longest",
            add_final_sep=False,
        )
        # TODO: check this supports variable length completions
        # TODO: make sure we have correct bos token for interleaved scoring
        print("Encoding completions", len(completions.sequences))
        encoded_completions = self.model.tokenizer.encode_completions(
            sequences=completions.sequences,
            positions=completions.positions,
            bos_token=self.bos_token,
            backbone_coords=completions.backbone_coords,
            backbone_coords_masks=completions.backbone_coords_masks,
        )
        L = encoded_completions.input_ids.shape[-1]
        batch_size = (self.scoring_max_tokens - encoded.input_ids.shape[-1]) // L
        # TODO: check this works with variable length completions
        device = self.model.device
        kwargs = {}
        if self.model.embed_coords:
            kwargs["input_coords"] = (
                torch.from_numpy(encoded.data["coords"][None]).to(device).float()
            )
            kwargs["completion_coords"] = (
                torch.from_numpy(encoded_completions.data["coords"][None])
                .to(device)
                .float()
            )
        if self.model.tokenizer.use_seq_pos:
            kwargs["input_seq_pos"] = torch.from_numpy(
                encoded.data["seq_pos"][None]
            ).to(device)
            kwargs["completion_seq_pos"] = torch.from_numpy(
                encoded_completions.data["seq_pos"][None]
            ).to(device)

        print("Scoring", flush=True)
        scores = self.model.score_seqs(
            input_ids=torch.from_numpy(encoded.input_ids[None]).to(device),
            completion_ids=torch.from_numpy(encoded_completions.input_ids[None]).to(
                device
            ),
            use_cache=True,
            batch_size=batch_size,  # TODO: infer based on completion length?
            **kwargs,
        )
        return scores, prompt
