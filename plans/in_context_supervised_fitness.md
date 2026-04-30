# In-Context Supervised Fitness Prediction (ProFam-ICL)

Plan for extending ProFam-1 to do supervised in-context learning of continuous
fitness values, fine-tuned on ProteinGym DMS substitutions and evaluated on held-out
assay clusters.

## Implementation status (v1 scaffolding)

- [x] §1 token-stream design — implemented in
      [`profam/data/icl_constants.py`](../profam/data/icl_constants.py) and
      [`profam/data/builders/proteingym_icl.py`](../profam/data/builders/proteingym_icl.py)
- [x] §2 architectural changes — `LlamaICLLitModule` in
      [`profam/models/llama_icl.py`](../profam/models/llama_icl.py)
      (linear value lift, regression head, joint CE+MSE loss, backbone-lr split,
      Fourier featuriser available as ablation)
- [x] §3 data pipeline — `ProteinGymICLDataset` (k-shot sampling, within-context
      z-score, token budget, query-value leakage guard) and
      `ICLDocumentBatchCollator` in
      [`profam/data/collators.py`](../profam/data/collators.py)
- [x] §3.1 cluster-split helper —
      [`data_creation_scripts/cluster_proteingym_assays.py`](../data_creation_scripts/cluster_proteingym_assays.py)
      (script ready; needs an mmseqs2 install + ProteinGym data to actually run)
- [x] §4 training procedure — Hydra configs at
      [`configs/data/proteingym_icl.yaml`](../configs/data/proteingym_icl.yaml) and
      [`configs/experiment/icl_finetune.yaml`](../configs/experiment/icl_finetune.yaml)
      (single-GPU defaults, exposed `ce_loss_weight` / `mse_loss_weight` knobs)
- [x] §7 file-by-file roadmap items 1–6 + 8 — see test files
      [`tests/test_icl_dataset.py`](../tests/test_icl_dataset.py),
      [`tests/test_icl_forward.py`](../tests/test_icl_forward.py),
      [`tests/test_icl_causal_mask.py`](../tests/test_icl_causal_mask.py)
      (all 12 tests passing; cover token layout, value-leakage guard, gradient
      flow, single-batch overfit, causal-mask invariance, context permutation
      invariance at the first `[VAL]`)
- [ ] §5 / §6 evaluation script and the headline experiments E1–E5
      (`scripts/evaluate_icl.py` and the actual fine-tune runs) are deferred to
      a follow-up; the v1 scaffolding above is complete and unit-tested
- [ ] §7 item 7 (`scripts/evaluate_icl.py`) — not yet implemented; tracked for
      the follow-up

Quick-start (once mmseqs2 and the full ProteinGym data are on disk)::

    python data_creation_scripts/cluster_proteingym_assays.py \
        --gym-dir ../ProFam-atlas/ProteinGym \
        --output-dir data/proteingym_icl/splits

    python -m profam.train experiment=icl_finetune \
        ckpt_path=model_checkpoints/profam-1/checkpoints/last.ckpt

The goal is **transfer**: at test time the model receives a small set of
labelled (sequence, fitness) pairs from an assay it has never seen, plus an
optional unlabelled MSA, and must predict fitness for query sequences from the
same assay.

---

## 1. Token-stream design

The fine-tuned model operates on a single packed document. **v1 layout (no MSA
prefix):**

```
[start-of-document] [DOC_TYPE]                     # ProFam BOS + document-type
                                                    #   (existing tokens 47, 63 etc.)
x_1 [VAL] [VAL_SLOT] [SEP]                         # labeled example 1
x_2 [VAL] [VAL_SLOT] [SEP]                         # labeled example 2
...
x_k [VAL] [VAL_SLOT] [SEP]                         # labeled example k
x_q [VAL]                                          # query (predict here, no [SEP])
```

A future extension can prepend an unlabelled MSA prefix (`H_1 [SEP] … H_n [SEP]`)
before the first labelled example — the rest of the design is unchanged. v1 omits it
to keep the dataset and ablations simple.

- `x_i` is the variant amino-acid sequence (existing AA vocabulary).
- `[VAL]` is a discrete marker that signals "next slot carries a continuous fitness value
  for this example". The hidden state at `[VAL]` is what gets passed to the regression head.
- `[VAL_SLOT]` is a placeholder token whose token embedding is **replaced at runtime**
  by `e(y_i) = W_in · ϕ(y_i)`, where `ϕ` is either identity (linear lift) or sinusoidal/Fourier
  features. The ID is needed only to occupy a slot in `input_ids`; the embedded value is
  what the model actually sees.
- `[SEP]` plays the role of the `[X]` separator from the design notes (already in vocab,
  ID 49). Keeping `[SEP]` reused makes the new dataset blend cleanly with existing ProFam
  collators and packing.
- The query ends with `[VAL]` and **no** `[VAL_SLOT]` — by causal masking, the prediction
  at the query `[VAL]` can only depend on tokens preceding it, so the model can't peek at
  any value that follows.

### Repurposing existing unused special tokens

The shipped tokenizer reserves `[SP1]`…`[SP10]` (IDs 52–61) as unused specials.
Confirmed mapping for this work:

| New role        | Tokenizer token | ID  | Purpose                                                               |
|-----------------|-----------------|-----|-----------------------------------------------------------------------|
| `[VAL]`         | `[SP1]`         | 52  | marker; hidden state at this position is regressed to predict y       |
| `[VAL_SLOT]`    | `[SP2]`         | 53  | placeholder; its token embedding is overwritten with `e(y_i)`         |

Both are special tokens (excluded from sampling, etc.). No vocab-size change is needed;
the existing 68-token model accommodates them.

`[VAL]` and `[VAL_SLOT]` are kept as **distinct** tokens (no reuse) so the regression
head reads from a position whose embedding is *never* overridden, which keeps the
forward pass conceptually clean.

---

## 2. Architectural changes

We keep the underlying `LlamaForCausalLM` exactly as in ProFam-1 (no width/depth change,
no re-tokenization), and add three small modules:

1. **Value input map** `W_in: ℝ^{F} → ℝ^{d}`
   - Either `F = 1` (raw scalar, simple linear) or `F = 2K` (sinusoidal features
     `[sin(ω₁ y), cos(ω₁ y), …, sin(ω_K y), cos(ω_K y)]`).
   - Frequencies `ω_k` set log-uniformly over a learned or fixed range (TabPFN /
     NeRF-style). A reasonable default is `K = 64` with `ω_k = 2π · 10^{-2 + 4k/(K-1)}`,
     plus a final `Linear(2K, d)` projection. Defer to the simplest version first
     (linear) and ablate to Fourier.

2. **Value output head** `W_out: ℝ^{d} → ℝ^{1}`
   - Linear regression head applied to `h[VAL_pos]`. (Optional: a small MLP — start linear.)

3. **Two new token embeddings**
   - The token embedding rows for `[VAL]` and `[VAL_SLOT]`. These already exist in the
     embedding table because the tokenizer slots are present in vocab, but they are
     untrained from pretraining. They are randomly re-initialized at fine-tune start.

### Forward pass

A new `LightningModule`, `LlamaICLLitModule`, subclasses or extends `BaseFamilyLitModule`
and overrides `forward` to:

1. Accept extra batch tensors:
   - `value_slot_mask`: `(B, L)` bool — True where `[VAL_SLOT]` appears.
   - `values`: `(B, L)` float — the scalar y to embed at value-slot positions
     (zero / NaN-masked elsewhere).
   - `val_marker_mask`: `(B, L)` bool — True where `[VAL]` appears.
   - `predict_mask`: `(B, L)` bool — subset of `val_marker_mask` indicating positions
     at which to compute the MSE loss / read predictions (typically all of them
     during training; just the query at inference).
   - `target_values`: `(B, L)` float — the y to predict at each `[VAL]` position
     (only meaningful where `predict_mask` is True).

2. Build `inputs_embeds`:
   ```python
   embeds = self.model.get_input_embeddings()(input_ids)               # (B, L, d)
   if value_slot_mask.any():
       embedded_y = self.value_in_proj(featurize(values))               # (B, L, d)
       embeds = torch.where(value_slot_mask.unsqueeze(-1), embedded_y, embeds)
   ```

3. Pass `inputs_embeds=embeds` (instead of `input_ids`) into the underlying
   `LlamaForCausalLM`, with `output_hidden_states=True` and `labels=None` (we'll handle
   loss ourselves).

4. **Discrete LM loss** (existing CE next-token objective): keep it on AA / `[SEP]`
   positions, but mask out anything whose target is `[VAL]`, `[VAL_SLOT]`, or whose
   *previous* position is `[VAL_SLOT]` (because predicting the embedded scalar as a
   discrete token is meaningless). This uses the existing `labels` ignore-index plumbing.

5. **Regression loss**:
   ```python
   h = outputs.hidden_states[-1]                                        # (B, L, d)
   pred = self.value_out_head(h).squeeze(-1)                            # (B, L)
   reg_loss = F.mse_loss(pred[predict_mask], target_values[predict_mask])
   ```

6. Total loss: `loss = α · ce_loss + β · reg_loss` (both default to 1.0; ablate α=0
   to test pure regression vs. joint training).

### Position IDs and packing

ProFam currently sets `position_ids` per packed document via `compute_res_pos_in_doc`
(reset at each `[BOS]`). The ICL document is a *single* packed unit beginning with one
`[BOS]`, so position_ids count 0..L-1 monotonically across the whole ICL document — the
existing implementation already works for our case. We do **not** want to reset
position_ids at `[SEP]`, only at `[BOS]`, which matches current behaviour.

### Inference

At test time, build the prompt as above, run a single forward pass (no autoregressive
generation), and read `pred[predict_mask]`. KV-cache reuse across many query sequences
sharing the same context would mirror `_score_seqs_kv_cache` and is a natural
optimisation but is **not** required for v1.

---

## 3. Data pipeline

### 3.1 Train/test split (assay-cluster level)

- Source: `ProFam-atlas/ProteinGym/DMS_substitutions.csv` (217 assays).
- For each assay row, the WT sequence is in `target_seq`. Concatenate them all into a
  FASTA, run `mmseqs easy-cluster --min-seq-id 0.30 -c 0.8 --cov-mode 0` to obtain
  clusters.
- 80 / 20 random split **at the cluster level** (clusters held out as a unit so no two
  assays sharing >30% identity straddle the split).
- Persist the split as `data/proteingym_icl/splits/cluster_split.csv` with columns
  `DMS_id, cluster_rep, split ∈ {train, test}`.

New script: `data_creation_scripts/cluster_proteingym_assays.py`
- Inputs: `DMS_substitutions.csv`, mmseqs binary path.
- Outputs: cluster table + train/test assay lists + summary stats (cluster sizes etc.).

### 3.2 Fitness-score normalisation

The raw `DMS_score` columns are on heterogeneous scales (counts, log-fold-change,
fluorescence ratios, …). v1 uses **within-context z-score (Option A)**:

At training time, sample `k` labelled examples; compute `μ, σ` from those `k`
*labels*; use `(y_i − μ) / σ` for the embedded values, and `(y_q − μ) / σ` as the
regression target for the query. This matches inference exactly: at test time we
compute `μ, σ` from the same `k` labels we feed in. **Crucially the query's value
is excluded from the mean/std** so no information leaks.

Edge cases:
- When `σ` is tiny, divide by `max(σ, σ_floor)` (`σ_floor = 1e-3`) — no leakage,
  just numerical robustness.
- For `k = 1`, σ is undefined; fall back to `σ = 1` (lift becomes a centring shift).
  Since the v1 schedule mandates `k ≥ 32` (§3.3) this is only a guard for inference-
  time edge cases.

### 3.3 New dataset: `ProteinGymICLDataset`

Lives in `profam/data/builders/proteingym_icl.py`. Per `__getitem__`:

1. Sample an assay `a` from the current fold (train or test).
2. Read its CSV (`DMS_ProteinGym_substitutions/{DMS_filename}`).
3. Sample `k+1` variants without replacement: the first `k` are labelled in-context
   examples, the last is the query. `k` itself is sampled from a schedule biased
   toward larger contexts (see below).
4. Compute within-context z-score using only the `k` labelled values (the query value
   is excluded from mean/std but kept as the regression target).
5. Tokenize: produce `input_ids` plus the four auxiliary masks/values described above.
   Reuses `ProFamTokenizer` for AA stretches, splices in special tokens manually.
6. Total token budget: respect `max_tokens_per_example` (configurable; default 8192 to
   match existing). If `k+1` variants exceed the budget, reduce `k` (always keep the
   query) and recompute the z-score from the remaining labels.

**k-shot schedule.** v1 trains with `k ≥ 32`. Default sampler: uniform over
`{32, 64, 128, 256}`, gated by token budget. Rationale: small-k regressions are too
underdetermined to give the head a useful signal during fine-tuning, and the variance
of the within-context z-score (§3.2) becomes unreliable below ~30 samples. Smaller k's
can still be evaluated at test time even though they aren't in the training schedule
(causal-mask invariance gives them for free in a forward pass with k=32).

### 3.4 Collator

The existing `DocumentBatchCollator` in `profam/data/collators.py` handles `input_ids`,
`attention_mask`, `labels`, and arbitrary string fields. We extend it (or write
`ICLDocumentBatchCollator`) to also stack the four new tensors with proper padding
(`-100` semantics for `target_values` outside `predict_mask`, `False` for the masks,
`0.0` for `values`). Keep packing turned off in v1 — one ICL document per "sample" is
already large.

### 3.5 Validation dataset(s)

Two validation streams during training:
- The standard ProteinGym zero-shot CE / spearman loader (unchanged) to confirm we
  haven't damaged the base model.
- A **held-out ICL** loader that uses the *test-cluster* assay list with a fixed
  `k` (e.g. 32 or 64), reporting per-assay Spearman / Pearson and aggregate MSE in
  z-score space.

---

## 4. Training procedure

- **Init.** Load `model_checkpoints/profam-1/checkpoints/last.ckpt`. The new params
  (`value_in_proj`, `value_out_head`, the two re-initialised token embedding rows) are
  randomly initialised; everything else inherits from ProFam-1.
- **Optimizer.** AdamW; lr ~1e-4 to 3e-4 for the new params, lr ~1e-5 to 3e-5 for the
  pretrained backbone (param-group split). Warmup 1k steps, cosine decay to 10%.
  Weight decay 0.1, betas (0.9, 0.95). Match preprint values where possible.
- **Loss.** `total = α · CE_LM_loss + β · MSE_value_loss`. Both `α` and `β` are exposed
  as first-class config knobs (e.g. `model.ce_loss_weight`, `model.mse_loss_weight`)
  so they're easy to sweep without code changes. Reasonable starting defaults are
  α=1, β=1; α=0 (head-only) and β=0 (sanity-only) are valid sweep endpoints.
- **Value embedding.** v1 uses a **linear lift** `e(y) = W_in · y + b` with `W_in ∈
  ℝ^{d×1}`. Fourier featurisation is left as ablation E3.
- **Compute.** v1 targets **single-GPU** runs. The data config and trainer preset
  drop the existing 4-GPU DDP defaults: `trainer.devices: 1`, `trainer.strategy: auto`,
  `trainer.num_nodes: 1`, with `accumulate_grad_batches` raised so the effective batch
  size matches what we'd get on multi-GPU. Multi-GPU is straightforward to re-enable
  later but isn't blocking v1.
- **Precision.** Same bf16-true / FlashAttention-2 as base ProFam.
- **Batching.** No multi-document packing initially. Effective batch via
  `accumulate_grad_batches`. One ICL document per sample; clip max length to 8192.
- **Reproducibility.** Fix the cluster split, k-shot schedule seed, and assay
  sampling seed in config.

### Sanity checks before launching long runs

1. With α=0, β=1, train **on a single assay** with k=32. Loss should drop and
   per-assay Spearman should approach the optimum a small ridge regression hits.
2. Confirm that flipping the order of in-context examples doesn't change the prediction
   at the *first* `[VAL]` (causal-mask invariance check).
3. Confirm that filling all values with a constant produces a model that ignores
   `[VAL_SLOT]` and falls back to LM-only behaviour without nan / large updates to the
   backbone.

---

## 5. Evaluation

### 5.1 Metrics

Per held-out assay, compute over a large pool of (k, query) draws:
- **Spearman ρ** between predicted ŷ_q and true y_q (the headline metric used in
  ProteinGym leaderboards).
- **Pearson r**.
- **MSE / NLL** in z-score space.
- **Calibration**: a residual-vs-prediction plot per assay if we add prediction
  intervals later.

### 5.2 Shot-count sweep

Evaluate at `k ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128}`. Because of causal masking, a single
forward pass over `(x_1, …, x_k, x_q)` already produces all `k'`-shot predictions for
`k' < k` — but the per-shot statistics will be more stable if we *also* do independent
random draws of (context, query) for each k. Plot ρ-vs-k curves per assay and
aggregated.

### 5.3 Baselines

- **Zero-shot ProFam log-likelihood** (existing pipeline): the baseline we are trying
  to beat for k > 0.
- **Ridge / Gaussian-process regression on frozen ESM-2 or ProFam embeddings**, using
  the same k-shot context. This is the "fair" baseline because it has identical access
  to label information.
- **Constant prediction (mean of context labels).** A floor.
- **k-NN in embedding space.** Cheap reference.

### 5.4 Generalisation slices

- Aggregate by **selection assay type** (`coarse_selection_type` in
  `DMS_substitutions.csv`: OrganismalFitness, Stability, Binding, Activity, Expression).
  Question: does the model transfer better to some assay types than others?
- Aggregate by **MSA depth** (`MSA_Neff_L_category`).
- Aggregate by **distance-to-nearest-train-cluster** (extra diagnostic; recompute via
  mmseqs).

---

## 6. Proposed experiments (collaborator's recommendations + extensions)

E1. **Cluster-split ICL (headline).** 80/20 cluster split, train ICL fine-tune,
    evaluate Spearman vs. k on held-out clusters. Compare to baselines in §5.3.
E2. **Shot-count curve.** Above, but for fine resolution of k.
E3. **Value embedding ablation.** Linear `e(y) = W·y + b` vs. Fourier
    sin/cos features (TabPFN-style). Test whether non-linear scalar lifting matters.
E4. **Joint vs. head-only loss.** α=1 (joint CE+MSE) vs. α=0 (regression head only).
    Hypothesis: joint helps generalisation by keeping AA representation intact.
E5. **OOD-by-assay-type.** Hold out one assay type entirely (e.g. Stability) to test
    out-of-distribution-by-assay-type transfer.
E6. **(Stretch)** Compare to PoET / supervised ESM-2 baselines on the same split.

**Future extensions (not in v1, called out so the design accommodates them):**

EF1. **MSA prefix.** Add an unlabelled-homolog prefix `H_1 [SEP] … H_n [SEP]` before
     the labelled examples. Hypothesis: helps low-k regimes by anchoring family
     representation. Could train with 50/50 prefix dropout for robustness.
EF2. **Backbone freezing phase.** Freeze the Llama backbone for the first N steps
     while the new heads stabilise, then unfreeze. Useful if E1 shows backbone
     drift / forgetting on the zero-shot CE validation stream.
EF3. **Per-assay calibration normalisation.** Reserve a fixed calibration subset per
     assay to compute `μ, σ`. More robust variance estimate; trades label budget.

Recommended **first run** (v1): E1 + E2 with α=β=1, linear value embedding, no MSA
prefix, within-context z-score normalisation, k-shot schedule biased toward
`{32, 64, 128, 256}`, on all 217 assays via the cluster split. Everything else gates
on that being non-trivially above the zero-shot baseline.

---

## 7. Implementation roadmap (file-by-file)

1. **Cluster split**
   - `data_creation_scripts/cluster_proteingym_assays.py` — runs mmseqs, writes
     `data/proteingym_icl/splits/cluster_split.csv`.

2. **Tokenizer constants**
   - No JSON change required; just constants for `[VAL]` (=`[SP1]`) and `[VAL_SLOT]`
     (=`[SP2]`) added to `profam/constants.py` (or a new
     `profam/data/icl_constants.py`).

3. **Dataset**
   - `profam/data/builders/proteingym_icl.py` — new `ProteinGymICLDataset`. Largely
     mirrors `proteingym.py`'s MSA loading; new logic for k-shot sampling, normalisation,
     and producing the four extra tensors.

4. **Collator**
   - Extend `DocumentBatchCollator` to optionally pad/stack `value_slot_mask`,
     `values`, `val_marker_mask`, `predict_mask`, `target_values`. Or new
     `ICLDocumentBatchCollator` to keep blast radius small.

5. **Model**
   - `profam/models/llama_icl.py` — `LlamaICLLitModule(BaseFamilyLitModule)`.
     Adds `value_in_proj`, `value_out_head`, overrides `forward`, `training_step`,
     `validation_step` for ICL semantics. Reuses `BaseFamilyLitModule.configure_optimizers`
     plus param-group lr split.

6. **Configs**
   - `profam/configs/data/proteingym_icl.yaml` — train/val ICL data mixture.
   - `profam/configs/experiment/icl_finetune.yaml` — preset that loads ProFam-1
     checkpoint, swaps in `LlamaICLLitModule`, uses the ICL data config, and sets
     warmup / lr / batch size for fine-tuning.

7. **Evaluation script**
   - `scripts/evaluate_icl.py` — load fine-tuned checkpoint, iterate over held-out
     assays, sweep k, dump per-assay metrics + plots.

8. **Tests**
   - `tests/test_icl_dataset.py` — token stream shape; masks line up; z-score
     leakage check.
   - `tests/test_icl_forward.py` — random-init small model; gradients flow through
     `value_in_proj` and `value_out_head`; constant-y collapse test.
   - `tests/test_icl_causal_mask.py` — predicting at `[VAL]` is invariant to tokens
     after it.

---

## 8. Risks & mitigations

- **Backbone collapse from a bad value-embedding init.** Mitigated by lower lr on the
  backbone parameter group; if the zero-shot-CE validation stream regresses sharply,
  fall back to the head-only warmup phase (extension EF2).
- **Train/test contamination through homologous variants.** The cluster split is on
  assay WTs at 30% identity, which is the standard guard. Worth a sanity check that
  no test-cluster WT has >30% identity to any train-cluster WT after the split.
- **Label-distribution shift between train and test assays.** Even with within-context
  z-score normalisation, per-assay label distributions are non-Gaussian (heavy left
  tails for OrganismalFitness, bimodal for Stability/Binding). The Fourier value
  embedding ablation (E3) is the natural mitigation if linear lifting bottlenecks
  generalisation.
- **Tokeniser drift.** Re-initialising `[SP1]`/`[SP2]` embedding rows is fine; just
  confirm we don't accidentally re-init the entire embedding matrix when loading the
  ProFam-1 state dict.
- **CE loss interaction with replaced embeddings.** The backbone never sees a
  "predict the `[VAL_SLOT]` token" CE signal at the value-slot position (we mask it
  out), but it does see "predict `[SEP]` given the embedded scalar plus prefix". That's
  the intended training signal for the AA branch and should be fine.

---

## 9. Quick glossary (mapping to design notes)

| Notes notation | This plan          | Implementation notes                            |
|----------------|--------------------|-------------------------------------------------|
| `[X]`          | `[SEP]`            | reused (id 49)                                  |
| `H_i`          | unlabelled homolog | as in existing ProteinGym MSA pipeline          |
| `x_i`          | labelled variant   | tokenised by `ProFamTokenizer`                  |
| `[VAL]`        | `[VAL]` (`[SP1]`)  | hidden state read out by `value_out_head`        |
| `e(y_i)`       | `[VAL_SLOT]` (`[SP2]`) with embedding overridden by `value_in_proj(featurise(y_i))` | only at labelled examples |
| `[BOS]`        | `[start-of-document]` + `[DOC_TYPE]` | exactly as ProFam-1 |
