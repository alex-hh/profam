import argparse
import glob
import os
import re
import shutil
from typing import List, Tuple

from src.evaluators.esmfold import ESMFoldSamplingEvaluator
from src.pipelines.pipeline import GenerationsEvaluatorPipeline
from src.sequence import fasta
from src.data.objects import ProteinDocument
from src import constants

"""
Created by Jude Wells 2025-09-12
Wrapper script that runs the ESMFold evaluation pipeline 
on pre-generated fasta files.
"""


def map_generated_to_prompt_path(gen_fasta_path: str, pattern: str, replacement: str) -> str:
    """Map a generated FASTA path to the corresponding prompt FASTA path via regex.

    Args:
        gen_fasta_path: Path to generated sequences FASTA file
        pattern: Regex pattern to match the generated path
        replacement: Replacement string used in re.sub to produce the prompt path

    Returns:
        Prompt FASTA path as string
    """
    fname = os.path.basename(gen_fasta_path).replace("_generated.fasta", ".fasta")
    if "foldseek" in gen_fasta_path:
        ds_name = "foldseek"
    elif "funfams" in gen_fasta_path:
        ds_name = "funfams"
    elif "pfam" in gen_fasta_path:
        ds_name = "pfam"
    else:
        raise ValueError(f"Unknown dataset name in {gen_fasta_path}")
    if "_val" in gen_fasta_path:
        split = "val"
    elif "_test" in gen_fasta_path:
        split = "test"
    else:
        raise ValueError(f"Unknown split in {gen_fasta_path}")
    path = f"../data/val_test_v2_fastas/{ds_name}/{split}/{fname}"
    return path


def infer_instance_id(gen_fasta_path: str, model_name: str, strategy: str = "auto") -> str:
    """Infer instance/family identifier from generated FASTA path.

    Strategies:
      - auto: if parent dir equals model_name, use parent of parent; else use parent
      - parent: basename of directory containing the FASTA file
      - stem: file stem (basename without extension)
    """
    parts = gen_fasta_path.split("/")
    fam_name = parts[-1].split(".")[0]
    ds_name = parts[-3]
    if "ensemble" in gen_fasta_path:
        method = "ensemble"
    elif "single" in gen_fasta_path:
        method = "single"
    else:
        method = ""
    return f"{ds_name}_{method}_{fam_name}"


def ensure_pipeline_layout(
    base_directory: str,
    pipeline_id: str,
    instance_id: str,
    model_name: str,
) -> Tuple[str, str]:
    """Ensure the directory structure for generations and prompts exists.

    Returns:
        (generations_dir, prompts_dir)
    """
    base_dir = os.path.join(base_directory, pipeline_id)
    generations_dir = os.path.join(base_dir, "generations", instance_id, model_name)
    prompts_dir = os.path.join(base_dir, "prompts", instance_id, model_name)
    os.makedirs(generations_dir, exist_ok=True)
    os.makedirs(prompts_dir, exist_ok=True)
    return generations_dir, prompts_dir


def write_prompt_json_from_fasta(prompt_fasta_path: str, identifier: str, output_json_path: str) -> None:
    names, sequences = fasta.read_fasta(prompt_fasta_path)
    sequences = [s.replace("-", "") for s in sequences]
    doc = ProteinDocument(
        sequences=sequences,
        accessions=names,
        identifier=identifier,
    )
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    doc.to_json(output_json_path)


class DummySampler:
    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = device

    def to(self, device: str):
        self.device = device


class FSGenerationsPipeline(GenerationsEvaluatorPipeline):
    """Pipeline that enumerates instances from filesystem-provided FASTAs.

    It loads per-instance prompt ProteinDocuments directly from the prompt FASTA paths
    provided via the mapping function.
    """

    def __init__(
        self,
        instance_ids: List[str],
        gen_to_prompt_map: dict,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._instance_ids = instance_ids
        self._gen_to_prompt_map = gen_to_prompt_map

    def instance_ids(self):
        return self._instance_ids

    def load_protein_document(self, instance_id: str) -> ProteinDocument:
        prompt_fasta_path = self._gen_to_prompt_map[instance_id]
        names, sequences = fasta.read_fasta(prompt_fasta_path)
        sequences = [s.replace("-", "") for s in sequences]
        return ProteinDocument(
            sequences=sequences,
            accessions=names,
            identifier=instance_id,
        )

    def get_instance_summary(self, instance_id: str):
        return {"num_sequences": 0.0}


def main():
    parser = argparse.ArgumentParser(description="Evaluate generations with ESMFold and existing pipeline code.")
    parser.add_argument("--generated-glob", required=True, help="Glob pattern for generated FASTA files (one per family).")
    parser.add_argument("--model-name", required=True, help="Model/sampler name (e.g., ProFam, PoET).")
    parser.add_argument("--pipeline-id", required=True, help="Pipeline identifier for output directory names.")
    parser.add_argument("--benchmark-dir", default=None, help="Base directory to store pipeline outputs. Defaults to constants.BENCHMARK_RESULTS_DIR.")
    parser.add_argument("--prompt-map-regex", required=True, help="Regex pattern applied to generated path to obtain prompt FASTA path.")
    parser.add_argument("--prompt-map-replacement", required=True, help="Regex replacement to produce prompt FASTA path.")
    parser.add_argument("--instance-id-strategy", choices=["auto", "parent", "stem"], default="auto")
    parser.add_argument("--device", default="cuda", help="Device string for ESMFold (e.g., cuda, cuda:0, cpu).")
    parser.add_argument("--esmfold-max-length", type=int, default=512, help="Max sequence length for ESMFold.")
    parser.add_argument("--half-precision", action="store_true", help="Use half precision for ESMFold.")
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copying FASTA files into pipeline directory.")
    args = parser.parse_args()

    gen_fasta_paths = sorted(glob.glob(args.generated_glob))
    if len(gen_fasta_paths) == 0:
        raise FileNotFoundError(f"No generated FASTA files found for glob: {args.generated_glob}")

    instance_ids = []
    instance_to_prompt_fasta = {}

    target_base_dir = args.benchmark_dir or constants.BENCHMARK_RESULTS_DIR

    for gen_fa in gen_fasta_paths:
        instance_id = infer_instance_id(gen_fa, args.model_name, strategy=args.instance_id_strategy)
        prompt_fa = map_generated_to_prompt_path(gen_fa, args.prompt_map_regex, args.prompt_map_replacement)
        if not os.path.isfile(prompt_fa):
            print(f"Prompt FASTA not found for generated file {gen_fa}: {prompt_fa}")
            continue

        generations_dir, prompts_dir = ensure_pipeline_layout(
            target_base_dir, args.pipeline_id, instance_id, args.model_name
        )

        dest_gen_fa = os.path.join(generations_dir, "sequences.fa")
        if os.path.islink(dest_gen_fa) or os.path.isfile(dest_gen_fa):
            os.remove(dest_gen_fa)
        if args.symlink:
            os.symlink(os.path.abspath(gen_fa), dest_gen_fa)
        else:
            shutil.copyfile(gen_fa, dest_gen_fa)

        prompt_json = os.path.join(prompts_dir, "prompt.json")
        write_prompt_json_from_fasta(prompt_fa, identifier=instance_id, output_json_path=prompt_json)

        instance_ids.append(instance_id)
        instance_to_prompt_fasta[instance_id] = prompt_fa

    pipeline = FSGenerationsPipeline(
        instance_ids=instance_ids,
        gen_to_prompt_map=instance_to_prompt_fasta,
        num_generations=0,
        pipeline_id=args.pipeline_id,
        benchmark_directory=args.benchmark_dir,
        save_results_to_file=True,
    )

    evaluator = ESMFoldSamplingEvaluator(
        name="esmfold",
        save_structures=True,
        use_precomputed_reference_structures=False,
        half_precision=args.half_precision,
        max_length=args.esmfold_max_length,
        verbose=True,
    )

    dummy_sampler = DummySampler(name=args.model_name, device=args.device)

    pipeline.run(
        sampler=dummy_sampler,
        evaluators=[evaluator],
        verbose=True,
        rerun_sampler=False,
        rerun_evaluator=True,
        sampling_only=False,
        offload_sampler=False,
        device=args.device,
        disable_tqdm=False,
    )


if __name__ == "__main__":
    main()


