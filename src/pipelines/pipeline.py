import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tqdm
from Bio import AlignIO
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src import constants
from src.data.objects import ProteinDocument
from src.evaluators.base import SamplingEvaluator
from src.sequence import fasta
from src.utils.utils import maybe_print


def load_named_pipeline(pipeline_name: str, overrides: Optional[List[str]] = None):
    with initialize_config_dir(
        os.path.join(constants.BASEDIR, "configs/pipeline"), version_base="1.3"
    ):
        pipeline_cfg = compose(config_name=pipeline_name, overrides=overrides)
    return instantiate(pipeline_cfg)


class BaseEvaluatorPipeline:

    """A validation pipeline handles loading of documents, running of models and storing of results.

    The pipeline basically wraps around an evaluator which determines the logic of input
    generation and metric computation.

    If multiple sets of metrics should be run on a single set of generations, the evaluator needs
    to be written appropriately.

    # TODO: separate results df for each evaluator - store in dict maybe?
    """

    def __init__(
        self,
        pipeline_id: str,
        benchmark_directory: str = None,
        save_results_to_file: bool = True,
    ):
        self.pipeline_id = pipeline_id
        self.pipeline_directory = os.path.join(
            benchmark_directory or constants.BENCHMARK_RESULTS_DIR,
            self.pipeline_id,
        )
        self.save_results_to_file = save_results_to_file
        self.reset()

    def instance_ids(self):
        raise NotImplementedError()

    def reset(self):
        self.results_dfs = {}

    def load_results(self, evaluator_name) -> pd.DataFrame:
        """Load results dataframe from local disk location.

        TODO: we really want different results files for different evaluators,
        so this should happen somewhere else.
        """
        results_path = os.path.join(
            self.pipeline_directory, evaluator_name, "results.csv"
        )
        if self.save_results_to_file and os.path.exists(results_path):
            self.results_dfs[evaluator_name] = pd.read_csv(results_path)
        else:
            self.results_dfs[evaluator_name] = pd.DataFrame(
                columns=["evaluator", "sampler", "instance"]
            )
        self.results_dfs[evaluator_name].set_index(
            ["evaluator", "sampler", "instance"], inplace=True
        )

    def has_result(self, evaluator_name: str, instance_id: str, model_id: str) -> bool:
        """Check if validation, instance, model combo is present in results df index."""
        return (evaluator_name, model_id, instance_id) in self.results_dfs[
            evaluator_name
        ].index

    def add_result(
        self,
        evaluator_name: str,
        instance_id: str,
        model_id: str,
        result: Dict[str, float],
    ) -> None:
        """Add a result to the results dataframe."""
        # drop any existing result for this instance, validation, model combo
        # then concatenate a new row to the df
        if evaluator_name not in self.results_dfs:
            self.results_dfs[evaluator_name] = pd.DataFrame(
                columns=["evaluator", "sampler", "instance"]
            ).set_index(["evaluator", "sampler", "instance"], inplace=True)
        self.results_dfs[evaluator_name].drop(
            index=(evaluator_name, model_id, instance_id), inplace=True, errors="ignore"
        )
        self.results_dfs[evaluator_name] = pd.concat(
            [
                self.results_dfs[evaluator_name],
                pd.DataFrame([result]).set_index(["evaluator", "sampler", "instance"]),
            ]
        )
        csv_save_path = os.path.join(
            self.pipeline_directory, evaluator_name, "results.csv"
        )
        os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
        self.results_dfs[evaluator_name].to_csv(csv_save_path, index=True)

    def save_results(self) -> None:
        """Save results dataframe to local disk location."""
        if self.save_results_to_file:
            for evaluator_name, results_df in self.results_dfs.items():
                results_path = os.path.join(
                    self.pipeline_directory, evaluator_name, "results.csv"
                )
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                results_df.to_csv(results_path, index=True)

    def make_summary(self):
        summaries = []
        for instance_id in self.instance_ids():
            summary = self.get_instance_summary(instance_id)
            summary["instance_id"] = instance_id
            summaries.append(summary)
        return pd.DataFrame.from_records(summaries)

    def get_instance_summary(self, instance_id: str) -> Dict[str, float]:
        raise NotImplementedError()

    def load_protein_document(self, instance_id: str) -> ProteinDocument:
        raise NotImplementedError()


class GenerationsEvaluatorPipeline(BaseEvaluatorPipeline):

    """Validation that computes metrics given a set of generated sequences."""

    def __init__(
        self,
        num_generations: int,
        pipeline_id: str,
        benchmark_directory: str = None,
        save_results_to_file: bool = True,
        max_tokens: int = None,
        max_generated_length: int = None,
    ):
        self.max_tokens = max_tokens
        self.num_generations = num_generations
        self.max_generated_length = max_generated_length
        self.generations = defaultdict(dict)
        self.prompts = defaultdict(dict)
        print(
            f"Initialised pipeline ID {pipeline_id} num generations {num_generations}"
        )
        super().__init__(
            pipeline_id,
            benchmark_directory=benchmark_directory,
            save_results_to_file=save_results_to_file,
        )

    def has_generations(self, instance_id: str, model_id: str) -> bool:
        # TODO: check prompt as well
        if not self.save_results_to_file:
            return (
                model_id in self.generations
                and instance_id in self.generations[model_id]
            )
        else:
            output_path = os.path.join(
                self.pipeline_directory,
                "generations",
                instance_id,
                model_id,
                "sequences.fa",
            )
            prompt_output_path = os.path.join(
                self.pipeline_directory, "prompts", instance_id, model_id, "prompt.json"
            )
            retval = os.path.isfile(output_path) and prompt_output_path
            return retval

    def has_all_generations(self, model_id: str) -> None:
        return all(
            [
                self.has_generations(instance_id, model_id)
                for instance_id in self.instance_ids()
            ]
        )

    def validate_configs(self, sampler_config, evaluator_config):
        # save configs to appropriate directory.
        # if rerunning, we check that the configs match, otherwise we raise
        # an exception. (TODO: allow overriding with an ignore_config_mismatch flag).
        raise NotImplementedError()

    def run_evaluator_on_instance(
        self,
        sampler_name: str,
        instance_id: str,
        evaluator: SamplingEvaluator,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        rerun_evaluator: bool = False,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        generated_sequences = self.load_generations(instance_id, sampler_name)
        if rerun_evaluator or not self.has_result(
            evaluator.name, instance_id, sampler_name
        ):
            output_dir = os.path.join(
                self.pipeline_directory, evaluator.name, instance_id, sampler_name
            )
            if rerun_evaluator:
                if os.path.isdir(output_dir):
                    shutil.rmtree(output_dir)

            metrics = evaluator.evaluate_samples(
                prompt=prompt,
                protein_document=protein_document,
                samples=generated_sequences,
                output_dir=output_dir,
                device=device,
            )

            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            if verbose:
                print(f"Instance {instance_id} {evaluator.name} metrics: {metrics_str}")

            metrics.update(self.get_instance_summary(instance_id))
            metrics["sampler"] = sampler_name
            metrics["instance"] = instance_id
            metrics["evaluator"] = evaluator.name
            metrics["first_5_generated_sequences"] = "|".join(generated_sequences[:5])
            metrics["first_5_prompt_sequences"] = "|".join(prompt.sequences[:5])
            self.add_result(evaluator.name, instance_id, sampler_name, metrics)
            if max([len(s) for s in prompt.sequences]) == 0:
                return
            self.analyze_sequence_similarity_and_logos(
                instance_id,
                sampler_name,
                prompt.sequences,
                generated_sequences,
            )

    def save_generations(self, instance_id, model_name, sequences: List[str]) -> None:
        if self.save_results_to_file:
            outputs_dir = os.path.join(
                self.pipeline_directory, "generations", instance_id, model_name
            )
            os.makedirs(outputs_dir, exist_ok=True)
            fasta.output_fasta(
                [f"seq{i}" for i in range(len(sequences))],
                sequences,
                os.path.join(outputs_dir, "sequences.fa"),
            )
        else:
            self.generations[model_name][instance_id] = sequences

    def save_prompt(self, instance_id, model_name, prompt: str) -> None:
        if self.save_results_to_file:
            outputs_dir = os.path.join(
                self.pipeline_directory, "prompts", instance_id, model_name
            )
            os.makedirs(outputs_dir, exist_ok=True)
            prompt.to_json(os.path.join(outputs_dir, "prompt.json"))
        else:
            self.prompts[model_name][instance_id] = prompt

    def load_generations(self, instance_id: str, sampler_name: str) -> List[str]:
        if self.save_results_to_file:
            outputs_dir = os.path.join(
                self.pipeline_directory, "generations", instance_id, sampler_name
            )
            fasta_file = os.path.join(outputs_dir, "sequences.fa")
            _, sequences = fasta.read_fasta(fasta_file)
            sequences = [s.replace("-", "") for s in sequences]
            return sequences
        else:
            sequences = self.generations[sampler_name][instance_id]
            return sequences

    def load_prompt(self, instance_id: str, sampler_name: str) -> ProteinDocument:
        if self.save_results_to_file:
            outputs_dir = os.path.join(
                self.pipeline_directory, "prompts", instance_id, sampler_name
            )
            prompt_file = os.path.join(outputs_dir, "prompt.json")
            prompt = ProteinDocument.from_json(prompt_file)
            return prompt
        else:
            prompt = self.prompts[sampler_name][instance_id]
            return prompt

    def run(
        self,
        sampler,
        evaluators: Union[List[SamplingEvaluator], SamplingEvaluator],
        verbose: bool = True,
        rerun_sampler: bool = False,
        rerun_evaluator: bool = True,
        sampling_only: bool = False,
        offload_sampler: bool = False,
        device: Optional[str] = None,
        disable_tqdm: bool = False,
    ):
        if not isinstance(evaluators, List):
            assert isinstance(evaluators, SamplingEvaluator)
            evaluators = [evaluators]
        for evaluator in evaluators:
            self.load_results(evaluator.name)

        instance_ids = self.instance_ids()
        if rerun_sampler:
            rerun_evaluator = True

        for instance_id in tqdm.tqdm(instance_ids, disable=verbose or disable_tqdm):
            maybe_print(
                "Running evaluation pipeline for instance", instance_id, verbose=verbose
            )
            protein_document = self.load_protein_document(instance_id)
            if rerun_sampler or not self.has_generations(instance_id, sampler.name):
                maybe_print(
                    f"Running generations for instance: {instance_id}",
                    verbose=verbose,
                    flush=True,
                )
                # TODO: it's a bit awkward that this is a method on evaluator...
                # it should produce the same output regardless of the evaluator
                generations, scores, prompt = sampler.sample_seqs(
                    protein_document=protein_document,
                    num_samples=self.num_generations,
                    max_tokens=self.max_tokens,
                    max_generated_length=self.max_generated_length,
                )
                self.save_generations(instance_id, sampler.name, generations)
                self.save_prompt(instance_id, sampler.name, prompt)
            else:
                maybe_print(
                    f"Loading generations for instance: {instance_id}",
                )
                generations = self.load_generations(instance_id, sampler.name)
                prompt = self.load_prompt(instance_id, sampler.name)

            sampler_device = sampler.device
            if not sampling_only:
                if offload_sampler:
                    sampler.to(
                        "cpu"
                    )  # offload memory to CPU. TODO: consider avoiding all this device switching
                for evaluator in evaluators:
                    try:
                        self.run_evaluator_on_instance(
                            sampler.name,
                            instance_id=instance_id,
                            evaluator=evaluator,
                            prompt=prompt,
                            protein_document=protein_document,
                            rerun_evaluator=rerun_evaluator,
                            device=device,
                        )
                    except Exception as e:
                        print("Failed to run validation on instance", instance_id)
                        raise e
                if offload_sampler:
                    sampler.to(sampler_device)  # move back to original device

        if sampling_only:
            return

        # TODO format to limit decimal places
        outputs = {}
        for evaluator in evaluators:
            sampler_results = self.results_dfs[evaluator.name].loc[
                (evaluator.name, sampler.name)
            ]
            avg_metrics = sampler_results.select_dtypes(include=np.number).mean()
            avg_metrics_str = ", ".join(
                [f"{k}: {v:.3f}" for k, v in avg_metrics.items()]
            )
            maybe_print(
                f"Validation `{evaluator.name}` model {sampler.name} average metrics: "
                f"{avg_metrics_str} ({len(sampler_results)} instances)",
                verbose=verbose,
            )
            outputs[evaluator.name] = sampler_results

        self.save_results()
        return outputs

    def analyze_sequence_similarity_and_logos(
        self,
        instance_id: str,
        sampler_name: str,
        prompt_sequences: List[str],
        sampled_sequences: List[str],
        output_dir: Optional[str] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Analyze sequence similarity between prompt and sampled sequences, and create logos.

        This method:
        1. Aligns both prompt and sampled sequences using MAFFT
        2. Computes maximum sequence similarity for each sampled sequence with prompt sequences
        3. Computes maximum sequence similarity between sampled sequences
        4. Creates logos for both prompt and sampled sequences

        Args:
            instance_id: The instance ID to analyze
            sampler_name: The sampler name
            output_dir: Directory to save logos (defaults to pipeline directory)
            threads: Number of threads for MAFFT alignment
            verbose: Whether to print verbose output

        Returns:
            Dictionary containing similarity statistics
        """
        # Load sequences

        if verbose:
            print(
                f"Analyzing {len(prompt_sequences)} prompt sequences and {len(sampled_sequences)} sampled sequences"
            )

        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(
                self.pipeline_directory, "sequence_analysis", instance_id, sampler_name
            )
        os.makedirs(output_dir, exist_ok=True)

        # Helper functions from the script
        def write_fasta(sequences, accessions, fasta_path):
            """Write sequences to FASTA file."""
            with open(fasta_path, "w") as f:
                for acc, seq in zip(accessions, sequences):
                    f.write(f">{acc}\n{seq}\n")

        def run_alignment_with_mafft(fasta_input, fasta_output, threads=1):
            """Run MAFFT alignment."""
            cmd = ["mafft", "--thread", str(threads), "--auto", fasta_input]
            if verbose:
                print(f"Running: {' '.join(cmd)}")
            with open(fasta_output, "w") as fout:
                subprocess.run(cmd, check=True, stdout=fout)

        def create_logo_from_fasta(alignment_fasta, output_logo):
            """Create sequence logo from aligned FASTA."""
            try:
                import logomaker
            except ImportError:
                print("logomaker not found, skipping logo creation")
                return
            alignment = AlignIO.read(alignment_fasta, "fasta")
            sequences = [str(record.seq) for record in alignment]

            # Build logomaker counts matrix
            counts_matrix = logomaker.alignment_to_matrix(sequences)
            logo = logomaker.Logo(
                counts_matrix,
                color_scheme="weblogo_protein",
                width=0.8,
                figsize=(60, 2.5),
            )
            logo.fig.savefig(output_logo)
            if verbose:
                print(f"Sequence logo saved as {output_logo}")

        def compute_sequence_similarity(seq1: str, seq2: str) -> float:
            """Compute sequence similarity between two sequences."""
            seq_len = max(len(seq1.replace("-", "")), len(seq2.replace("-", "")))
            matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != "-")
            return matches / seq_len if seq_len > 0 else 0.0

        prompt_sequences = [s.replace("[SEP]", "") for s in prompt_sequences]
        # Create temporary files for alignments
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare sequences with accessions
            prompt_accessions = [f"prompt_{i}" for i in range(len(prompt_sequences))]
            sampled_accessions = [f"sampled_{i}" for i in range(len(sampled_sequences))]

            # Write separate FASTA files for individual logos
            prompt_fasta = os.path.join(temp_dir, "prompt.fasta")
            sampled_fasta = os.path.join(temp_dir, "sampled.fasta")
            write_fasta(prompt_sequences, prompt_accessions, prompt_fasta)
            write_fasta(sampled_sequences, sampled_accessions, sampled_fasta)

            # Create combined FASTA file for similarity analysis
            combined_sequences = prompt_sequences + sampled_sequences
            combined_accessions = prompt_accessions + sampled_accessions
            combined_fasta = os.path.join(temp_dir, "combined.fasta")
            write_fasta(combined_sequences, combined_accessions, combined_fasta)

            # Run alignments
            prompt_aligned = os.path.join(temp_dir, "prompt_aligned.fasta")
            sampled_aligned = os.path.join(temp_dir, "sampled_aligned.fasta")
            combined_aligned = os.path.join(temp_dir, "combined_aligned.fasta")

            try:
                run_alignment_with_mafft(prompt_fasta, prompt_aligned, threads)
                run_alignment_with_mafft(sampled_fasta, sampled_aligned, threads)
                run_alignment_with_mafft(combined_fasta, combined_aligned, threads)
            except subprocess.CalledProcessError as e:
                print(f"MAFFT alignment failed: {e}")
                return {}

            # Create logos
            prompt_logo = os.path.join(output_dir, "prompt_logo.png")
            sampled_logo = os.path.join(output_dir, "sampled_logo.png")
            create_logo_from_fasta(prompt_aligned, prompt_logo)
            create_logo_from_fasta(sampled_aligned, sampled_logo)

            # Read aligned sequences for similarity computation from combined alignment
            combined_alignment = AlignIO.read(combined_aligned, "fasta")
            all_aligned_seqs = [str(record.seq) for record in combined_alignment]
            all_accessions = [record.id for record in combined_alignment]

            # Separate prompt and sampled sequences from combined alignment
            prompt_aligned_seqs = []
            sampled_aligned_seqs = []

            for i, acc in enumerate(all_accessions):
                if acc.startswith("prompt_"):
                    prompt_aligned_seqs.append(all_aligned_seqs[i])
                elif acc.startswith("sampled_"):
                    sampled_aligned_seqs.append(all_aligned_seqs[i])

            # Compute maximum similarity for each sampled sequence with prompt sequences
            sampled_to_prompt_max_similarities = []
            for sampled_seq in sampled_aligned_seqs:
                max_similarity = max(
                    compute_sequence_similarity(sampled_seq, prompt_seq)
                    for prompt_seq in prompt_aligned_seqs
                )
                sampled_to_prompt_max_similarities.append(max_similarity)

            # Compute maximum similarity between sampled sequences
            sampled_to_sampled_max_similarities = []
            for i, sampled_seq1 in enumerate(sampled_aligned_seqs):
                max_similarity = max(
                    compute_sequence_similarity(sampled_seq1, sampled_seq2)
                    for j, sampled_seq2 in enumerate(sampled_aligned_seqs)
                    if i != j  # Don't compare sequence with itself
                )
                sampled_to_sampled_max_similarities.append(max_similarity)

            # Compute statistics
            results = {
                "sampled_to_prompt_min_max_similarity": min(
                    sampled_to_prompt_max_similarities
                ),
                "sampled_to_prompt_mean_max_similarity": np.mean(
                    sampled_to_prompt_max_similarities
                ),
                "sampled_to_prompt_max_max_similarity": max(
                    sampled_to_prompt_max_similarities
                ),
                "sampled_to_sampled_min_max_similarity": min(
                    sampled_to_sampled_max_similarities
                ),
                "sampled_to_sampled_mean_max_similarity": np.mean(
                    sampled_to_sampled_max_similarities
                ),
                "sampled_to_sampled_max_max_similarity": max(
                    sampled_to_sampled_max_similarities
                ),
                "num_prompt_sequences": len(prompt_sequences),
                "num_sampled_sequences": len(sampled_sequences),
            }
            json_path = os.path.join(output_dir, "sequence_analysis.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)
            if verbose:
                print(f"Similarity statistics: {results}")

            return results
