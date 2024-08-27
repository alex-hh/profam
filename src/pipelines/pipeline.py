import os
import shutil
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from src import constants
from src.data import fasta
from src.data.objects import ProteinDocument
from src.evaluators.base import SamplingEvaluator
from src.utils.utils import maybe_print


class BaseEvaluatorPipeline:

    """A validation pipeline handles loading of documents, running of models and storing of results.

    The pipeline basically wraps around an evaluator which determines the logic of input
    generation and metric computation.

    If multiple sets of metrics should be run on a single set of generations, the evaluator needs
    to be written appropriately.
    """

    def __init__(
        self,
        pipeline_id: str,
        benchmark_directory: str = None,
        save_to_file: bool = True,
    ):
        self.pipeline_id = pipeline_id
        self.pipeline_directory = os.path.join(
            benchmark_directory or constants.BENCHMARK_RESULTS_DIR,
            self.pipeline_id,
        )
        self.save_to_file = save_to_file
        self.load_results()

    def instance_ids(self):
        raise NotImplementedError()

    def load_results(self) -> pd.DataFrame:
        """Load results dataframe from local disk location."""
        results_path = os.path.join(self.pipeline_directory, "results.csv")
        if self.save_to_file and os.path.exists(results_path):
            self.results_df = pd.read_csv(results_path)
        else:
            self.results_df = pd.DataFrame(columns=["evaluator", "sampler", "instance"])
        self.results_df.set_index(["evaluator", "sampler", "instance"], inplace=True)

    def has_result(self, validation_id: str, instance_id: str, model_id: str) -> bool:
        """Check if validation, instance, model combo is present in results df index."""
        return (validation_id, model_id, instance_id) in self.results_df.index

    def add_result(
        self,
        validation_id: str,
        instance_id: str,
        model_id: str,
        result: Dict[str, float],
    ) -> None:
        """Add a result to the results dataframe."""
        # drop any existing result for this instance, validation, model combo
        # then concatenate a new row to the df
        self.results_df.drop(
            index=(validation_id, model_id, instance_id), inplace=True, errors="ignore"
        )
        self.results_df = pd.concat(
            [
                self.results_df,
                pd.DataFrame([result]).set_index(["evaluator", "sampler", "instance"]),
            ]
        )

    def save_results(self) -> None:
        """Save results dataframe to local disk location."""
        if self.save_to_file:
            results_path = os.path.join(self.pipeline_directory, "results.csv")
            self.results_df.to_csv(results_path, index=True)

    def make_summary(self):
        summaries = []
        for instance_id in self.instance_ids():
            summary = self.get_instance_summary(instance_id)
            summary["instance_id"] = instance_id
            summaries.append(summary)
        return pd.DataFrame.from_records(summaries)

    def get_instance_summary(self, instance_id: str) -> Dict[str, float]:
        raise NotImplementedError()

    def run(
        self,
        model,
        rerun_model: bool = False,
        rerun_evaluator: bool = False,
    ):
        raise NotImplementedError()


class GenerationsEvaluatorPipeline(BaseEvaluatorPipeline):

    """Validation that computes metrics given a set of generated sequences."""

    def __init__(
        self,
        num_generations: int,
        pipeline_id: str,
        benchmark_directory: str = None,
        save_to_file: bool = True,
    ):
        self.num_generations = num_generations
        self.generations = defaultdict(dict)
        print(
            f"Initialised pipeline ID {pipeline_id} num generations {num_generations}"
        )
        super().__init__(
            pipeline_id,
            benchmark_directory=benchmark_directory,
            save_to_file=save_to_file,
        )

    def has_generations(self, instance_id: str, model_id: str) -> bool:
        if not self.save_to_file:
            return (
                model_id in self.generations
                and instance_id in self.generations[model_id]
            )
        else:
            output_path = os.path.join(
                self.pipeline_directory, instance_id, model_id, "sequences.fa"
            )
            retval = os.path.isfile(output_path)
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

    def load_protein_document(self, instance_id):
        raise NotImplementedError()

    def run_evaluator_on_instance(
        self,
        sampler_name: str,
        instance_id: str,
        evaluator: SamplingEvaluator,
        protein_document: ProteinDocument,
        rerun_evaluator: bool = False,
    ) -> None:
        generated_sequences = self.load_generations(instance_id, sampler_name)
        if rerun_evaluator or not self.has_result(
            evaluator.name, instance_id, sampler_name
        ):
            output_dir = os.path.join(
                self.pipeline_directory, instance_id, sampler_name, evaluator.name
            )
            if rerun_evaluator:
                if os.path.isdir(output_dir):
                    shutil.rmtree(output_dir)
            metrics = evaluator.evaluate_samples(
                protein_document,
                generated_sequences,
                output_dir=output_dir,
            )
            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            print(f"Instance {instance_id} metrics: {metrics_str}")
            metrics.update(self.get_instance_summary(instance_id))
            metrics["sampler"] = sampler_name
            metrics["instance"] = instance_id
            metrics["evaluator"] = evaluator.name
            self.add_result(evaluator.name, instance_id, sampler_name, metrics)

    def save_generations(self, instance_id, model_name, sequences: List[str]) -> None:
        if self.save_to_file:
            outputs_dir = os.path.join(self.pipeline_directory, instance_id, model_name)
            os.makedirs(outputs_dir, exist_ok=True)
            fasta.output_fasta(
                [f"seq{i}" for i in range(len(sequences))],
                sequences,
                os.path.join(outputs_dir, "sequences.fa"),
            )
        else:
            self.generations[model_name][instance_id] = sequences

    def load_generations(self, instance_id: str, sampler_name: str) -> List[str]:
        if self.save_to_file:
            outputs_dir = os.path.join(
                self.pipeline_directory, instance_id, sampler_name
            )
            fasta_file = os.path.join(outputs_dir, "sequences.fa")
            _, sequences = fasta.read_fasta(fasta_file)
            return sequences
        else:
            sequences = self.generations[sampler_name][instance_id]
            return sequences

    def run(
        self,
        sampler,
        evaluator,
        verbose: bool = True,
        rerun_sampler: bool = False,
        rerun_evaluator: bool = True,
        sampling_only: bool = False,
    ):
        instance_ids = self.instance_ids()
        for instance_id in instance_ids:
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
                generated_sequences = evaluator.run_sampling(
                    sampler,
                    protein_document,
                    self.num_generations,
                )
                self.save_generations(instance_id, sampler.name, generated_sequences)
            else:
                generated_sequences = self.load_generations(instance_id, sampler.name)

            if not sampling_only:
                try:
                    self.run_evaluator_on_instance(
                        sampler.name,
                        instance_id=instance_id,
                        evaluator=evaluator,
                        protein_document=protein_document,
                        rerun_evaluator=rerun_evaluator,
                    )
                except Exception as e:
                    print("Failed to run validation on instance", instance_id)
                    raise e

        if sampling_only:
            return

        # TODO format to limit decimal places
        combo_results = self.results_df.loc[(evaluator.name, sampler.name)]
        avg_metrics = combo_results.mean()
        avg_metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in avg_metrics.items()])
        maybe_print(
            f"Validation `{evaluator.name}` model {sampler.name} average metrics: "
            f"{avg_metrics_str} ({len(combo_results)} instances)",
            verbose=verbose,
        )

        self.save_results()
        return combo_results
