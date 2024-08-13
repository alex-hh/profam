import os
from typing import Dict, List

import pandas as pd

from src import constants
from src.data import fasta
from src.data.objects import ProteinDocument
from src.evaluators.base import SamplingEvaluator


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
    ):
        self.pipeline_id = pipeline_id
        self.pipeline_directory = os.path.join(
            benchmark_directory or constants.BENCHMARK_RESULTS_DIR,
            self.pipeline_id,
        )
        self.load_results()

    def instance_ids(self):
        raise NotImplementedError()

    def load_results(self) -> pd.DataFrame:
        """Load results dataframe from local disk location."""
        results_path = os.path.join(self.pipeline_directory, "results.csv")
        if os.path.exists(results_path):
            self.results_df = pd.read_csv(results_path)
        else:
            self.results_df = pd.DataFrame(
                columns=["validation_id", "model_id", "target_id"]
            )
        self.results_df.set_index(
            ["validation_id", "model_id", "target_id"], inplace=True
        )

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
        # drop any existing result for this target, validation, model combo
        # then concatenate a new row to the df
        self.results_df.drop(
            index=(validation_id, model_id, instance_id), inplace=True, errors="ignore"
        )
        self.results_df = pd.concat(
            [
                self.results_df,
                pd.DataFrame([result]).set_index(
                    ["validation_id", "model_id", "target_id"]
                ),
            ]
        )

    def save_results(self) -> None:
        """Save results dataframe to local disk location."""
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
        evaluator: SamplingEvaluator,
        num_generations: int,
        pipeline_id: str,
        benchmark_directory: str = None,
    ):
        self.evaluator = evaluator
        self.num_generations = num_generations
        super().__init__(
            pipeline_id,
            benchmark_directory=benchmark_directory,
        )

    def has_generations(self, target_id: str, model_id: str) -> bool:
        output_path = os.path.join(
            self.form_outputs_path(target_id, model_id), "sequences.fa"
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

    def load_protein_document(self, instance_id):
        raise NotImplementedError()

    def run_evaluator_on_instance(
        self,
        model_id: str,
        instance_id: str,
        protein_document: ProteinDocument,
        rerun_evaluator: bool = False,
    ) -> None:
        generated_sequences = self.load_generations(instance_id, model_id)
        if rerun_evaluator or not self.has_result(
            self.evaluator.name, instance_id, model_id
        ):
            metrics = self.evaluator.evaluate_samples(
                protein_document, generated_sequences
            )
            metrics.update(self.get_instance_summary(instance_id))
            metrics["model_id"] = model_id
            metrics["target_id"] = instance_id
            metrics["validation_id"] = self.evaluator.name
            self.add_result(self.evaluator.name, instance_id, model_id, metrics)

    def save_generations(
        self, sequences: List[str], target_id: str, model_id: str, outputs_dir: str
    ) -> None:
        fasta.output_fasta(
            [f"seq{i}" for i in range(len(sequences))],
            sequences,
            os.path.join(outputs_dir, "sequences.fa"),
        )

    def load_generations(self, target_id: str, model_id: str) -> List[str]:
        outputs_dir = self.form_outputs_path(target_id, model_id)
        fasta_file = os.path.join(outputs_dir, "sequences.fa")
        _, sequences = fasta.read_fasta(fasta_file)
        return sequences

    def run_sampling(self, model, model_name, rerun: bool = False):
        instance_ids = self.instance_ids()
        for instance_id in instance_ids:
            protein_document = self.load_protein_document(instance_id)
            if rerun or not self.has_generations(instance_id, model_name):
                print(f"Running generations for target: {instance_id}")
                outputs_dir = os.path.join(
                    self.pipeline_directory, instance_id, model_name
                )
                os.makedirs(outputs_dir, exist_ok=True)
                generated_sequences = self.evaluator.run_sampling(
                    model, protein_document, self.num_generations
                )
                self.save_generations(
                    generated_sequences, instance_id, model_name, outputs_dir
                )

    def run_evaluation(self, model_name: str, rerun: bool = False):
        instance_ids = self.instance_ids()
        print(f"Running evaluation `{self.evaluator.name}` for model `{model_name}`")
        for instance_id in instance_ids:
            print(f"Running instance `{instance_id}`")
            protein_document = self.load_protein_document(instance_id)

            try:
                self.run_evaluator_on_instance(
                    model_name,
                    protein_document,
                    rerun_evaluator=rerun,
                )
            except Exception as e:
                print("Failed to run validation on instance", instance_id)
                raise e

        # TODO format to limit decimal places
        print(self.results_df)
        combo_results = self.results_df.loc[(self.evaluator.name, model_name)]
        avg_metrics = combo_results.mean()
        avg_metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in avg_metrics.items()])
        print(
            f"Validation `{self.evaluator.name}` model {model_name} average metrics: {avg_metrics_str} ({len(combo_results)} instances)"
        )

        self.save_results()

    def run(
        self,
        model,
        rerun_model: bool = False,
        rerun_evaluator: bool = False,
    ):
        """Run the validation pipeline for a given model and set of validations."""
        # TODO: instead of looping twice we could just loop once and evaluate as we go...
        # TODO: handle storing of outputs on evaluator side possibly?
        # 1. produce intermediate outputs (e.g. generated sequences) by running model on inputs
        self.run_sampling(model, rerun=rerun_model)

        # 2. evaluate the intermediate outputs with each validation
        self.run_evaluation(model.name, rerun=rerun_evaluator)
