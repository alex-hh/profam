import os
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import tqdm

from src import constants
from src.data.objects import ProteinDocument
from src.data.preprocessing import BasePreprocessor
from src.evaluators.base import BaseEvaluator, SamplingEvaluator, ScoringEvaluator
from src.evaluators.models import (
    BaseModelForEvaluation,
    SamplingModelForEvaluation,
    ScoringModelForEvaluation,
)
from src.sequence import fasta
from src.utils.utils import maybe_print


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
        preprocessor: Optional[
            BasePreprocessor
        ] = None,  # we only use the build_document method
        benchmark_directory: str = None,
        save_results_to_file: bool = True,
    ):
        """preprocessor: a bare preprocessor (no transform_fns), to build document from raw data."""
        self.pipeline_id = pipeline_id
        self.preprocessor = preprocessor
        # assert (
        #     self.preprocessor.transform_fns is None
        # ), "Pipeline preprocessor should not have transforms"  # doesnt matter: they dont get called
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

    def save_prompt(self, instance_id, model_name, prompt: str) -> None:
        if self.save_results_to_file:
            outputs_dir = os.path.join(
                self.pipeline_directory, "prompts", instance_id, model_name
            )
            os.makedirs(outputs_dir, exist_ok=True)
            prompt.to_json(os.path.join(outputs_dir, "prompt.json"))
        else:
            self.prompts[model_name][instance_id] = prompt

    def load_prompt(self, instance_id: str, model_name: str) -> ProteinDocument:
        if self.save_results_to_file:
            outputs_dir = os.path.join(
                self.pipeline_directory, "prompts", instance_id, model_name
            )
            prompt_file = os.path.join(outputs_dir, "prompt.json")
            prompt = ProteinDocument.from_json(prompt_file)
            return prompt
        else:
            prompt = self.prompts[model_name][instance_id]
            return prompt

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

    def get_instance_summary(
        self, instance_id: str, protein_document: Optional[ProteinDocument] = None
    ) -> Dict[str, float]:
        raise NotImplementedError()

    def aggregate_results(
        self,
        evaluators: List[BaseEvaluator],
        model: BaseModelForEvaluation,
        verbose: bool = False,
    ):
        # TODO format to limit decimal places
        outputs = {}
        for evaluator in evaluators:
            model_results = self.results_dfs[evaluator.name].loc[
                (evaluator.name, model.name)
            ]
            avg_metrics = model_results.mean()
            avg_metrics_str = ", ".join(
                [f"{k}: {v:.3f}" for k, v in avg_metrics.items()]
            )
            maybe_print(
                f"Validation `{evaluator.name}` model {model.name} average metrics: "
                f"{avg_metrics_str} ({len(model_results)} instances)",
                verbose=verbose,
            )
            outputs[evaluator.name] = model_results

        return outputs

    def validate_configs(self, sampler_config, evaluator_config):
        # save configs to appropriate directory.
        # if rerunning, we check that the configs match, otherwise we raise
        # an exception. (TODO: allow overriding with an ignore_config_mismatch flag).
        raise NotImplementedError()

    def get_protein_example(self, instance_id):
        """Load a protein example (a dict to be parsed by preprocessor)."""
        raise NotImplementedError()

    def load_protein_document(self, instance_id):
        """Load a protein document for a given instance.

        The document should be raw: no subsampling or positional encoding.
        i.e. the output of preprocessor.build_document with max_tokens=None, shuffle=False.
        """
        example = self.get_protein_example(instance_id)
        return self.preprocessor.build_document(example, max_tokens=None, shuffle=False)


class GenerationsEvaluatorPipeline(BaseEvaluatorPipeline):

    """Validation that computes metrics given a set of generated sequences."""

    def __init__(
        self,
        num_generations: int,
        pipeline_id: str,
        preprocessor: Optional[BasePreprocessor] = None,
        benchmark_directory: str = None,
        save_results_to_file: bool = True,
    ):
        self.num_generations = num_generations
        self.generations = defaultdict(dict)
        self.prompts = defaultdict(dict)
        print(
            f"Initialised pipeline ID {pipeline_id} num generations {num_generations}"
        )
        super().__init__(
            pipeline_id,
            preprocessor=preprocessor,
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

    def run_evaluator_on_instance(
        self,
        instance_id: str,
        sampler_name: str,
        generated_sequences: List[str],
        evaluator: SamplingEvaluator,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        rerun_evaluator: bool = False,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
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
            self.add_result(evaluator.name, instance_id, sampler_name, metrics)

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

    def load_generations(self, instance_id: str, sampler_name: str) -> List[str]:
        if self.save_results_to_file:
            outputs_dir = os.path.join(
                self.pipeline_directory, "generations", instance_id, sampler_name
            )
            fasta_file = os.path.join(outputs_dir, "sequences.fa")
            _, sequences = fasta.read_fasta(fasta_file)
            return sequences
        else:
            sequences = self.generations[sampler_name][instance_id]
            return sequences

    def run(
        self,
        sampler: SamplingModelForEvaluation,
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
                generations, prompt = sampler.sample_seqs(
                    protein_document, self.num_generations
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
                            instance_id=instance_id,
                            sampler_name=sampler.name,
                            generated_sequences=generations,
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

        self.save_results()
        outputs = self.aggregate_results(evaluators, sampler, verbose=verbose)

        return outputs


class ResiduePredictionsEvaluatorPipeline(BaseEvaluatorPipeline):
    """Pipeline for evaluating amino acid predictions over documents (perplexity, accuracy etc).

    Evaluators are initially expected to take a HF model outputs object.
        - if we find this useful, we can figure out exactly which outputs are required,
        and extend to work with equivalent outputs from non-HF baselines.

        (probably would need to be something like predicted probs for 20 standard aas)

    TODO: might be easier to just use validation_step.
    """

    pass


class CompletionScoringEvaluatorPipeline(BaseEvaluatorPipeline):
    """Pipeline for scoring completions (e.g. ProteinGym mutants) given documents."""

    def __init__(
        self,
        pipeline_id: str,
        preprocessor: Optional[BasePreprocessor] = None,
        benchmark_directory: str = None,
        save_results_to_file: bool = True,
    ):
        self.scored_completions = defaultdict(dict)
        self.prompts = defaultdict(dict)
        print(f"Initialised pipeline ID {pipeline_id}")
        super().__init__(
            pipeline_id,
            preprocessor=preprocessor,
            benchmark_directory=benchmark_directory,
            save_results_to_file=save_results_to_file,
        )

    def has_scored_completions(self, instance_id: str, model_id: str) -> bool:
        # TODO: check prompt as well
        if not self.save_results_to_file:
            return (
                model_id in self.scored_completions
                and instance_id in self.scored_completions[model_id]
            )
        else:
            output_path = os.path.join(
                self.pipeline_directory,
                "scored_completions",
                instance_id,
                model_id,
                "scored_completions.csv",
            )
            prompt_output_path = os.path.join(
                self.pipeline_directory, "prompts", instance_id, model_id, "prompt.json"
            )
            retval = os.path.isfile(output_path) and prompt_output_path
            return retval

    def has_all_scored_completions(self, model_id: str) -> None:
        return all(
            [
                self.has_scored_completions(instance_id, model_id)
                for instance_id in self.instance_ids()
            ]
        )

    def run_evaluator_on_instance(
        self,
        scorer_name: str,
        instance_id: str,
        scored_completions_df: pd.DataFrame,
        evaluator: ScoringEvaluator,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        rerun_evaluator: bool = False,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        if rerun_evaluator or not self.has_result(
            evaluator.name, instance_id, scorer_name
        ):
            output_dir = os.path.join(
                self.pipeline_directory, evaluator.name, instance_id, scorer_name
            )
            if rerun_evaluator:
                if os.path.isdir(output_dir):
                    shutil.rmtree(output_dir)

            metrics = evaluator.evaluate_scored_completions(
                prompt=prompt,
                scored_completions_df=scored_completions_df,
                device=device,
            )

            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            if verbose:
                print(f"Instance {instance_id} {evaluator.name} metrics: {metrics_str}")

            metrics.update(
                self.get_instance_summary(
                    instance_id, protein_document=protein_document
                )
            )
            metrics["scorer"] = scorer_name
            metrics["instance"] = instance_id
            metrics["evaluator"] = evaluator.name
            self.add_result(evaluator.name, instance_id, scorer_name, metrics)

    def save_scored_completions(
        self, instance_id, model_name, scored_completions_df: pd.DataFrame
    ) -> None:
        if self.save_results_to_file:
            outputs_dir = os.path.join(
                self.pipeline_directory, "scored_completions", instance_id, model_name
            )
            os.makedirs(outputs_dir, exist_ok=True)
            scored_completions_df.to_csv(
                os.path.join(outputs_dir, "scored_completions.csv"), index=False
            )
        else:
            self.scored_completions[model_name][instance_id] = scored_completions_df

    def load_scored_completions(
        self, instance_id: str, scorer_name: str
    ) -> pd.DataFrame:
        if self.save_results_to_file:
            outputs_dir = os.path.join(
                self.pipeline_directory, "scored_completions", instance_id, scorer_name
            )
            return pd.read_csv(os.path.join(outputs_dir, "scored_completions.csv"))
        else:
            scored_completions_df = self.scored_completions[scorer_name][instance_id]
            return scored_completions_df

    def load_completions(
        self, instance_id: str
    ) -> Tuple[pd.DataFrame, ProteinDocument]:
        raise NotImplementedError()

    def run(
        self,
        scorer: ScoringModelForEvaluation,
        evaluators: Union[List[ScoringEvaluator], ScoringEvaluator],
        verbose: bool = True,
        rerun_scorer: bool = False,
        rerun_evaluator: bool = True,
        scoring_only: bool = False,
        offload_scorer: bool = False,
        device: Optional[str] = None,
        disable_tqdm: bool = False,
    ):
        if not isinstance(evaluators, List):
            assert isinstance(evaluators, ScoringEvaluator)
            evaluators = [evaluators]
        for evaluator in evaluators:
            self.load_results(evaluator.name)

        instance_ids = self.instance_ids()
        if rerun_scorer:
            rerun_evaluator = True

        for instance_id in tqdm.tqdm(instance_ids, disable=verbose or disable_tqdm):
            maybe_print(
                "Running evaluation pipeline for instance", instance_id, verbose=verbose
            )
            protein_document = self.load_protein_document(instance_id)
            completions_df, completions = self.load_completions(instance_id)
            if rerun_scorer or not self.has_scored_completions(
                instance_id, scorer.name
            ):
                maybe_print(
                    f"Running generations for instance: {instance_id}",
                    verbose=verbose,
                    flush=True,
                )
                scored_completions, prompt = scorer.score_completions(
                    protein_document, completions
                )
                scored_completions_df = completions_df.copy()
                scored_completions_df["score"] = scored_completions
                self.save_scored_completions(
                    instance_id, scorer.name, scored_completions_df
                )
                self.save_prompt(instance_id, scorer.name, prompt)
            else:
                maybe_print(
                    f"Loading generations for instance: {instance_id}",
                )
                scored_completions_df = self.load_scored_completions(
                    instance_id, scorer.name
                )
                prompt = self.load_prompt(instance_id, scorer.name)

            scorer_device = scorer.device
            if not scoring_only:
                if offload_scorer:
                    scorer.to(
                        "cpu"
                    )  # offload memory to CPU. TODO: consider avoiding all this device switching
                for evaluator in evaluators:
                    try:
                        self.run_evaluator_on_instance(
                            scorer_name=scorer.name,
                            instance_id=instance_id,
                            scored_completions_df=scored_completions_df,
                            evaluator=evaluator,
                            prompt=prompt,
                            protein_document=protein_document,
                            rerun_evaluator=rerun_evaluator,
                            device=device,
                        )
                    except Exception as e:
                        print("Failed to run validation on instance", instance_id)
                        raise e
                if offload_scorer:
                    scorer.to(scorer_device)  # move back to original device

        if scoring_only:
            return

        self.save_results()
        outputs = self.aggregate_results(evaluators, scorer, verbose=verbose)
        return outputs
