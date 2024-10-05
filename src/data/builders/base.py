from typing import Any, Dict, List, Optional

from src.data.processors import ProteinDocumentPreprocessor
from src.data.tokenizers import ProFamTokenizer


class BaseProteinDataset:
    def __init__(
        self,
        name: str,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
    ):
        self.name = name
        self.preprocessor = preprocessor
        if preprocessor is not None:
            self.interleave_structure_sequence = (
                preprocessor.interleave_structure_sequence
            )
        else:
            self.interleave_structure_sequence = False

    def process(
        self,
        dataset: Any,
        tokenizer: ProFamTokenizer,
    ):
        raise NotImplementedError("Must implement process method")

    def load(self, data_dir="data", world_size: int = 1, verbose: bool = False):
        raise NotImplementedError("Must implement load method")

    def _build_document(self, example):
        # private method has fixed signature; static methods can have variable signature
        raise NotImplementedError(
            "Must implement build_document method on child dataset builder"
        )

    def preprocess_example(
        self,
        example: Any,
        tokenizer: ProFamTokenizer,
    ) -> Dict[str, Any]:
        # example is a single row dict in this case
        proteins = self._build_document(
            example,
        )
        example = self.preprocessor.preprocess_protein_data(
            proteins,
            tokenizer,
        )
        example["ds_name"] = self.name
        if "identifier" in example:
            example["identifier"] = self.name + "/" + example["identifier"]
        return example

    def batched_preprocess_examples(
        self,
        examples: List[Any],
        tokenizer: ProFamTokenizer,
    ) -> Dict[str, List[Any]]:
        """Function to be mapped.

        a map is an instruction for converting an example to a new example.
        it should return a datapoint dict.

        a batched map is an instruction for converting a set of examples to a
        new set of examples (not necessarily of the same size). it should return a dict of lists,
        where the length of the lists determines the size of the new set of examples.
        """
        proteins_list = [self._build_document(example) for example in examples]
        examples = self.preprocessor.batched_preprocess_protein_data(
            proteins_list,
            tokenizer,
        )
        examples["ds_name"] = [self.name] * len(examples["input_ids"])
        if "identifier" in examples:
            examples["identifier"] = [
                self.name + "/" + ident for ident in examples["identifier"]
            ]
        return examples
