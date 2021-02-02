import copy
import sys
from typing import Any, Callable, Dict, Iterator, Optional

import bitarray
from datasketch import MinHash
import farmhash
import nltk
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from . import assertion
from . import constants
from .projection_operator import ProjectionOperator, PradoProjectionOperator


class ProjectionDataset(Dataset):
    @staticmethod
    def _check_init_args(
        self,
        dataset: Dataset,
        transformation_map: Dict[int, Callable[[str], Any]],
        verbosity: int,
    ):
        assertion.assert_type("dataset", dataset, Dataset)
        assertion.assert_type("transformation_map", transformation_map, dict)
        assertion.assert_type("verbosity", verbosity, int)

        assertion.assert_all_int("transfomation_map keys", transformation_map.keys())
        assertion.assert_all_non_negative(
            "transformation_map keys", transformation_map.keys()
        )
        assertion.assert_all_callable(
            "transformation_map values", transformation_map.values()
        )

        return dataset, transformation_map, verbosity

    def __init__(
        self,
        dataset: Dataset,
        transformation_map: Dict[int, Callable[[str], Any]],
        verbosity: int = 0,
    ):
        super().__init__()

        dataset, transformation_map, verbosity = ProjectionDataset._check_init_args(
            dataset=dataset, transformation_map=transformation_map, verbosity=verbosity
        )

        self._content = list()

        iterator = iter(dataset)

        if verbosity > 0:
            iterator = tqdm(
                iterator,
                desc="Transforming dataset",
                total=len(dataset),
                file=sys.stdout,
            )

        for row in iterator:
            new_content = copy.deepcopy(row)

            for index, function in transformation_map.items():
                new_content[index] = function(new_content[index])

            self._content.append(new_content)

    def __getitem__(self, key):
        return self._content[key]

    def __len__(self):
        return len(self._content)


class Projector:
    def transform_text(self, text: str) -> Any:
        raise NotImplementedError

    def generate_dataset(
        self,
        original_dataset: Dataset,
        application_columns: Iterator[int] = (0,),
        column_transformation: Optional[Callable[[str], Any]] = None,
        verbosity: int = 0,
    ) -> ProjectionDataset:
        assertion.assert_iterable("application_columns", application_columns)

        assertion.assert_all_int("application_columns", application_columns)
        assertion.assert_all_non_negative("application_columns", application_columns)

        if column_transformation is None:
            column_transformation = self.transform_text

        assertion.assert_callable("column_transformation", column_transformation)

        transformation_map = {k: column_transformation for k in application_columns}

        return ProjectionDataset(
            dataset=original_dataset,
            transformation_map=transformation_map,
            verbosity=verbosity,
        )


class PradoProjectorConfig:
    __feature_length_key = "feature_length"

    @staticmethod
    def _check_init_args(self, feature_length: int):
        assertion.assert_type("feature_length", feature_length, int)

        assertion.assert_positive("feature_length", feature_length)

        return feature_length

    def __init__(self, feature_length: int):
        feature_length = PradoProjectorConfig._check_init_args(
            feature_length=feature_length
        )

        self.feature_length = feature_length

    def serialize(self):
        return {PradoProjectorConfig.__feature_length_key: self.feature_length}

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]) -> "PradoProjectorConfig":
        feature_length = dictionary.get(PradoProjectorConfig.__feature_length_key)

        return PradoProjectorConfig(feature_length=feature_length)


class PradoProjector(nn.Module, Projector):
    def __init__(
        self,
        feature_length: int = None,
        config: Optional[PradoProjectorConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = PradoProjectorConfig(feature_length=feature_length)

        self._config = copy.deepcopy(config)
        self._hashobj = MinHash(num_perm=self.n_permutations, hashfunc=farmhash.hash32)
        self._projection_operator = PradoProjectionOperator()

    # region Properties
    @property
    def feature_length(self) -> int:
        return self._config.feature_length

    @property
    def B(self) -> int:
        return self.feature_length

    @property
    def n_permutations(self) -> int:
        return (2 * self.B + 32 - 1) // 32

    # endregion

    def transform_text(self, text: str) -> Any:
        # (N, ) - note that we will remove meaningless tokens
        # from the nltk tokenization, check the regex to fully
        # understand what that means.
        tokens = [
            token
            for token in nltk.tokenize.word_tokenize(text)
            if constants.MEANINGLESS_REGEX.fullmatch(token) is None
        ]

        token_features = list()

        for token in tokens:
            self._hashobj.update(token)

            # (4 * n_permutations, )
            token_as_bytes = b"".join(
                int(x).to_bytes(4, "big") for x in self._hashobj.digest()
            )

            # (32 * n_permutations, )
            token_as_bits = bitarray.bitarray()
            token_as_bits.frombytes(token_as_bytes)

            # (2B, ) (note that self.feature_length is B)
            torch_bits = torch.tensor(
                token_as_bits[: 2 * self.feature_length], dtype=torch.float
            )

            token_features.append(torch_bits)
            self._hashobj.clear()

        # (N, 2B)
        token_features = torch.tensor(token_features, dtype=torch.float)

        # (N, B, 2)
        token_features = torch.reshape(token_features, (token_features.shape[0], -1, 2))

        # (N, B)
        fingerprint = self._projection_operator(token_features)

        return fingerprint

    # Why so clunky? Because reprocessing inputs should yield
    # better throughput and ONNX should quantize the rest of the
    # model better since it's more like an ordinary DL model.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Doesn't do anything but pass the tensor over. You should make sure to
        transform the text beforehand.

        Args:
            x (torch.Tensor): A torch.Tensor of shape (number of tokens, B).

        Returns:
            torch.Tensor: A torch.Tensor of shape (number of tokens, B).
        """
        return x
