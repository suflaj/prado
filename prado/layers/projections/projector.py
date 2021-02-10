import copy
from typing import Any, Dict, List, Optional

import bitarray
from datasketch import MinHash
import farmhash
import torch

from . import assertion
from .projection_operator import PradoProjectionOperator


class Projector:
    pass


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


class PradoProjector(Projector):
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

    def forward(self, x: List[str]) -> torch.Tensor:
        # In case we didn't clear the hash object before this.
        self._hashobj.clear()

        token_features = list()

        for token in x:
            self._hashobj.update(token)

            # (4 * n_permutations, )
            token_as_bytes = b"".join(
                int(x).to_bytes(4, "big") for x in self._hashobj.digest()
            )

            # (32 * n_permutations, )
            token_as_bits = bitarray.bitarray()
            token_as_bits.frombytes(token_as_bytes)

            # (2B, ) - MinHash can give us larger hashes than
            # we need. It is recommended to set B up so this
            # doesn't destroy/skip data. In other words, B should
            # be a multiplier of 16.
            torch_bits = torch.tensor(token_as_bits[: 2 * self.B], dtype=torch.float)

            token_features.append(torch_bits)

            # We clear the hash object here because each word
            # hash should be independent of the previous word
            # hashes.
            self._hashobj.clear()

        # (N, 2B)
        token_features = torch.tensor(token_features, dtype=torch.float)

        # (N, B, 2)
        token_features = torch.reshape(token_features, (token_features.shape[0], -1, 2))

        # (N, B)
        fingerprint = self._projection_operator(token_features)

        return fingerprint
