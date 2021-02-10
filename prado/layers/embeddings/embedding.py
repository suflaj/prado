import copy
from typing import Any, Dict, Optional

import torch
from torch import nn

from . import assertion


class PradoEmbeddingConfig:
    __in_features_key = "in_features"
    __out_features_key = "out_features"
    __bias_key = "bias"

    @staticmethod
    def _check_init_args(in_features: int, out_features: int, bias: bool):
        assertion.assert_type("in_features", in_features, int)
        assertion.assert_type("out_features", out_features, int)
        assertion.assert_type("bias", bias, bool)

        assertion.assert_positive("in_features", in_features)
        assertion.assert_positive("out_features", out_features)

        return in_features, out_features, bias

    def __init__(self, in_features: int, out_features: int, bias: bool):
        in_features, out_features, bias = PradoEmbeddingConfig._check_init_args(
            in_features=in_features, out_features=out_features, bias=bias
        )

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def serialize(self):
        return {
            PradoEmbeddingConfig.__in_features_key: self.in_features,
            PradoEmbeddingConfig.__out_features_key: self.out_features,
            PradoEmbeddingConfig.__bias_key: self.bias,
        }

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]) -> "EmbeddingConfig":
        in_features = dictionary.get(PradoEmbeddingConfig.__in_features_key)
        out_features = dictionary.get(PradoEmbeddingConfig.__out_features_key)
        bias = dictionary.get(PradoEmbeddingConfig.__bias_key)

        return PradoEmbeddingConfig(
            in_features=in_features, out_features=out_features, bias=bias
        )


class PradoEmbedding(nn.Module):
    def __init__(
        self,
        in_features: int = None,
        out_features: int = None,
        bias: bool = False,
        config: Optional[PradoEmbeddingConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = PradoEmbeddingConfig(
                in_features=in_features, out_features=out_features, bias=bias
            )

        self._config = copy.deepcopy(config)

        self._linear = nn.Linear(
            in_features=self.in_features, out_features=self.out_features, bias=self.bias
        )

    # region Properties
    @property
    def in_features(self):
        return self._config.in_features

    @property
    def out_features(self):
        return self._config.out_features

    @property
    def bias(self):
        return self._config.bias

    # endregion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x)
