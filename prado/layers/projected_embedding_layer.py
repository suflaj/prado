import copy
from typing import Optional

import torch
from torch import nn

from .embeddings.embedding import PradoEmbedding
from .projections.projector import PradoProjector


class ProjectedEmbeddingLayerConfig:
    __feature_length_key = ("feature_length",)
    __embedding_length_key = "embedding_length"

    def __init__(self, feature_length: int, embedding_length: int):
        # There is no need to check these since they will be
        # automatically checked when subcomponents are created.
        self.feature_length = feature_length
        self.embedding_length = embedding_length


class ProjectedEmbeddingLayer(nn.Module):
    def __init__(
        self,
        feature_length: int = None,
        embedding_length: int = None,
        config: Optional[ProjectedEmbeddingLayerConfig] = None,
    ):
        if config is None:
            config = ProjectedEmbeddingLayerConfig(
                feature_length=feature_length, embedding_length=embedding_length
            )

        self._config = copy.deepcopy(config)

        self._projection = PradoProjector(feature_length=self.B)
        self._embedding = PradoEmbedding(in_features=self.B, out_features=self.d)

    # region Properties
    @property
    def feature_length(self):
        return self._config.feature_length

    @property
    def B(self):
        return self.feature_length

    @property
    def embedding_length(self):
        return self._config.embedding_length

    @property
    def d(self):
        return self.embedding_length

    # endregion

    def forward(self, x):
        projection = self._projection(x)
        projected_embedding = self._embedding(projection)

        return projected_embedding
