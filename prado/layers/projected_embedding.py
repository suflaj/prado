import copy
from typing import Optional

from torch import nn

from .embeddings.embedding import PradoEmbedding
from .projections.projector import PradoProjector


class ProjectedEmbeddingLayerConfig:
    # TODO Serialization
    __feature_length_key = "feature_length"
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
        super().__init__()

        if config is None:
            # The reason we use these indirect values is because
            # they are checked and corrected. It might seem
            # counterintuitive to initialize the config after
            # member variables, but this saves us from
            # unnecessary validation.
            config = ProjectedEmbeddingLayerConfig(
                feature_length=feature_length,
                embedding_length=embedding_length,
            )

        self._config = copy.deepcopy(config)

        self._projection = PradoProjector(feature_length=self.B)
        self._embedding = PradoEmbedding(in_features=self.B, out_features=self.d)

        # Do this to ensure that values which were corrected are
        # correctly corrected in this config as well.
        self._config.feature_length = self._projection.feature_length
        self._config.embedding_length = self._embedding.out_features

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
