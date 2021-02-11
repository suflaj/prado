import copy
from typing import List, Optional

import torch
from torch import nn

from .projected_embedding import ProjectedEmbeddingLayer
from .convolutions.projected_attention import ProjectedAttentionLayer


class PradoCoreConfig:
    def __init__(
        self,
        feature_length: int,
        embedding_length: int,
        dropout: float,
        out_channels: int,
        skipgram_patterns: List[int],
        out_features: int,
    ):
        self.feature_length = feature_length
        self.embedding_length = embedding_length
        self.dropout = dropout
        self.out_channels = out_channels
        self.skipgram_patterns = copy.deepcopy(skipgram_patterns)
        self.out_features = out_features


class PradoCore(nn.Module):
    def __init__(
        self,
        feature_length: int,
        embedding_length: int,
        dropout: float,
        out_channels: int,
        skipgram_patterns: List[str],
        out_features: int,
        config: Optional[PradoCoreConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = PradoCoreConfig(
                feature_length=feature_length,
                embedding_length=embedding_length,
                dropout=dropout,
                out_channels=out_channels,
                skipgram_patterns=skipgram_patterns,
                out_features=out_features,
            )

        self._config = copy.deepcopy(config)

        self._projected_embedding_layer = ProjectedEmbeddingLayer(
            feature_length=feature_length, embedding_length=embedding_length
        )

        self._projected_attention_layers = nn.ModuleList(
            [
                ProjectedAttentionLayer(
                    in_channels=1,
                    out_channels=self.out_channels,
                    embedding_length=self.embedding_length,
                    skipgram_pattern=skipgram_pattern,
                )
                for skipgram_pattern in self.skipgram_patterns
            ]
        )

        self._decoder = nn.Linear(
            in_features=self.encoder_out_features,
            out_features=self.out_features,
            bias=True,
        )

        self._dropout = nn.Dropout(p=self.dropout)
        self._tanh = nn.Tanh()

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

    @property
    def dropout(self):
        return self._config.dropout

    @property
    def out_channels(self):
        return self._config.out_channels

    @property
    def skipgram_patterns(self):
        return self._config.skipgram_patterns

    @property
    def encoder_out_features(self):
        return sum([x.out_channels for x in self._projected_attention_layers])

    @property
    def out_features(self):
        return self._config.out_features

    # endregion

    def forward(self, x: torch.Tensor):
        # (batch_size, N, 2B) -> (batch_size, N, d)
        projections = self._projected_embedding_layer(x)
        projections = self._tanh(projections)
        projections = self._dropout(projections)

        # (batch_size, N, d) -> (batch_size, 1, N, d)
        projections = projections.reshape(
            (projections.shape[0], 1, *projections.shape[1:])
        )

        # (n_convolutions, batch_size, out_channels)
        #
        # n_convolutions is len(self._projected_attention_layers)
        features = [layer(projections) for layer in self._projected_attention_layers]

        # (n_convolution, batch_size, out_channels) ->
        # (batch_size, encoder_out_features)
        flattened_features = torch.cat(features, dim=1)
        flattened_features = self._tanh(flattened_features)

        # (batch_size, encoder_out_features) ->
        # (batch_size, out_features)
        decoded = self._decoder(flattened_features)

        return decoded
