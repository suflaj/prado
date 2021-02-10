import copy
from prado.layers.convolutions.skipgram_convolution import SkipgramConvolution
from typing import Optional

import torch
from torch import nn


class ProjectedAttentionLayerConfig:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_length: int,
        skipgram_pattern: str,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_length = embedding_length
        self.skipgram_pattern = skipgram_pattern


class ProjectedAttentionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_length: int,
        skipgram_pattern: str,
        config: Optional[ProjectedAttentionLayerConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = ProjectedAttentionLayerConfig(
                in_channels=in_channels,
                out_channels=out_channels,
                embedding_length=embedding_length,
                skipgram_pattern=skipgram_pattern,
            )

        self._config = copy.deepcopy(config)

        self._projected_feature_layer = SkipgramConvolution(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            embedding_length=self.embedding_length,
            skipgram_pattern=self.skipgram_pattern,
        )

        self._attention_layer = SkipgramConvolution(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            embedding_length=self.embedding_length,
            skipgram_pattern=self.skipgram_pattern,
        )

        self._softmax = nn.Softmax(dim=-1)

    # region Properties
    @property
    def in_channels(self):
        return self._config.in_channels

    @property
    def out_channels(self):
        return self._config.out_channels

    @property
    def embedding_length(self):
        return self._config.embedding_length

    @property
    def d(self):
        return self.embedding_length

    @property
    def skipgram_pattern(self):
        return self._config.skipgram_pattern

    # endregion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_channels, N, d) -> (B, out_channels, N)
        features = self._projected_feature_layer(x).squeeze(dim=-1)

        # (B, in_channels, N, d) -> (B, out_channels, N)
        raw_attention = self._attention_layer(x).squeeze(dim=-1)
        # (B, out_channels, N) -> (B, out_channels, N)
        attention = self._softmax(raw_attention)

        # (B, out_channels, N)
        logits = attention * features

        # (B, out_channels, N) -> (B, out_channels)
        encoding = torch.sum(logits, dim=-1)

        return encoding
