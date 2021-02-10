import copy
from typing import Optional

import torch
from torch import nn

from masked_convolution import MaskedConvolution


class SkipgramConvolutionConfig:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_length: int,
        skipgram_pattern,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_length = embedding_length
        self.skipgram_pattern = skipgram_pattern


class SkipgramConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_length: int,
        skipgram_pattern: str,
        config: Optional[SkipgramConvolutionConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = SkipgramConvolutionConfig(
                in_channels=in_channels,
                out_channels=out_channels,
                embedding_length=embedding_length,
                skipgram_pattern=skipgram_pattern,
            )

        self._config = copy.deepcopy(config)

        with torch.no_grad():
            self._convolution_mask = torch.ones(
                (len(self.skipgram_pattern), self.d), dtype=torch.float32
            )

            for i, char in enumerate(skipgram_pattern):
                if char == "0":
                    self._convolution_mask[i] *= 0

        self._conv = MaskedConvolution(
            conv_layer=nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self._convolution_mask.shape,
                stride=1,
                padding=(self._convolution_mask.shape[0] // 2, 0),
                bias=True,
                padding_mode="zeros",
            ),
            mask=self._convolution_mask,
        )

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
        # (B, in_features, N, d) -> (B, out_features, N, d)
        return self._conv(x)
