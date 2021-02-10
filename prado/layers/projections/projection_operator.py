import torch
from torch import nn


class ProjectionOperator:
    pass


class PradoProjectionOperator(nn.Module, ProjectionOperator):
    def __init__(self):
        super().__init__()

        self._linear = nn.Linear(in_features=2, out_features=1, bias=False)

        # We set the weights explicitly so we get the mapping:
        #   00 ->  0
        #   01 ->  1
        #   10 -> -1
        #   11 ->  0
        # Also, we automatically do float computation since we'll
        # need floats later, and it's efficient on the GPU.
        with torch.no_grad():
            self._linear.weight = nn.Parameter(
                torch.tensor(
                    [[-1, 1]],
                    dtype=torch.float,
                )
            )

        # The paper doesn't use learnable projection operators
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self._linear(x)
