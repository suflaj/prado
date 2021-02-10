import copy
import string
from typing import List, Optional, Tuple

import nltk
import numpy as np

from . import assertion
from . import constants


# region Text Transformations
class SpaceOutDecorators:
    def __call__(self, x: str) -> str:
        return constants.DECORATOR_GROUP_REGEX.sub(r" \1 ", x)


class DoubleQuotesToSingle:
    def __call__(self, x: str) -> str:
        return constants.DOUBLE_QUOTE_REGEX.sub("'", x)


class UnquoteText:
    def __call__(self, x: str) -> str:
        return constants.QUOTED_TEXT_REGEX.sub(r" \1 ", x)


class LowerText:
    def __call__(self, x: str) -> str:
        return str(x).lower()


class WordTokenize:
    def __call__(self, x: str) -> List[str]:
        return nltk.tokenize.word_tokenize(x)


class FilterMeaninglessTokens:
    def __call__(self, x: List[str]) -> List[str]:
        return [y for y in x if constants.MEANINGLESS_REGEX.fullmatch(y) is None]


class BasicPradoTransform:
    def __init__(self):
        self._sequence = [
            SpaceOutDecorators(),
            DoubleQuotesToSingle(),
            UnquoteText(),
            LowerText(),
            WordTokenize(),
            FilterMeaninglessTokens(),
        ]

    def __call__(self, x: str) -> List[str]:
        for transform in self._sequence:
            x = transform(x)

        if len(x) == 0:
            x = [""]

        return x


# endregion


# region Stochastic Augmentation
class StochasticAugmentation:
    def _check_init_args(probability: float):
        assertion.assert_type("probability", probability, float)

        assertion.assert_in_range(
            "probability", probability, (0.0, 1.0), inclusivity=(True, True)
        )

        return probability

    def __init__(self, probability: float):
        probability = StochasticAugmentation._check_init_args(probability=probability)

        self._probability = probability

    # region Properties
    @property
    def probability(self) -> float:
        return self._probability

    @property
    def p(self) -> float:
        return self.probability

    # endregion

    def _roll(self) -> bool:
        return np.random.uniform(low=0.0, high=1.0) < self.probability


class StochasticInsertionAugmentation(StochasticAugmentation):
    @staticmethod
    def _check_init_args(alphabet: Optional[str]):
        if alphabet is None:
            alphabet = string.ascii_lowercase

        assertion.assert_type("alphabet", alphabet, str)

        assertion.assert_not_empty("alphabet", alphabet)

        return alphabet

    def __init__(self, probability: float, alphabet: Optional[str] = None):
        super().__init__(probability=probability)

        alphabet = StochasticInsertionAugmentation._check_init_args(alphabet=alphabet)

        self._alphabet = copy.deepcopy(alphabet)

    # region Properties
    @property
    def alphabet(self):
        return self._alphabet

    # endregion

    def __call__(self, x: str) -> str:
        if self._roll():
            char_to_insert = self.alphabet[np.random.randint(len(self.alphabet))]

            if len(x) == 0:
                return char_to_insert

            breaking_point = int(np.random.randint(len(x)))

            return x[:breaking_point] + char_to_insert + x[breaking_point:]

        return x


class StochasticDeletionAugmentation(StochasticAugmentation):
    def __init__(self, probability: float):
        super().__init__(probability=probability)

    def __call__(self, x: str) -> str:
        if len(x) != 0 and self._roll():
            if len(x) == 1:
                return ""
            else:
                index_to_remove = np.random.randint(len(x))

                return x[:index_to_remove] + x[index_to_remove + 1 :]

        return x


class StochasticSwapAugmentation(StochasticAugmentation):
    def __init__(self, probability: float):
        super().__init__(probability=probability)

    def __call__(self, x: str) -> str:
        if len(x) > 1 and self._roll():
            indices_to_swap = np.random.choice(2, 2, replace=False)

            t = list(x)
            t[indices_to_swap[0]], t[indices_to_swap[1]] = (
                t[indices_to_swap[1]],
                t[indices_to_swap[0]],
            )

            return "".join(t)

        return x


class BasicPradoAugmentation:
    @staticmethod
    def _check_init_args(
        insertion_probability: Optional[float],
        deletion_probability: Optional[float],
        swap_probability: Optional[float],
    ):
        if insertion_probability is None:
            insertion_probability = 0.0

        if deletion_probability is None:
            deletion_probability = 0.0

        if swap_probability is None:
            swap_probability = 0.0

        assertion.assert_type("insertion_probability", insertion_probability, float)
        assertion.assert_type("deletion_probability", deletion_probability, float)
        assertion.assert_type("swap_probability", swap_probability, float)

        return insertion_probability, deletion_probability, swap_probability

    def __init__(
        self,
        insertion_probability: Optional[float] = None,
        deletion_probability: Optional[float] = None,
        swap_probability: Optional[float] = None,
    ):
        (
            insertion_probability,
            deletion_probability,
            swap_probability,
        ) = BasicPradoAugmentation._check_init_args(
            insertion_probability=insertion_probability,
            deletion_probability=deletion_probability,
            swap_probability=swap_probability,
        )

        self._sequence = [
            StochasticInsertionAugmentation(probability=insertion_probability),
            StochasticDeletionAugmentation(probability=deletion_probability),
            StochasticSwapAugmentation(probability=swap_probability),
        ]

    def __call__(self, x: str) -> str:
        for transform in self._sequence:
            x = transform(x)

        return x


# endregion
