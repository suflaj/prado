import copy
import sys
from typing import Any, Callable, Dict

from torch.utils.data import Dataset
from tqdm import tqdm

from . import assertion


class ProcessedDataset(Dataset):
    @staticmethod
    def _check_init_args(
        original_dataset: Dataset,
        transformation_map: Dict[int, Callable[[str], Any]],
        verbosity: int,
    ):
        assertion.assert_type("original_dataset", original_dataset, Dataset)
        assertion.assert_type("transformation_map", transformation_map, dict)
        assertion.assert_type("verbosity", verbosity, int)

        assertion.assert_all_int("transfomation_map keys", transformation_map.keys())
        assertion.assert_all_non_negative(
            "transformation_map keys", transformation_map.keys()
        )
        assertion.assert_all_callable(
            "transformation_map values", transformation_map.values()
        )

        return original_dataset, transformation_map, verbosity

    def __init__(
        self,
        original_dataset: Dataset,
        transformation_map: Dict[int, Callable[[str], Any]],
        verbosity: int = 0,
    ):
        super().__init__()

        dataset, transformation_map, verbosity = ProcessedDataset._check_init_args(
            original_dataset=original_dataset,
            transformation_map=transformation_map,
            verbosity=verbosity,
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
            new_content = list(row)

            for index, function in transformation_map.items():
                new_content[index] = function(new_content[index])

            self._content.append(tuple(new_content))

    def __getitem__(self, key):
        return self._content[key]

    def __len__(self):
        return len(self._content)

    def __repr__(self):
        return f"ProcessedDataset(len={len(self)})"

    def __str__(self):
        return repr(self)
