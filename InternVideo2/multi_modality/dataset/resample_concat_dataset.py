import bisect
import warnings
import logging
from typing import (
    Iterable,
    List,
    TypeVar,
)

logger = logging.getLogger(__name__)

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

from torch.utils.data import Dataset, IterableDataset


class ResampleConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum_with_sample_weight(sequence, sample_weights):
        r, s = [], 0
        for i, e in enumerate(sequence):
            l = int(len(e) * sample_weights[i]) # NOTE
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset], sample_weights: List[int]) -> None:
        super(ResampleConcatDataset, self).__init__()
        
        self.datasets = list(datasets)
        self.sample_weights = sample_weights
        assert len(self.datasets) == len(self.sample_weights), f"{len(self.datasets)} != {len(self.sample_weights)}"
        logging.info(f"datasets: {self.datasets} sample weight: {self.sample_weights}")
        for i in range(len(self.sample_weights)):
            assert self.sample_weights[i] >= 1

        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ResampleConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum_with_sample_weight(self.datasets, self.sample_weights)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx // self.sample_weights[dataset_idx] # NOTE
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

