from typing import Self

import numpy as np
import pandas as pd

from pandas_weights.typing_ import D1NumericArray


class BaseWeightedAccessor[T: pd.Series | pd.DataFrame]:
    """
    Base class for weight accessors.

    Parameters
    ----------
    pandas_obj : Series or DataFrame
        The pandas object to which the accessor is attached.
    """

    def __init__(self, pandas_obj: T) -> None:
        self.obj = pandas_obj
        self._weights: pd.Series | None = None

    @classmethod
    def _init_validated(cls, pandas_obj: T, weights: pd.Series) -> Self:
        self = cls(pandas_obj)
        self._weights = weights

        return self

    @property
    def weights(self) -> pd.Series:
        if self._weights is None:
            raise ValueError("Weights have not been set. Set weights with `.wt(...)`.")
        return self._weights

    @weights.setter
    def weights(self, value: D1NumericArray) -> None:
        if len(value) != len(self.obj):
            raise ValueError("Length of weights must match number of rows in the data.")
        if isinstance(value, np.ndarray) and value.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        self._weights = pd.Series(value, index=self.obj.index)
