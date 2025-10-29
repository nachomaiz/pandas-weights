from typing import Generic, TypeVar, Union

import numpy as np
import pandas as pd

from pandas_weights.typing_ import D1NumericArray

T = TypeVar("T", pd.Series, pd.DataFrame)
A = TypeVar("A", bound="BaseWeightedAccessor")


class BaseWeightedAccessor(Generic[T]):
    """
    Base class for weight accessors.

    Parameters
    ----------
    pandas_obj : Series or DataFrame
        The pandas object to which the accessor is attached.
    """

    def __init__(self, pandas_obj: T) -> None:
        self.obj = pandas_obj
        self._weights: Union[pd.Series, None] = None

    @classmethod
    def _init_validated(cls: type[A], pandas_obj: T, weights: pd.Series) -> A:
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
            raise ValueError("Weights must be one-dimensional")

        if isinstance(value, pd.Series):
            self._weights = value.reindex(self.obj.index)
        else:
            self._weights = pd.Series(value, index=self.obj.index)
