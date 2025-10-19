from typing import Self

import numpy as np
import pandas as pd


class BaseWeightedAccessor[T: pd.Series | pd.DataFrame]:
    _weights: pd.Series | None
    """
    Base class for weight accessors.

    Parameters
    ----------
    pandas_obj : Series or DataFrame
        The pandas object to which the accessor is attached.
    """

    def __init__(self, pandas_obj: T) -> None:
        self.obj = pandas_obj
        self._weights = None

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
    def weights(self, value: list[int | float] | pd.Series | np.ndarray) -> None:
        if len(value) != len(self.obj):
            raise ValueError("Length of weights must match number of rows in DataFrame")
        if isinstance(value, np.ndarray) and value.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        self._weights = pd.Series(value, index=self.obj.index)

    def weighted(self) -> T:
        """
        Return the weighted version of the underlying pandas object.

        Returns
        -------
        Series or DataFrame
            The weighted pandas object.

        Raises
        ------
        ValueError
            If weights have not been set.
        """
        return self.obj.mul(self.weights, axis=0)  # type: ignore[return-value]
