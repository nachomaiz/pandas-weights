from enum import Enum
from typing import TYPE_CHECKING, Hashable, Literal

import numpy as np
import pandas as pd
from pandas.core.groupby import SeriesGroupBy

from .accessor import register_accessor as _register_accessor

if TYPE_CHECKING:
    from pandas._typing import (
        AxisIndex,
        GroupByObjectNonScalar,
        Scalar,
        AggFuncType,
    )


class _NoDefault(Enum):
    no_default = object()


class Series(pd.Series):
    wt: "WeightedSeriesAccessor"


@_register_accessor("wt", pd.Series)
class WeightedSeriesAccessor:
    def __init__(self, pandas_obj: pd.Series) -> None:
        self.obj = pandas_obj
        self._weights: pd.Series | np.ndarray | None = None

    def __call__(
        self, weights: pd.Series | np.ndarray | list[float]
    ) -> "WeightedSeriesAccessor":
        if isinstance(weights, list):
            weights = np.array(weights)

        if not isinstance(weights, (pd.Series, np.ndarray)):
            raise ValueError("weights must be a pandas Series, numpy array, or list")

        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")

        if len(weights) != len(self.obj):
            raise ValueError("Length of weights must match number of rows in Series")

        self._weights = weights
        return self

    @classmethod
    def _init_weight(
        cls, obj: pd.Series, weights: pd.Series | np.ndarray
    ) -> "WeightedSeriesAccessor":
        accessor = cls(obj)
        accessor._weights = weights
        return accessor

    @property
    def T(self) -> Series:
        return Series(self.weighted().T)

    @property
    def weights(self) -> pd.Series | np.ndarray:
        if self._weights is None:
            raise ValueError("Weights have not been set. Set weights with `.wt(...)`.")
        return self._weights

    @weights.setter
    def weights(self, value: list[int | float] | pd.Series | np.ndarray) -> None:
        if len(value) != len(self.obj):
            raise ValueError("Length of weights must match number of rows in DataFrame")
        if isinstance(value, list):
            value = np.array(value)
        elif isinstance(value, np.ndarray) and value.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        self._weights = pd.Series(value, index=self.obj.index)

    def weighted(self) -> Series:
        """Return a DataFrame with the weights applied to the whole DataFrame."""
        return Series(self.obj.mul(self.weights))

    def groupby(
        self,
        by: "Scalar | GroupByObjectNonScalar | pd.MultiIndex | None" = None,
        axis: "AxisIndex | Literal[_NoDefault.no_default]" = _NoDefault.no_default,
        level: int | str | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool | Literal[_NoDefault.no_default] = _NoDefault.no_default,
        dropna: bool = True,
    ) -> "WeightedSeriesGroupBy":
        kwargs = {
            "keys": by,
            "level": level,
            "as_index": as_index,
            "sort": sort,
            "group_keys": group_keys,
            "dropna": dropna,
        }
        if axis is not _NoDefault.no_default:
            kwargs["axis"] = axis
        if observed is not _NoDefault.no_default:
            kwargs["observed"] = observed

        return WeightedSeriesGroupBy(self.weights, self.obj, **kwargs)

    def count(self, axis: "AxisIndex" = 0, skipna: bool = True) -> pd.Series:
        if skipna:
            weights = self.obj.notna().mul(self.weights, axis=0)
        else:
            weights = pd.Series(self.weights, index=self.obj.index).fillna(1.0)
        return weights.sum(axis=axis)

    def sum(self, axis: "AxisIndex" = 0, min_count: int = 0) -> pd.Series:
        return self.weighted().sum(axis=axis, min_count=min_count)

    def mean(self, axis: "AxisIndex" = 0, skipna: bool = True) -> pd.Series:
        return self.sum(axis=axis, min_count=1) / self.count(axis=axis, skipna=skipna)

    def var(
        self, axis: "AxisIndex" = 0, ddof: int = 1, skipna: bool = True
    ) -> pd.Series:
        sum_ = self.sum(axis=axis, min_count=1)
        count = self.count(axis=axis, skipna=skipna)
        diff = self.obj.sub(sum_ / count, axis=1 if axis == 0 else 0)
        diff_squared = diff.mul(diff)
        return diff_squared.sum(axis=axis) / (count - ddof)

    def std(
        self, axis: "AxisIndex" = 0, ddof: int = 1, skipna: bool = True
    ) -> pd.Series:
        return self.var(axis=axis, ddof=ddof, skipna=skipna).pow(0.5)

    def apply(
        self,
        func: "AggFuncType",
        axis: "AxisIndex" = 0,
        convertDType: bool = False,
        args: tuple = (),
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        return self.weighted().apply(  # type: ignore
            func,  # type: ignore
            axis=axis,
            convertDType=convertDType,
            args=args,
            **kwargs,
        )


class WeightedSeriesGroupBy(SeriesGroupBy):
    obj: Series
    _grouper: pd.Grouper

    def __init__(self, weights: pd.Series | np.ndarray, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = weights

    def _group_keys(
        self, group_cols: list[Hashable]
    ) -> list[Hashable] | list[tuple[Hashable, ...]]:
        if len(group_cols) == 1:
            return self.obj.index.get_level_values(group_cols[0]).tolist()  # type: ignore
        return pd.MultiIndex.from_frame(self.obj.reset_index()[group_cols]).tolist()

    def count(self, dropna: bool = True) -> pd.Series:
        if dropna:
            weights = self.obj.notna().mul(self.weights)
        else:
            weights = pd.Series(self.weights, index=self.obj.index).fillna(1.0)
        return weights.groupby(self._grouper).sum()

    def sum(self, min_count: int = 0) -> pd.Series:
        weighted_obj = self.obj.mul(self.weights)
        return weighted_obj.groupby(self._grouper).sum(min_count=min_count)

    def mean(self, skipna: bool = True) -> pd.Series:
        return self.sum(min_count=1) / self.count(dropna=skipna)

    def var(self, ddof: int = 1, skipna: bool = True) -> pd.Series:
        sum_ = self.sum(min_count=1)
        count = self.count(dropna=skipna)
        diff = self.obj.sub((sum_ / count).loc[self.obj.index], axis=0)
        diff_squared = diff.mul(diff)
        weighted_diff_squared = diff_squared.mul(self.weights)
        return weighted_diff_squared.groupby(self._grouper).sum() / (count - ddof)

    def std(self, ddof: int = 1, skipna: bool = True) -> pd.Series:
        return self.var(ddof=ddof, skipna=skipna).pow(0.5)
