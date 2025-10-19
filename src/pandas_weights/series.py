from enum import Enum
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import pandas as pd
from pandas.core.groupby import SeriesGroupBy
from pandas.core.groupby.ops import BaseGrouper

from pandas_weights.base import BaseWeightedAccessor

from .accessor import register_accessor as _register_accessor

if TYPE_CHECKING:
    from pandas._typing import (
        AggFuncType,
        AxisIndex,
        GroupByObjectNonScalar,
        Scalar,
    )


class _NoDefault(Enum):
    no_default = object()


class Series(pd.Series):
    wt: "WeightedSeriesAccessor"


@_register_accessor("wt", pd.Series)
class WeightedSeriesAccessor(BaseWeightedAccessor[pd.Series]):
    def __call__(self, weights: pd.Series | np.ndarray | list[int | float]) -> Self:
        self.weights = weights
        return self

    @property
    def T(self) -> Series:
        return Series(self.weighted().T)

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
    _grouper: BaseGrouper
    obj: Series

    def __init__(self, weights: pd.Series | np.ndarray, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = weights

    def _group_keys(self) -> pd.Index | pd.MultiIndex:
        if len(names := self._grouper.names) == 1:
            return pd.Index(self.obj.reset_index()[names[0]])
        return pd.MultiIndex.from_frame(self.obj.reset_index()[names])

    def count(self, dropna: bool = True) -> pd.Series:
        if dropna:
            weights = self.obj.notna().mul(self.weights)
        else:
            weights = pd.Series(self.weights, index=self.obj.index).fillna(1.0)
        return weights.groupby(self._grouper).sum()  # type: ignore[return-value]

    def sum(self, min_count: int = 0) -> pd.Series:
        weighted_obj = self.obj.mul(self.weights)
        return weighted_obj.groupby(self._grouper).sum(min_count=min_count)  # type: ignore[return-value]

    def mean(self, skipna: bool = True) -> pd.Series:
        return self.sum(min_count=1) / self.count(dropna=skipna)

    def var(self, ddof: int = 1, skipna: bool = True) -> pd.Series:
        sum_ = self.sum(min_count=1)
        count = self.count(dropna=skipna)
        diff = self.obj.sub((sum_ / count).loc[self._group_keys()], axis=0)
        diff_squared = diff.mul(diff)
        weighted_diff_squared = diff_squared.mul(self.weights)
        return weighted_diff_squared.groupby(self._grouper).sum() / (count - ddof)  # type: ignore[return-value]

    def std(self, ddof: int = 1, skipna: bool = True) -> pd.Series:
        return self.var(ddof=ddof, skipna=skipna).pow(0.5)
