from collections.abc import Hashable, Iterator
from enum import Enum
from typing import TYPE_CHECKING, Callable, Literal, Self, overload

import numpy as np
import pandas as pd
from pandas.core.groupby import SeriesGroupBy

from pandas_weights.base import BaseWeightedAccessor
from pandas_weights.typing_ import D1NumericArray

if TYPE_CHECKING:
    from pandas._typing import (
        AggFuncType,
        AxisIndex,
        GroupByObjectNonScalar,
        Scalar,
    )

    from pandas_weights.frame import DataFrame


class _NoDefault(Enum):
    no_default = object()


class Series(pd.Series):
    wt: "WeightedSeriesAccessor"


@pd.api.extensions.register_series_accessor("wt")
class WeightedSeriesAccessor(BaseWeightedAccessor[Series]):
    def __call__(self, weights: list[int | float] | pd.Series | np.ndarray, /) -> Self:
        self.weights = weights
        return self

    def weighted(self) -> Series:
        return self.obj.mul(self.weights)  # type: ignore[return-value]

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

    def count(self, axis: "AxisIndex" = 0, skipna: bool = True) -> float:
        if skipna:
            weights = self.obj.notna().mul(self.weights, axis=0)
        else:
            weights = pd.Series(self.weights, index=self.obj.index).fillna(1.0)
        return weights.sum(axis=axis)

    def sum(self, axis: "AxisIndex" = 0, min_count: int = 0) -> float:
        return self.weighted().sum(axis=axis, min_count=min_count)

    def mean(self, axis: "AxisIndex" = 0, skipna: bool = True) -> float:
        return self.sum(axis=axis, min_count=1) / self.count(axis=axis, skipna=skipna)

    def var(self, axis: "AxisIndex" = 0, ddof: int = 1, skipna: bool = True) -> float:
        sum_ = self.sum(axis=axis, min_count=1)
        count = self.count(axis=axis, skipna=skipna)
        diff = self.obj.sub(sum_ / count)
        diff_squared = diff.mul(diff)
        return diff_squared.sum(axis=axis) / (count - ddof)

    def std(self, axis: "AxisIndex" = 0, ddof: int = 1, skipna: bool = True) -> float:
        return self.var(axis=axis, ddof=ddof, skipna=skipna) ** 0.5

    @overload
    def apply(
        self,
        func: Callable[..., "Scalar"],
        args: tuple = ...,
        **kwargs,
    ) -> "Series": ...
    @overload
    def apply(
        self,
        func: Callable[..., D1NumericArray],
        args: tuple = ...,
        **kwargs,
    ) -> "DataFrame": ...
    def apply(
        self,
        func: "AggFuncType",
        args: tuple = (),
        **kwargs,
    ) -> "Series | DataFrame":
        return self.weighted().apply(
            func,  # type: ignore
            args=args,
            **kwargs,
        )


class WeightedSeriesGroupBy:
    def __init__(self, weights: pd.Series, *args, **kwargs) -> None:
        self._groupby = SeriesGroupBy(*args, **kwargs)
        self.weights = weights

    @classmethod
    def _init_groupby(cls, weights: pd.Series, groupby: SeriesGroupBy) -> Self:
        obj = cls.__new__(cls)
        obj._groupby = groupby
        obj.weights = weights
        return obj

    def __iter__(self) -> Iterator[tuple[Hashable, WeightedSeriesAccessor]]:
        weights_groupby: SeriesGroupBy = self.weights.groupby(self._groupby._grouper)
        for (key, group), (_, group_weights) in zip(self._groupby, weights_groupby):
            yield (
                key,
                WeightedSeriesAccessor._init_validated(
                    group,  # type: ignore[arg-type]
                    group_weights,
                ),
            )

    def _group_keys(self) -> pd.Index | pd.MultiIndex:
        if len(names := self._groupby._grouper.names) == 1:
            return pd.Index(self._groupby.obj.reset_index()[names[0]])
        return pd.MultiIndex.from_frame(self._groupby.obj.reset_index()[names])

    def count(self, skipna: bool = True) -> Series:
        if skipna:
            weights = self._groupby.obj.notna().mul(self.weights)
        else:
            weights = self.weights.fillna(1.0)
        return weights.groupby(self._groupby._grouper).sum()  # type: ignore[arg-type]

    def sum(self, min_count: int = 0) -> Series:
        weighted = self._groupby.obj.mul(self.weights)  # type: ignore
        return weighted.groupby(self._groupby._grouper).sum(min_count=min_count)  # type: ignore[arg-type]

    def mean(self, skipna: bool = True) -> Series:
        return self.sum(min_count=1) / self.count(skipna=skipna)  # type: ignore[return-value]

    def var(self, ddof: int = 1, skipna: bool = True) -> Series:
        weighted = self._groupby.obj.mul(self.weights)
        sum_ = self.sum(min_count=1)
        count = self.count(skipna=skipna)
        diff = weighted.sub(
            (sum_ / count).loc[self._group_keys()].set_axis(self._groupby.obj.index)
        )
        diff_squared = diff.mul(diff)
        return diff_squared.groupby(self._groupby._grouper).sum() / (count - ddof)  # type: ignore[arg-type]

    def std(self, ddof: int = 1, skipna: bool = True) -> Series:
        return self.var(ddof=ddof, skipna=skipna).pow(0.5)  # type: ignore[return-value]

    def apply(self, func: "AggFuncType", *args, **kwargs) -> Series:
        return (
            self._groupby.obj.mul(self.weights)
            .groupby(self._groupby._grouper)  # type: ignore[arg-type]
            .apply(func, *args, **kwargs)
        )
