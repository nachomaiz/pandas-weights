import datetime as dt
from collections.abc import Hashable, Iterator
from enum import Enum
from typing import TYPE_CHECKING, Callable, Literal, Optional, overload

import numpy as np
import pandas as pd
from pandas.core.groupby import SeriesGroupBy

from pandas_weights._stats import (
    weighted_correlation,
    variance_from_weighted_moments,
    weighted_sum_of_squares,
)
from pandas_weights.base import BaseWeightedAccessor
from pandas_weights.typing_ import D1NumericArray, Number

if TYPE_CHECKING:
    from pandas._typing import (
        AggFuncType,
        AxisIndex,
        GroupByObjectNonScalar,
        Level,
        Scalar,
        Frequency,
        TimedeltaConvertibleTypes,
        TimeGrouperOrigin,
        TimestampConvertibleTypes,
    )
    from pandas.core.resample import Resampler

    from pandas_weights.frame import DataFrame


class _NoDefault(Enum):
    no_default = object()


class Series(pd.Series):
    wt: "WeightedSeriesAccessor"


@pd.api.extensions.register_series_accessor("wt")
class WeightedSeriesAccessor(BaseWeightedAccessor[Series]):
    """Series Weights Accessor

    Initialize by calling the accessor with an array-like of weights.

    >>> s.wt([0.1, 0.5, 0.4, ...])  # array-like of weights

    Attributes
    ----------
    weights : Series
        The weights associated with the Series.

    Methods
    -------
    weighted() -> Series
        Get the Series with applied weights.
    groupby(...) -> WeightedSeriesGroupBy
        Perform a weighted groupby operation on the Series.
    count(...) -> float
        Count observations weighted by the weights.
    sum(...) -> float
        Sum of values weighted by the weights.
    mean(...) -> float
        Mean of values weighted by the weights.
    var(...) -> float
        Variance of values weighted by the weights.
    std(...) -> float
        Standard deviation of values weighted by the weights.
    apply(func, ..., **kwargs) -> Series or DataFrame
        Apply a function to each element of the Series.
    """

    def __call__(
        self, weights: D1NumericArray, /, na_weight: Optional[Number] = None
    ) -> "WeightedSeriesAccessor":
        """Set weights for the Series

        Parameters
        ----------
        weights : D1NumericArray
            Array of weights
        na_weight : Number, optional
            Weight to fill missing weight values, by default None

        Returns
        -------
        WeightedSeriesAccessor
            Initialized Series Weights Accessor
        """
        self._weights = pd.Series(weights, index=self.obj.index)
        if na_weight is not None:
            self._weights = self._weights.fillna(na_weight)

        return self

    def weighted(self) -> Series:
        """Get the Series with applied weights

        Returns
        -------
        Series
            Series with applied weights
        """
        return self.obj.mul(self.weights)  # type: ignore[return-value]

    def groupby(
        self,
        by: "Scalar | GroupByObjectNonScalar | pd.MultiIndex | None" = None,
        level: Optional["Level"] = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool | Literal[_NoDefault.no_default] = _NoDefault.no_default,
        dropna: bool = True,
    ) -> "WeightedSeriesGroupBy":
        """Perform a weighted groupby operation on the Series.

        See `pandas.Series.groupby` for more details on the parameters.
        """
        kwargs = {
            "keys": by,
            "level": level,
            "as_index": as_index,
            "sort": sort,
            "group_keys": group_keys,
            "dropna": dropna,
        }
        if observed is not _NoDefault.no_default:
            kwargs["observed"] = observed

        return WeightedSeriesGroupBy(self.weights, self.obj, **kwargs)

    def resample(
        self,
        rule: "Frequency | dt.timedelta",
        closed: Literal["right", "left"] | None = None,
        label: Literal["right", "left"] | None = None,
        on: "Level | None" = None,
        level: "Level | None" = None,
        origin: "TimeGrouperOrigin | TimestampConvertibleTypes" = "start_day",
        offset: "TimedeltaConvertibleTypes | None" = None,
        group_keys: bool = False,
    ) -> "WeightedSeriesResampler":
        """Perform a weighted resample operation on the Series.

        See `pandas.Series.resample` for more details on the parameters.
        """
        return WeightedSeriesResampler(
            self.obj,
            self.weights,
            rule,
            closed=closed,
            label=label,
            on=on,
            level=level,
            origin=origin,
            offset=offset,
            group_keys=group_keys,
        )

    def count(self, axis: "AxisIndex" = 0, skipna: bool = True) -> float:
        """Count observations weighted by the weights

        See `pandas.Series.count` for more details on the parameters.

        If `skipna` is `True`, missing values in the Series are not counted towards
        the total weight for that element.

        If `skipna` is `False`, missing values are treated as having a count of 1,
        and weighted by the weights if available,
        but missing weights are still treated as 0.
        If including missing weights is desired, fill missing weights using
        `series.wt(..., na_weight=1)`.
        """
        if skipna:
            weights = self.obj.notna().mul(self.weights, axis=0)
        else:
            weights = pd.Series(self.weights, index=self.obj.index)
        return weights.sum(axis=axis)

    def sum(self, axis: "AxisIndex" = 0, min_count: int = 0) -> float:
        """Sum observations weighted by the weights

        See `pandas.Series.sum` for more details on the parameters.
        """

        return self.weighted().sum(axis=axis, min_count=min_count)

    def mean(self, axis: "AxisIndex" = 0, skipna: bool = True) -> float:
        """Calculate the mean of observations weighted by the weights

        See `pandas.Series.mean` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        return self.sum(axis=axis, min_count=1) / self.count(axis=axis, skipna=skipna)

    def var(self, axis: "AxisIndex" = 0, ddof: int = 1, skipna: bool = True) -> float:
        """Calculate the variance of observations weighted by the weights

        See `pandas.Series.var` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        sum_ = self.sum(axis=axis, min_count=1)
        count = self.count(axis=axis, skipna=skipna)
        sum_squared = weighted_sum_of_squares(self.obj, self.weights, axis=axis)
        return variance_from_weighted_moments(sum_, sum_squared, count, ddof=ddof)  # type: ignore[return-value]

    def std(self, axis: "AxisIndex" = 0, ddof: int = 1, skipna: bool = True) -> float:
        """Calculate the standard deviation of observations weighted by the weights

        See `pandas.Series.std` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        return np.sqrt(self.var(axis=axis, ddof=ddof, skipna=skipna))

    def corr(
        self,
        other: pd.Series,
        method: Literal["pearson"] = "pearson",
        min_periods: int | None = None,
        ddof: int = 1,
    ) -> float:
        """Weighted correlation with another Series.

        This method currently supports weighted Pearson correlation.
        """
        if method != "pearson":
            raise NotImplementedError(
                "Only 'pearson' weighted correlation is supported."
            )

        left, right = self.obj.align(other, join="inner")
        left_weights, _ = self.weights.align(other, join="inner")

        return weighted_correlation(
            left,
            right,
            left_weights,
            ddof=ddof,
            min_periods=1 if min_periods is None else min_periods,
        )

    @overload
    def apply(
        self, func: Callable[..., "Scalar"], args: tuple = ..., **kwargs
    ) -> Series: ...
    @overload
    def apply(
        self, func: Callable[..., D1NumericArray], args: tuple = ..., **kwargs
    ) -> "DataFrame": ...
    def apply(
        self, func: "AggFuncType", args: tuple = (), **kwargs
    ) -> "Series | DataFrame":
        """Apply a function to observations weighted by the weights

        See `pandas.Series.apply` for more details on the parameters.
        """
        return self.weighted().apply(func, args=args, **kwargs)  # type: ignore


class WeightedSeriesResampler:
    def __init__(self, obj: pd.Series, weights: pd.Series, rule, *args, **kwargs):
        self._obj = obj
        self.weights = weights
        self._rule = rule
        self._args = args
        self._kwargs = kwargs

    def _resample(self, obj: pd.Series) -> "Resampler":
        return obj.resample(self._rule, *self._args, **self._kwargs)

    def count(self, skipna: bool = True) -> Series:
        """Count resampled observations weighted by the weights.

        See `pandas.Series.resample.count` for more details on the parameters.

        If `skipna` is `True`, missing values in the Series are not counted towards
        the total weight for that element.

        If `skipna` is `False`, missing values are treated as having a count of 1,
        and weighted by the weights if available,
        but missing weights are still treated as 0.
        If including missing weights is desired, fill missing weights using
        `series.wt(..., na_weight=1)`.
        """
        if skipna:
            weights = self._obj.notna().mul(self.weights, axis=0)
        else:
            weights = pd.Series(self.weights, index=self._obj.index)
        return self._resample(weights).sum().rename(self._obj.name)  # type: ignore[return-value]

    def sum(self, min_count: int = 0) -> Series:
        """Sum of resampled values weighted by the weights.

        See `pandas.Series.resample.sum` for more details on the parameters.
        """
        weighted = self._obj.mul(self.weights)
        return self._resample(weighted).sum(min_count=min_count).rename(self._obj.name)

    def mean(self, skipna: bool = True) -> Series:
        """Mean of resampled values weighted by the weights.

        See `pandas.Series.resample.mean` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        return self.sum(min_count=1) / self.count(skipna=skipna)  # type: ignore[return-value]

    def var(self, ddof: int = 1, skipna: bool = True) -> Series:
        """Calculate the variance of observations weighted by the weights

        See `pandas.Series.resample.var` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        sum_ = self.sum(min_count=1)
        count = self.count(skipna=skipna)
        sum_squared = self._resample(self._obj.pow(2).mul(self.weights)).sum(
            min_count=1
        )
        return variance_from_weighted_moments(
            sum_, sum_squared, count, ddof=ddof
        ).rename(self._obj.name)  # type: ignore[return-value]

    def std(self, ddof: int = 1, skipna: bool = True) -> Series:
        """Calculate the standard deviation of observations weighted by the weights

        See `pandas.Series.resample.std` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        return np.sqrt(self.var(ddof=ddof, skipna=skipna))  # type: ignore[return-value]


class WeightedSeriesGroupBy:
    def __init__(self, weights: pd.Series, *args, **kwargs) -> None:
        self._groupby = SeriesGroupBy(*args, **kwargs)
        self.weights = weights

    @classmethod
    def _init_groupby(
        cls, weights: pd.Series, groupby: SeriesGroupBy
    ) -> "WeightedSeriesGroupBy":
        obj = cls.__new__(cls)
        obj._groupby = groupby
        obj.weights = weights
        return obj

    def __iter__(self) -> Iterator[tuple[Hashable, WeightedSeriesAccessor]]:
        weights_groupby: SeriesGroupBy = self.weights.groupby(self._groupby._grouper)
        for (key, group), (_, group_weights) in zip(self._groupby, weights_groupby):
            yield (key, WeightedSeriesAccessor._init_validated(group, group_weights))

    def _group_keys(self) -> pd.Index | pd.MultiIndex:
        if len(names := self._groupby._grouper.names) == 1:
            return pd.Index(self._groupby.obj.reset_index()[names[0]])
        return pd.MultiIndex.from_frame(self._groupby.obj.reset_index()[names])

    def count(self, skipna: bool = True) -> Series:
        """Count observations in each group weighted by the weights

        See `pandas.SeriesGroupBy.count` for more details on the parameters.

        If `skipna` is `True`, missing values in the Series are not counted towards
        the total weight for that element.

        If `skipna` is `False`, missing values are treated as having a count of 1,
        and weighted by the weights if available,
        but missing weights are still treated as 0.
        If including missing weights is desired, fill missing weights using
        `series.wt(..., na_weight=1)`.
        """
        if skipna:
            weights = self._groupby.obj.notna().mul(self.weights)
        else:
            weights = self.weights
        return (
            weights.groupby(self._groupby._grouper).sum().rename(self._groupby.obj.name)
        )  # type: ignore[arg-type,return-value]

    def sum(self, min_count: int = 0) -> Series:
        """Sum observations in each group weighted by the weights

        See `pandas.SeriesGroupBy.sum` for more details on the parameters.
        """
        weighted = self._groupby.obj.mul(self.weights)  # type: ignore[arg-type]
        return (
            weighted.groupby(self._groupby._grouper)
            .sum(min_count=min_count)
            .rename(self._groupby.obj.name)
        )  # type: ignore[arg-type]

    def mean(self, skipna: bool = True) -> Series:
        """Calculate the mean of observations in each group weighted by the weights

        See `pandas.SeriesGroupBy.mean` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        return self.sum(min_count=1) / self.count(skipna=skipna)  # type: ignore[return-value]

    def var(self, ddof: int = 1, skipna: bool = True) -> Series:
        """Calculate the variance of observations in each group weighted by the weights

        See `pandas.SeriesGroupBy.var` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        sum_ = self.sum(min_count=1)
        count = self.count(skipna=skipna)
        sum_squared = (
            self._groupby.obj.pow(2)
            .mul(self.weights)
            .groupby(self._groupby._grouper)
            .sum(min_count=1)
            .rename(self._groupby.obj.name)
        )
        return variance_from_weighted_moments(sum_, sum_squared, count, ddof=ddof)  # type: ignore[return-value]

    def std(self, ddof: int = 1, skipna: bool = True) -> Series:
        """Calculate the standard deviation of observations in each group weighted by the weights

        See `pandas.SeriesGroupBy.std` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        return np.sqrt(self.var(ddof=ddof, skipna=skipna))  # type: ignore[return-value]

    def corr(
        self,
        other: pd.Series,
        method: Literal["pearson"] = "pearson",
        min_periods: int | None = None,
        ddof: int = 1,
    ) -> Series:
        """Weighted correlation of each group with another Series.

        This method currently supports weighted Pearson correlation.
        """
        if method != "pearson":
            raise NotImplementedError(
                "Only 'pearson' weighted correlation is supported."
            )

        result = {}
        for key, group in self:
            left, right = group.obj.align(other, join="inner")
            left_weights, _ = group.weights.align(other, join="inner")
            result[key] = weighted_correlation(
                left,
                right,
                left_weights,
                ddof=ddof,
                min_periods=1 if min_periods is None else min_periods,
            )
        out = pd.Series(result, name=self._groupby.obj.name)
        group_names = self._groupby._grouper.names
        if len(group_names) == 1:
            out.index = pd.Index(out.index, name=group_names[0])
        else:
            out.index = pd.MultiIndex.from_tuples(out.index, names=group_names)  # type: ignore[arg-type]
        return out  # type: ignore[return-value]

    def apply(self, func: "AggFuncType", *args, **kwargs) -> Series:
        """Apply a function to observations in each group weighted by the weights

        The function is applied to the weighted observations in each group.

        See `pandas.SeriesGroupBy.apply` for more details on the parameters.
        """
        return (
            self._groupby.obj.mul(self.weights)
            .groupby(self._groupby._grouper)  # type: ignore[arg-type,return-value]
            .apply(func, *args, **kwargs)
        )
