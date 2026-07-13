import datetime as dt
from collections.abc import Hashable, Iterator, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Callable, Literal, Optional, overload

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

from pandas_weights._stats import (
    weighted_correlation,
    variance_from_weighted_moments,
    weighted_sum_of_squares,
)
from pandas_weights.base import BaseWeightedAccessor
from pandas_weights.series import WeightedSeriesAccessor, WeightedSeriesGroupBy
from pandas_weights.typing_ import D1NumericArray, Number

if TYPE_CHECKING:
    from pandas._typing import (
        AggFuncType,
        Axis,
        GroupByObjectNonScalar,
        Level,
        Scalar,
        WindowingEngine,
        WindowingEngineKwargs,
        Frequency,
        TimedeltaConvertibleTypes,
        TimeGrouperOrigin,
        TimestampConvertibleTypes,
    )
    from pandas.core.resample import Resampler

    from pandas_weights.series import Series


class _NoDefault(Enum):
    no_default = object()


class DataFrame(pd.DataFrame):
    wt: "WeightedDataFrameAccessor"


@pd.api.extensions.register_dataframe_accessor("wt")
class WeightedDataFrameAccessor(BaseWeightedAccessor[DataFrame]):
    """DataFrame Weights Accessor

    Initialize by calling the accessor with the weights column name or array-like of weights.

    >>> df.wt("weights")  # "weights" column name within the DataFrame
    >>> df.wt([0.1, 0.5, 0.4, ...])  # array-like of weights

    Attributes
    ----------
    weights : Series
        The weights associated with the DataFrame.

    Methods
    -------
    weighted() -> DataFrame
        Get the DataFrame with applied weights.
    groupby(...) -> WeightedFrameGroupBy
        Perform a weighted groupby operation on the DataFrame.
    count(...) -> Series
        Count observations weighted by the weights.
    sum(...) -> Series
        Sum of values weighted by the weights.
    mean(...) -> Series
        Mean of values weighted by the weights.
    var(...) -> Series
        Variance of values weighted by the weights.
    std(...) -> Series
        Standard deviation of values weighted by the weights.
    apply(func, ..., **kwargs) -> Series or DataFrame
        Apply a function along the axis of the DataFrame.
    """

    def __call__(
        self,
        weights: Hashable | D1NumericArray,
        /,
        na_weight: Optional[Number] = None,
    ) -> "WeightedDataFrameAccessor":
        """Set weights for the DataFrame

        Parameters
        ----------
        weights : Hashable | D1NumericArray
            Weights column name within the DataFrame or array of weights
        na_weight : Number, optional
            Weight to fill missing weight values, by default None

        Returns
        -------
        WeightedDataFrameAccessor
            Initialized DataFrame Weights Accessor
        """
        if isinstance(weights, (list, pd.Series, np.ndarray)):
            self._weights = pd.Series(weights, index=self.obj.index)
        else:
            self._weights = self.obj[weights]  # we know it's the right length
            self.obj = self.obj.drop(columns=weights)

        if na_weight is not None:
            self._weights = self._weights.fillna(na_weight)

        return self

    @overload
    def __getitem__(self, key: Hashable) -> "WeightedSeriesAccessor": ...
    @overload
    def __getitem__(self, key: Sequence[Hashable]) -> "WeightedDataFrameAccessor": ...
    def __getitem__(
        self, key: Hashable | Sequence[Hashable]
    ) -> "WeightedDataFrameAccessor | WeightedSeriesAccessor":
        if isinstance(key, list):
            return WeightedDataFrameAccessor._init_validated(
                self.obj[key], self._weights
            )
        return WeightedSeriesAccessor._init_validated(self.obj[key], self._weights)

    def weighted(self) -> DataFrame:
        """Get the DataFrame with applied weights

        Returns
        -------
        DataFrame
            DataFrame with applied weights
        """
        return self.obj.mul(self.weights, axis=0)  # type: ignore[return-value]

    def groupby(
        self,
        by: "Scalar | GroupByObjectNonScalar | pd.MultiIndex | None" = None,
        level: Optional["Level"] = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool | Literal[_NoDefault.no_default] = _NoDefault.no_default,
        dropna: bool = True,
    ) -> "WeightedFrameGroupBy":
        """Perform a weighted groupby operation on the DataFrame.

        See `pandas.DataFrame.groupby` for more details on the parameters.
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

        return WeightedFrameGroupBy(self.weights, self.obj, **kwargs)

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
    ) -> "WeightedFrameResampler":
        """Perform a weighted resample operation on the DataFrame.

        See `pandas.DataFrame.resample` for more details on the parameters.
        """
        return WeightedFrameResampler(
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

    def count(self, axis: "Axis" = 0, skipna: bool = True) -> "Series":
        """Count observations weighted by the weights.

        See `pandas.DataFrame.count` for more details on the parameters.

        If `skipna` is `True`, missing values in the DataFrame are not counted towards
        the total weight for that row/column.

        If `skipna` is `False`, missing values are treated as having a count of 1,
        and weighted by the weights if available,
        but missing weights are still treated as 0.
        If including missing weights is desired, fill missing weights using
        `df.wt(..., na_weight=1)`.
        """
        obj = self.obj

        if skipna:
            weights = obj.notna().mul(self.weights, axis=0)
        else:
            weights = pd.DataFrame(
                np.broadcast_to(np.asarray(self.weights).reshape(-1, 1), obj.shape),
                index=obj.index,
                columns=obj.columns,
            )
        return weights.sum(axis=axis)  # type: ignore[return-value]

    def sum(self, axis: "Axis" = 0, min_count: int = 0) -> "Series":
        """Sum of values weighted by the weights.

        See `pandas.DataFrame.sum` for more details on the parameters.
        """
        return self.weighted().sum(axis=axis, min_count=min_count)  # type: ignore[return-value]

    def mean(self, axis: "Axis" = 0, skipna: bool = True) -> "Series":
        """Mean of values weighted by the weights.

        See `pandas.DataFrame.mean` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        return self.sum(axis=axis, min_count=1) / self.count(axis=axis, skipna=skipna)  # type: ignore[return-value]

    def var(self, axis: "Axis" = 0, ddof: int = 1, skipna: bool = True) -> "Series":
        """Variance of values weighted by the weights.

        See `pandas.DataFrame.var` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        sum_ = self.sum(axis=axis, min_count=1)
        count = self.count(axis=axis, skipna=skipna)
        sum_squared = weighted_sum_of_squares(self.obj, self.weights, axis=axis)
        return variance_from_weighted_moments(sum_, sum_squared, count, ddof=ddof)  # type: ignore[return-value]

    def std(self, axis: "Axis" = 0, ddof: int = 1, skipna: bool = True) -> "Series":
        """Standard deviation of values weighted by the weights.

        See `pandas.DataFrame.std` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """

        return np.sqrt(self.var(axis=axis, ddof=ddof, skipna=skipna))  # type: ignore[return-value]

    def corr(
        self,
        method: Literal["pearson"] = "pearson",
        min_periods: int = 1,
        ddof: int = 1,
    ) -> DataFrame:
        """Pairwise weighted correlation of columns.

        This method currently supports weighted Pearson correlation.
        """
        if method != "pearson":
            raise NotImplementedError(
                "Only 'pearson' weighted correlation is supported."
            )

        numeric = self.obj.select_dtypes(include=["number", "bool"])
        columns = numeric.columns
        result = pd.DataFrame(np.nan, index=columns, columns=columns, dtype=float)

        for i, left_col in enumerate(columns):
            for j in range(i, len(columns)):
                right_col = columns[j]
                corr = weighted_correlation(
                    numeric[left_col],
                    numeric[right_col],
                    self.weights,
                    ddof=ddof,
                    min_periods=min_periods,
                )
                result.loc[left_col, right_col] = corr
                result.loc[right_col, left_col] = corr

        return DataFrame(result)

    @overload
    def apply(
        self,
        func: Callable[..., "Scalar"],
        axis: "Axis" = ...,
        raw: bool = ...,
        result_type: Optional[Literal["reduce"]] = ...,
        args: tuple = ...,
        by_row: Literal[False, "compat"] = ...,
        engine: Literal["python", "numba"] = ...,
        engine_kwargs: Optional[dict[str, bool]] = ...,
        **kwargs,
    ) -> "Series": ...
    @overload
    def apply(
        self,
        func: Callable[..., "Scalar"],
        axis: "Axis" = ...,
        raw: bool = ...,
        result_type: Literal["expand", "broadcast"] = ...,
        args: tuple = ...,
        by_row: Literal[False, "compat"] = ...,
        engine: Literal["python", "numba"] = ...,
        engine_kwargs: Optional[dict[str, bool]] = ...,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def apply(
        self,
        func: Callable[..., D1NumericArray],
        axis: "Axis" = 0,
        raw: bool = ...,
        result_type: Literal["expand", "broadcast"] | None = ...,
        args: tuple = ...,
        by_row: Literal[False, "compat"] = ...,
        engine: Literal["python", "numba"] = ...,
        engine_kwargs: Optional[dict[str, bool]] = ...,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def apply(
        self,
        func: Callable[..., D1NumericArray],
        axis: "Axis" = 0,
        raw: bool = ...,
        result_type: Literal["reduce"] = "reduce",
        args: tuple = ...,
        by_row: Literal[False, "compat"] = ...,
        engine: Literal["python", "numba"] = ...,
        engine_kwargs: Optional[dict[str, bool]] = ...,
        **kwargs,
    ) -> "Series": ...
    def apply(
        self,
        func: "AggFuncType",
        axis: "Axis" = 0,
        raw: bool = False,
        result_type: Optional[Literal["expand", "reduce", "broadcast"]] = None,
        args: tuple = (),
        by_row: Literal[False, "compat"] = "compat",
        engine: Literal["python", "numba"] = "python",
        engine_kwargs: Optional[dict[str, bool]] = None,
        **kwargs,
    ) -> "Series | DataFrame":
        """Apply a function along the axis of the DataFrame.

        The function is applied on the weighted data.

        See `pandas.DataFrame.apply` for more details on the parameters.
        """
        return self.weighted().apply(  # type: ignore
            func,  # type: ignore
            axis=axis,
            raw=raw,
            result_type=result_type,  # type: ignore
            args=args,
            by_row=by_row,
            engine=engine,
            engine_kwargs=engine_kwargs,
            **kwargs,
        )


class WeightedFrameResampler:
    def __init__(self, obj: pd.DataFrame, weights: pd.Series, rule: "Frequency | dt.timedelta", *args, **kwargs):
        self._obj = obj
        self.weights = weights
        self._rule = rule
        self._args = args
        self._kwargs = kwargs

    def _resample(self, obj: pd.DataFrame) -> "Resampler":
        return obj.resample(self._rule, *self._args, **self._kwargs)

    def count(self, skipna: bool = True) -> DataFrame:
        """Count resampled observations weighted by the weights.

        See `pandas.DataFrame.resample.count` for more details on the parameters.

        If `skipna` is `True`, missing values in the DataFrame are not counted towards
        the total weight for that row/column.

        If `skipna` is `False`, missing values are treated as having a count of 1,
        and weighted by the weights if available,
        but missing weights are still treated as 0.
        If including missing weights is desired, fill missing weights using
        `df.wt(..., na_weight=1)`.
        """
        if skipna:
            weights = self._obj.notna().mul(self.weights, axis=0)
        else:
            weights = pd.DataFrame(
                np.broadcast_to(
                    np.asarray(self.weights).reshape(-1, 1), self._obj.shape
                ),
                index=self._obj.index,
                columns=self._obj.columns,
            )
        return self._resample(weights).sum()  # type: ignore[return-value]

    def sum(self, min_count: int = 0) -> DataFrame:
        """Sum of resampled values weighted by the weights.

        See `pandas.DataFrame.resample.sum` for more details on the parameters.
        """
        weighted = self._obj.mul(self.weights, axis=0)
        return self._resample(weighted).sum(min_count=min_count)  # type: ignore[return-value]

    def mean(self, skipna: bool = True) -> DataFrame:
        """Mean of resampled values weighted by the weights.

        See `pandas.DataFrame.resample.mean` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        return self.sum(min_count=1) / self.count(skipna=skipna)  # type: ignore[return-value]

    def var(self, ddof: int = 1, skipna: bool = True) -> DataFrame:
        """Variance of values weighted by the weights.

        See `pandas.DataFrame.resample.var` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        sum_ = self.sum(min_count=1)
        count = self.count(skipna=skipna)
        sum_squared = self._resample(self._obj.pow(2).mul(self.weights, axis=0)).sum(
            min_count=1
        )
        return variance_from_weighted_moments(sum_, sum_squared, count, ddof=ddof)  # type: ignore[return-value]

    def std(self, ddof: int = 1, skipna: bool = True) -> DataFrame:
        """Standard deviation of values weighted by the weights.

        See `pandas.DataFrame.resample.std` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """

        return np.sqrt(self.var(ddof=ddof, skipna=skipna))  # type: ignore[return-value]


class WeightedFrameGroupBy:
    def __init__(self, weights: pd.Series, *args, **kwargs) -> None:
        self._groupby = DataFrameGroupBy(*args, **kwargs)
        self.weights = weights

    @classmethod
    def _init_groupby(
        cls, weights: pd.Series, groupby: DataFrameGroupBy
    ) -> "WeightedFrameGroupBy":
        obj = cls.__new__(cls)
        obj._groupby = groupby
        obj.weights = weights
        return obj

    def __iter__(self) -> Iterator[tuple[Hashable, WeightedDataFrameAccessor]]:
        weights_groupby: SeriesGroupBy = self.weights.groupby(self._groupby._grouper)  # type: ignore[arg-type]
        for (key, group), (_, group_weights) in zip(self._groupby, weights_groupby):
            yield (key, WeightedDataFrameAccessor._init_validated(group, group_weights))

    @overload
    def __getitem__(self, key: Hashable) -> "WeightedSeriesGroupBy": ...
    @overload
    def __getitem__(self, key: Sequence[Hashable]) -> "WeightedFrameGroupBy": ...
    def __getitem__(
        self, key: Hashable | Sequence[Hashable]
    ) -> "WeightedFrameGroupBy | WeightedSeriesGroupBy":
        if isinstance(key, list):
            return WeightedFrameGroupBy._init_groupby(self.weights, self._groupby[key])
        return WeightedSeriesGroupBy._init_groupby(self.weights, self._groupby[key])  # type: ignore[arg-type]

    def _group_keys(self) -> pd.Index | pd.MultiIndex:
        if len(names := self._groupby._grouper.names) == 1:
            return pd.Index(self._groupby.obj.reset_index()[names[0]])
        return pd.MultiIndex.from_frame(self._groupby.obj.reset_index()[names])

    def _broadcast_weights(self, skipna: bool = True) -> DataFrame:
        obj = self._groupby._selected_obj.drop(
            columns=self._groupby.exclusions, errors="ignore"
        )  # type: ignore[return-value]
        if skipna:
            return obj.notna().mul(self.weights, axis=0)  # type: ignore[arg-type,return-value]
        return DataFrame(
            np.broadcast_to(np.asarray(self.weights).reshape(-1, 1), obj.shape),
            index=self._groupby.obj.index,
            columns=obj.columns,
        )

    def _numeric_columns(self) -> pd.Index:
        return (
            self._groupby._selected_obj.drop(
                columns=self._groupby.exclusions, errors="ignore"
            )
            .select_dtypes(include=["number", "bool"])
            .columns
        )

    def _weighted(self, numeric_cols: Optional[pd.Index] = None) -> DataFrame:
        weighted: pd.DataFrame = self._groupby._selected_obj.copy()  # type: ignore[assignment]
        if numeric_cols is None:
            numeric_cols = self._numeric_columns()
        weighted[numeric_cols] = weighted[numeric_cols].mul(self.weights, axis=0)
        return weighted  # type: ignore[arg-type,return-value]

    def count(self, skipna: bool = True) -> DataFrame:
        """Count of grouped observations with applied weights.

        See `pandas.DataFrameGroupBy.count` for more details on the parameters.

        If `skipna` is `True`, missing values in the DataFrame are not counted towards
        the total weight for that row/column.

        If `skipna` is `False`, missing values are treated as having a count of 1,
        and weighted by the weights if available,
        but missing weights are still treated as 0.
        If including missing weights is desired, fill missing weights using
        `df.wt(..., na_weight=1)`.
        """

        weights = self._broadcast_weights(skipna=skipna)
        return weights.groupby(self._groupby._grouper).sum()  # type: ignore[arg-type,return-value]

    def _count_numeric(self, numeric_cols: pd.Index, skipna: bool = True) -> DataFrame:
        weights = self._broadcast_weights(skipna=skipna)
        return weights.groupby(self._groupby._grouper)[numeric_cols].sum()  # type: ignore[arg-type,return-value]

    def sum(
        self,
        min_count: int = 0,
        engine: "WindowingEngine" = None,
        engine_kwargs: "WindowingEngineKwargs" = None,
    ) -> DataFrame:
        """Sum of grouped values with applied weights.

        See `pandas.DataFrameGroupBy.sum` for more details on the parameters.
        """

        numeric_cols = self._numeric_columns()
        return self._sum_weighted(
            self._weighted(numeric_cols),
            numeric_cols=numeric_cols,
            min_count=min_count,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def _sum_weighted(
        self,
        weighted: pd.DataFrame,
        numeric_cols: pd.Index,
        min_count: int = 0,
        engine: "WindowingEngine" = None,
        engine_kwargs: "WindowingEngineKwargs" = None,
    ) -> DataFrame:
        return weighted.groupby(self._groupby._grouper)[numeric_cols].sum(  # type: ignore[arg-type, return-value]
            min_count=min_count,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def mean(
        self,
        skipna: bool = True,
        engine: "WindowingEngine" = None,
        engine_kwargs: "WindowingEngineKwargs" = None,
    ) -> DataFrame:
        """Mean of grouped values with applied weights.

        See `pandas.DataFrameGroupBy.mean` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        numeric_cols = self._numeric_columns()
        weighted = self._weighted(numeric_cols)
        return self._sum_weighted(
            weighted, numeric_cols, engine=engine, engine_kwargs=engine_kwargs
        ) / self._count_numeric(numeric_cols, skipna)

    def var(
        self,
        ddof: int = 1,
        skipna: bool = True,
        engine: "WindowingEngine" = None,
        engine_kwargs: "WindowingEngineKwargs" = None,
    ) -> DataFrame:
        """Variance of grouped values with applied weights.

        See `pandas.DataFrameGroupBy.var` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        numeric_cols = self._numeric_columns()

        sum_ = self.sum(engine=engine, engine_kwargs=engine_kwargs)[numeric_cols]
        count = self._count_numeric(numeric_cols, skipna)
        sum_squared = (
            self._groupby._selected_obj[numeric_cols]
            .pow(2)
            .mul(self.weights, axis=0)
            .groupby(self._groupby._grouper)
            .sum(engine=engine, engine_kwargs=engine_kwargs)
        )

        return variance_from_weighted_moments(sum_, sum_squared, count, ddof=ddof)  # type: ignore[return-value]

    def std(
        self,
        ddof: int = 1,
        skipna: bool = True,
        engine: "WindowingEngine" = None,
        engine_kwargs: "WindowingEngineKwargs" = None,
    ) -> DataFrame:
        """Standard deviation of grouped values with applied weights.

        See `pandas.DataFrameGroupBy.std` for more details on the parameters.

        See `skipna` parameter in `count` method for how missing values are treated.
        """
        return np.sqrt(
            self.var(ddof, skipna=skipna, engine=engine, engine_kwargs=engine_kwargs)
        )  # type: ignore[return-value]

    def corr(
        self,
        method: Literal["pearson"] = "pearson",
        min_periods: int = 1,
        ddof: int = 1,
    ) -> DataFrame:
        """Pairwise weighted correlation of columns within each group.

        This method currently supports weighted Pearson correlation.
        """
        if method != "pearson":
            raise NotImplementedError(
                "Only 'pearson' weighted correlation is supported."
            )

        grouped_corr: dict[Hashable, pd.DataFrame] = {}
        for key, group in self:
            grouped_corr[key] = group.corr(
                method=method,
                min_periods=min_periods,
                ddof=ddof,
            )

        if not grouped_corr:
            return DataFrame()

        group_names = [
            name for name in self._groupby._grouper.names if name is not None
        ]
        result = pd.concat(grouped_corr, names=group_names)
        return DataFrame(result)

    @overload
    def apply(self, func: Callable[..., "Scalar"], *args, **kwargs) -> "Series": ...
    @overload
    def apply(
        self, func: Callable[..., D1NumericArray], *args, **kwargs
    ) -> DataFrame: ...
    def apply(self, func: "AggFuncType", *args, **kwargs) -> "Series | DataFrame":
        """Apply a function to each group.

        The function is applied on the weighted data.

        See `pandas.DataFrameGroupBy.apply` for more details on the parameters.
        """
        return (
            self._weighted()
            .groupby(self._groupby._grouper)  # type: ignore[arg-type,return-value]
            .apply(func, *args, **kwargs)
        )
