from collections.abc import Hashable, Iterator, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Callable, Literal, Union, overload

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

from pandas_weights.base import BaseWeightedAccessor
from pandas_weights.series import WeightedSeriesAccessor, WeightedSeriesGroupBy
from pandas_weights.typing_ import D1NumericArray, Number

if TYPE_CHECKING:
    from pandas._typing import (
        AggFuncType,
        Axis,
        GroupByObjectNonScalar,
        Scalar,
        WindowingEngine,
        WindowingEngineKwargs,
    )

    from pandas_weights.series import Series


class _NoDefault(Enum):
    no_default = object()


class DataFrame(pd.DataFrame):
    wt: "WeightedDataFrameAccessor"


@pd.api.extensions.register_dataframe_accessor("wt")
class WeightedDataFrameAccessor(BaseWeightedAccessor[DataFrame]):
    def __call__(
        self,
        weights: Union[Hashable, D1NumericArray],
        /,
        na_weight: Union[Number, None] = None,
    ) -> "WeightedDataFrameAccessor":
        if isinstance(weights, (list, pd.Series, np.ndarray)):
            self.weights = weights
        else:
            self._weights = self.obj[weights]  # we know it's the right length

        if na_weight is not None:
            self._weights = self.weights.fillna(na_weight)

        return self

    @overload
    def __getitem__(self, key: Hashable) -> "WeightedSeriesAccessor": ...
    @overload
    def __getitem__(self, key: Sequence[Hashable]) -> "WeightedDataFrameAccessor": ...
    def __getitem__(
        self, key: Union[Hashable, Sequence[Hashable]]
    ) -> Union["WeightedDataFrameAccessor", "WeightedSeriesAccessor"]:
        if isinstance(key, list):
            return WeightedDataFrameAccessor._init_validated(
                self.obj[key], self.weights
            )
        return WeightedSeriesAccessor._init_validated(self.obj[key], self.weights)  # type: ignore[arg-type]

    def weighted(self) -> DataFrame:
        """Get the DataFrame with applied weights

        Returns
        -------
        DataFrame
            DataFrame with applied weights
        """
        return self._clean_obj().mul(self.weights, axis=0)

    def _clean_obj(self) -> DataFrame:
        if (weights_col := self.weights.name) in self.obj.columns:
            return self.obj.drop(columns=weights_col)  # type: ignore[return-value]
        return self.obj  # type: ignore[return-value]

    def groupby(
        self,
        by: Union["Scalar", "GroupByObjectNonScalar", pd.MultiIndex, None] = None,
        axis: Union["Axis", Literal[_NoDefault.no_default]] = _NoDefault.no_default,
        level: Union[int, str, None] = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: Union[bool, Literal[_NoDefault.no_default]] = _NoDefault.no_default,
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
        if axis is not _NoDefault.no_default:
            kwargs["axis"] = axis
        if observed is not _NoDefault.no_default:
            kwargs["observed"] = observed

        return WeightedFrameGroupBy(self.weights, self._clean_obj(), **kwargs)

    def count(self, axis: "Axis" = 0, skipna: bool = True) -> "Series":
        """Count observations weighted by the weights.

        See `pandas.DataFrame.count` for more details on the parameters.
        """
        obj = self._clean_obj()

        if skipna:
            weights = obj.notna().mul(self.weights, axis=0)
        else:
            weights = pd.DataFrame(
                np.broadcast_to(np.asarray(self.weights).reshape(-1, 1), obj.shape),
                index=obj.index,
                columns=obj.columns,
            ).fillna(1.0)
        return weights.sum(axis=axis)  # type: ignore[return-value]

    def sum(self, axis: "Axis" = 0, min_count: int = 0) -> "Series":
        """Sum of values weighted by the weights.

        See `pandas.DataFrame.sum` for more details on the parameters.
        """
        return self.weighted().sum(axis=axis, min_count=min_count)  # type: ignore[return-value]

    def mean(self, axis: "Axis" = 0, skipna: bool = True) -> "Series":
        """Mean of values weighted by the weights.

        See `pandas.DataFrame.mean` for more details on the parameters.
        """
        return self.sum(axis=axis, min_count=1) / self.count(axis=axis, skipna=skipna)  # type: ignore[return-value]

    def var(self, axis: "Axis" = 0, ddof: int = 1, skipna: bool = True) -> "Series":
        """Variance of values weighted by the weights.

        See `pandas.DataFrame.var` for more details on the parameters.
        """
        sum_ = self.sum(axis=axis, min_count=1)
        count = self.count(axis=axis, skipna=skipna)
        diff = self._clean_obj().sub(sum_ / count, axis=1 if axis == 0 else 0)
        diff_squared = diff.mul(diff)
        return diff_squared.sum(axis=axis) / (count - ddof)  # type: ignore[return-value]

    def std(self, axis: "Axis" = 0, ddof: int = 1, skipna: bool = True) -> "Series":
        """Standard deviation of values weighted by the weights.

        See `pandas.DataFrame.std` for more details on the parameters.
        """

        return self.var(axis=axis, ddof=ddof, skipna=skipna).pow(0.5)  # type: ignore[return-value]

    @overload
    def apply(
        self,
        func: Callable[..., "Scalar"],
        axis: "Axis" = ...,
        raw: bool = ...,
        result_type: Union[Literal["reduce"], None] = ...,
        args: tuple = ...,
        by_row: Literal[False, "compat"] = ...,
        engine: Literal["python", "numba"] = ...,
        engine_kwargs: Union[dict[str, bool], None] = ...,
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
        engine_kwargs: Union[dict[str, bool], None] = ...,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def apply(
        self,
        func: Callable[..., D1NumericArray],
        axis: "Axis" = 0,
        raw: bool = ...,
        result_type: Union[Literal["expand", "broadcast"], None] = ...,
        args: tuple = ...,
        by_row: Literal[False, "compat"] = ...,
        engine: Literal["python", "numba"] = ...,
        engine_kwargs: Union[dict[str, bool], None] = ...,
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
        engine_kwargs: Union[dict[str, bool], None] = ...,
        **kwargs,
    ) -> "Series": ...
    def apply(
        self,
        func: "AggFuncType",
        axis: "Axis" = 0,
        raw: bool = False,
        result_type: Union[Literal["expand", "reduce", "broadcast"], None] = None,
        args: tuple = (),
        by_row: Literal[False, "compat"] = "compat",
        engine: Literal["python", "numba"] = "python",
        engine_kwargs: Union[dict[str, bool], None] = None,
        **kwargs,
    ) -> Union["Series", DataFrame]:
        """Apply a function along the axis of the DataFrame.

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
    def __getitem__(self, key: "Hashable") -> "WeightedSeriesGroupBy": ...
    @overload
    def __getitem__(self, key: "Sequence[Hashable]") -> "WeightedFrameGroupBy": ...
    def __getitem__(
        self, key: Union[Hashable, Sequence[Hashable]]
    ) -> Union["WeightedFrameGroupBy", "WeightedSeriesGroupBy"]:
        if isinstance(key, list):
            return WeightedFrameGroupBy._init_groupby(self.weights, self._groupby[key])
        return WeightedSeriesGroupBy._init_groupby(self.weights, self._groupby[key])  # type: ignore[arg-type]

    def _group_keys(self) -> Union[pd.Index, pd.MultiIndex]:
        if len(names := self._groupby._grouper.names) == 1:
            return pd.Index(self._groupby.obj.reset_index()[names[0]])
        return pd.MultiIndex.from_frame(self._groupby.obj.reset_index()[names])

    def _broadcast_weights(self, skipna: bool = True) -> DataFrame:
        if skipna:
            return (  # type: ignore[arg-type,return-value]
                self._groupby.obj.drop(columns=self._groupby.exclusions)
                .notna()
                .mul(self.weights, axis=0)
            )
        return DataFrame(
            np.broadcast_to(
                np.asarray(self.weights).reshape(-1, 1),
                self._groupby.obj.drop(columns=self._groupby.exclusions).shape,
            ),
            index=self._groupby.obj.index,
            columns=self._groupby.obj.drop(columns=self._groupby.exclusions).columns,
        ).fillna(1.0)

    def _numeric_columns(self) -> pd.Index:
        return (
            self._groupby.obj.drop(columns=self._groupby.exclusions, errors="ignore")
            .select_dtypes(include=["number", "bool"])
            .columns
        )

    def _weighted(self, numeric_cols: Union[pd.Index, None] = None) -> DataFrame:
        weighted = self._groupby.obj.copy()
        if numeric_cols is None:
            numeric_cols = self._numeric_columns()
        weighted[numeric_cols] = weighted[numeric_cols].mul(self.weights, axis=0)
        return weighted  # type: ignore[arg-type,return-value]

    def count(self, skipna: bool = True) -> DataFrame:
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
        numeric_cols = self._numeric_columns()
        group_keys = self._group_keys()
        weighted = self._weighted(numeric_cols)

        sum_ = self._sum_weighted(
            weighted,
            numeric_cols=numeric_cols,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )[numeric_cols]
        count = self._count_numeric(numeric_cols, skipna)

        diff = (
            weighted.reset_index()
            .set_index(group_keys)[numeric_cols]
            .sub((sum_ / count).loc[group_keys], axis=0)
        )
        diff_squared = diff.mul(diff)

        return diff_squared.groupby(self._groupby._grouper).sum(  # type: ignore[arg-type]
            engine=engine, engine_kwargs=engine_kwargs
        ) / (count - ddof)

    def std(
        self,
        ddof: int = 1,
        skipna: bool = True,
        engine: "WindowingEngine" = None,
        engine_kwargs: "WindowingEngineKwargs" = None,
    ) -> DataFrame:
        return self.var(
            ddof, skipna=skipna, engine=engine, engine_kwargs=engine_kwargs
        ).pow(0.5)

    @overload
    def apply(
        self,
        func: Callable[..., "Scalar"],
        *args,
        **kwargs,
    ) -> "Series": ...
    @overload
    def apply(
        self,
        func: Callable[..., D1NumericArray],
        *args,
        **kwargs,
    ) -> DataFrame: ...
    def apply(
        self,
        func: "AggFuncType",
        *args,
        **kwargs,
    ) -> Union["Series", DataFrame]:
        return (
            self._weighted()
            .groupby(self._groupby._grouper)  # type: ignore[arg-type]
            .apply(func, *args, **kwargs)
        )
