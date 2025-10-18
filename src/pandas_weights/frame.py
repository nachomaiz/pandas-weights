from collections.abc import Hashable
from enum import Enum
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from .accessor import register_accessor as _register_accessor

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        GroupByObjectNonScalar,
        Scalar,
        WindowingEngine,
        WindowingEngineKwargs,
        AggFuncType,
    )


class _NoDefault(Enum):
    no_default = object()


class DataFrame(pd.DataFrame):
    wt: "WeightedDataFrameAccessor"


@_register_accessor("wt", pd.DataFrame)
class WeightedDataFrameAccessor:
    def __init__(self, obj: pd.DataFrame) -> None:
        self.obj = DataFrame(obj)
        self._weights: pd.Series | np.ndarray | None = None

    def __call__(
        self, weights: Hashable | list[int | float] | pd.Series | np.ndarray, /
    ) -> Self:
        if isinstance(weights, (list, pd.Series, np.ndarray)):
            self.weights = weights
        else:
            self._weights = self.obj[weights]  # we know it's the right length
            self.obj = self.obj.drop(columns=weights)
        return self

    def __getitem__(self, key: list[Hashable]) -> "WeightedDataFrameAccessor":
        return WeightedDataFrameAccessor._init_weight(self.obj[key], self.weights)

    @classmethod
    def _init_weight(
        cls, pandas_obj: pd.DataFrame, weights: pd.Series | np.ndarray
    ) -> Self:
        self = cls(pandas_obj)
        self.weights = weights

        return self

    @property
    def T(self) -> DataFrame:
        return DataFrame(self.weighted().T)

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

    def weighted(self) -> DataFrame:
        """Return a DataFrame with the weights applied to the whole DataFrame."""
        return self.obj.mul(self.weights, axis=0)

    def raw(
        self,
        weights_col: Hashable | tuple[Hashable, ...] | None = None,
        conflict: Literal["overwrite", "raise"] = "raise",
        default: Hashable | tuple[Hashable, ...] = "weights",
    ) -> pd.DataFrame:
        """Return the original DataFrame without weights applied.

        If weights were set using a column name, the original DataFrame is returned.

        If weights were set using a Series or array, it will be added as a new column to the
        DataFrame. The name of the new column can be specified using the `weights_col`
        parameter. If `weights_col` is None, the name will be set to `default`. If a column
        with the same name already exists in the DataFrame, a ValueError will be raised
        unless `conflict` is set to "overwrite".
        """
        if weights_col is None:
            weights_col = default

        if weights_col in self.obj.columns:
            if conflict == "raise":
                raise ValueError(
                    f"Column '{weights_col}' already exists in DataFrame. "
                    "Set `conflict='overwrite'` to overwrite it."
                )
            df = self.obj.drop(columns=weights_col)
        else:
            df = self.obj

        if nlevels := self.obj.columns.nlevels and not isinstance(weights_col, tuple):
            weights_col = (weights_col,) * nlevels

        if isinstance(self.weights, pd.Series):
            if not self.weights.name:
                self.weights.name = weights_col
            weights = self.weights
        else:
            weights = pd.Series(self.weights, index=self.obj.index, name=weights_col)
        return pd.concat([weights, df], axis=1)

    def groupby(
        self,
        by: "Scalar | GroupByObjectNonScalar | pd.MultiIndex | None" = None,
        axis: "Axis | Literal[_NoDefault.no_default]" = _NoDefault.no_default,
        level: int | str | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool | Literal[_NoDefault.no_default] = _NoDefault.no_default,
        dropna: bool = True,
    ) -> "WeightedFrameGroupBy":
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

        return WeightedFrameGroupBy(self.weights, self.obj, **kwargs)

    def count(self, axis: "Axis" = 0, skipna: bool = True) -> pd.Series:
        if skipna:
            weights = self.obj.notna().mul(self.weights, axis=0)
        else:
            weights = pd.DataFrame(
                np.broadcast_to(
                    np.asarray(self.weights).reshape(-1, 1), self.obj.shape
                ),
                index=self.obj.index,
                columns=self.obj.columns,
            ).fillna(1.0)
        return weights.sum(axis=axis)

    def sum(self, axis: "Axis" = 0, min_count: int = 0) -> pd.Series:
        return self.weighted().sum(axis=axis, min_count=min_count)

    def mean(self, axis: "Axis" = 0, skipna: bool = True) -> pd.Series:
        return self.sum(axis=axis, min_count=1) / self.count(axis=axis, skipna=skipna)

    def var(self, axis: "Axis" = 0, ddof: int = 1, skipna: bool = True) -> pd.Series:
        sum_ = self.sum(axis=axis, min_count=1)
        count = self.count(axis=axis, skipna=skipna)
        diff = self.obj.sub(sum_ / count, axis=1 if axis == 0 else 0)
        diff_squared = diff.mul(diff)
        return diff_squared.sum(axis=axis) / (count - ddof)

    def std(self, axis: "Axis" = 0, ddof: int = 1, skipna: bool = True) -> pd.Series:
        return self.var(axis=axis, ddof=ddof, skipna=skipna).pow(0.5)

    def apply(
        self,
        func: "AggFuncType",
        axis: "Axis" = 0,
        raw: bool = False,
        result_type: Literal["expand", "reduce", "broadcast"] | None = None,
        args: tuple = (),
        by_row: Literal[False, "compat"] = "compat",
        engine: Literal["python", "numba"] = "python",
        engine_kwargs: dict[str, bool] | None = None,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
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


class WeightedFrameGroupBy(DataFrameGroupBy):
    obj: DataFrame
    _grouper: pd.Grouper

    def __init__(self, weights: pd.Series | np.ndarray, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = weights

    def _broadcast_weights(self, skipna: bool = True) -> DataFrame:
        if skipna:
            return (
                self.obj.drop(columns=self.exclusions).notna().mul(self.weights, axis=0)
            )
        return DataFrame(
            np.broadcast_to(
                np.asarray(self.weights).reshape(-1, 1),
                self.obj.drop(columns=self.exclusions).shape,
            ),
            index=self.obj.index,
            columns=self.obj.drop(columns=self.exclusions).columns,
        ).fillna(1.0)

    def _group_keys(
        self, group_cols: list[Hashable]
    ) -> list[Hashable] | list[tuple[Hashable, ...]]:
        if len(group_cols) == 1:
            return self.obj.reset_index()[group_cols[0]].tolist()
        return pd.MultiIndex.from_frame(self.obj.reset_index()[group_cols]).tolist()

    def count(self, skipna: bool = True) -> DataFrame:
        weights = self._broadcast_weights(skipna=skipna)
        return DataFrame(weights.groupby(self._grouper).sum())

    def sum(
        self,
        min_count: int = 0,
        engine: "WindowingEngine" = None,
        engine_kwargs: "WindowingEngineKwargs" = None,
    ) -> DataFrame:
        numeric_cols = self.obj.select_dtypes(include=["number", "bool"]).columns
        self.obj[numeric_cols] = self.obj[numeric_cols].mul(self.weights, axis=0)
        return DataFrame(
            super().sum(
                numeric_only=True,
                min_count=min_count,
                engine=engine,
                engine_kwargs=engine_kwargs,
            )
        )

    def mean(
        self,
        skipna: bool = True,
        engine: "WindowingEngine" = None,
        engine_kwargs: "WindowingEngineKwargs" = None,
    ) -> DataFrame:
        return self.sum() / self.count(skipna=skipna)

    def var(
        self,
        ddof: int = 1,
        skipna: bool = True,
        engine: "WindowingEngine" = None,
        engine_kwargs: "WindowingEngineKwargs" = None,
    ) -> DataFrame:
        numeric_cols = (
            self.obj.drop(columns=self.exclusions, errors="ignore")
            .select_dtypes(include=["number", "bool"])
            .columns
        )
        group_cols = _sorted_group_columns(self)

        sum_ = self.sum(engine=engine, engine_kwargs=engine_kwargs)[numeric_cols]
        count = self.count(skipna=skipna)[numeric_cols]

        diff = (
            self.obj.reset_index()
            .set_index(group_cols)[numeric_cols]
            .sub((sum_ / count).loc[self._group_keys(group_cols)], axis=0)
        )
        diff_squared = diff.mul(diff)

        return DataFrame(
            diff_squared.groupby(self._grouper).sum(
                engine=engine, engine_kwargs=engine_kwargs
            )
            / (count - ddof)
        )

    def std(
        self,
        ddof: int = 1,
        skipna: bool = True,
        engine: "WindowingEngine" = None,
        engine_kwargs: "WindowingEngineKwargs" = None,
    ) -> DataFrame:
        return self.var(ddof, skipna=skipna).pow(0.5)


def _sorted_group_columns(grouped: DataFrameGroupBy) -> list[Hashable]:
    return sorted(
        grouped.exclusions,
        key=grouped.obj.columns.get_loc,  # type: ignore
    )
