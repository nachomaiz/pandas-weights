from typing import TypeVar, TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from pandas._typing import Axis

T = TypeVar("T", pd.Series, pd.DataFrame)

NAN = float("nan")


def weighted_sum_of_squares(
    obj: pd.Series | pd.DataFrame,
    weights: pd.Series,
    *,
    axis: "Axis" = 0,
    min_count: int = 1,
) -> pd.Series:
    return obj.mul(obj).mul(weights, axis=0).sum(axis=axis, min_count=min_count)  # type: ignore[arg-type]


def variance_from_weighted_moments(
    weighted_sum: T,
    weighted_sum_of_squares_: T,
    weighted_count: T,
    *,
    ddof: int,
) -> T:
    return (
        weighted_sum_of_squares_ - (weighted_sum * weighted_sum) / weighted_count
    ) / (weighted_count - ddof)


def weighted_correlation(
    x: pd.Series,
    y: pd.Series,
    weights: pd.Series,
    *,
    ddof: int = 1,
    min_periods: int = 1,
) -> float:
    valid = x.notna() & y.notna() & weights.notna()
    if valid.sum() < min_periods:
        return NAN

    x_valid = x[valid]
    y_valid = y[valid]
    w_valid = weights[valid]

    w_sum = w_valid.sum()
    if not np.isfinite(w_sum) or w_sum <= ddof:
        return NAN

    wx = x_valid.mul(w_valid)
    wy = y_valid.mul(w_valid)
    wxy = x_valid.mul(y_valid).mul(w_valid)
    wx2 = x_valid.mul(x_valid).mul(w_valid)
    wy2 = y_valid.mul(y_valid).mul(w_valid)

    wx_sum = wx.sum()
    wy_sum = wy.sum()

    denom = w_sum - ddof
    cov = (wxy.sum() - (wx_sum * wy_sum) / w_sum) / denom
    var_x = (wx2.sum() - (wx_sum * wx_sum) / w_sum) / denom
    var_y = (wy2.sum() - (wy_sum * wy_sum) / w_sum) / denom

    if var_x <= 0 or var_y <= 0:
        return NAN

    return cov / np.sqrt(var_x * var_y)
