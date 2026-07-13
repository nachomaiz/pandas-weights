import numpy as np
import pandas as pd
import pytest

from pandas_weights import series


def test_series_wt_init_weight():
    s = series.Series([1, 2, 3])
    weights_series = pd.Series([0.5, 1.5, 2.0])
    s_wt = series.WeightedSeriesAccessor._init_validated(s, weights_series)
    assert np.array_equal(s_wt.weights, weights_series)
    assert np.array_equal(s_wt.weighted(), s * weights_series)


@pytest.mark.parametrize(
    "weights",
    [
        [0.5, 1.5, 2.0],
        pd.Series([0.5, 1.5, 2.0]),
        np.array([0.5, 1.5, 2.0]),
    ],
)
def test_series_wt_weighted_types(weights):
    s = series.Series([1, 2, 3])
    s_wt = s.wt(weights)
    assert np.array_equal(s_wt.weights, weights)
    assert np.array_equal(s_wt.weighted(), s * pd.Series(weights, index=s.index))


def test_series_wt_weighted_na_weight():
    s = series.Series([1, 2, 3])
    weights_with_na = pd.Series([0.5, None, 2.0])
    na_weight = 1.0
    s_wt = s.wt(weights_with_na, na_weight=na_weight)
    expected_weights = pd.Series([0.5, na_weight, 2.0])
    assert np.array_equal(s_wt.weights, expected_weights)
    assert np.array_equal(s_wt.weighted(), s * expected_weights)


def test_series_wt_count():
    s = series.Series([1, 2, np.nan])
    weights_series = pd.Series([0.5, 1.5, 2.0])
    s_wt = s.wt(weights_series)

    expected_count_skipna = 2.0
    expected_count_noskipna = 4.0

    assert s_wt.count(skipna=True) == expected_count_skipna
    assert s_wt.count(skipna=False) == expected_count_noskipna


def test_series_wt_sum():
    s = series.Series([1, 2, 3])
    weights_series = pd.Series([0.5, 1.5, 2.0])
    s_wt = s.wt(weights_series)

    expected_sum = 9.5

    assert s_wt.sum() == expected_sum


def test_series_wt_sum_min_count():
    s = series.Series([1, 2, None]).astype(float)
    weights_series = pd.Series([0.5, 1.5, 2.0])
    s_wt = s.wt(weights_series)

    expected_sum = 3.5

    assert s_wt.sum(min_count=1) == expected_sum


def test_series_wt_mean():
    s = series.Series([1, 2, 3])
    weights_series = pd.Series([0.5, 1.5, 2.0])
    s_wt = s.wt(weights_series)

    expected_mean = 2.375

    assert s_wt.mean(axis=0) == expected_mean


def test_series_wt_var():
    s = series.Series([1, 2, 3])
    weights_series = pd.Series([0.5, 1.5, 2.0])
    s_wt = s.wt(weights_series)

    expected_var = 0.6458333333333334

    assert s_wt.var() == expected_var


def test_series_wt_std():
    s = series.Series([1, 2, 3])
    weights_series = pd.Series([0.5, 1.5, 2.0])
    s_wt = s.wt(weights_series)

    expected_std = 0.8036375634160796

    assert s_wt.std() == expected_std


def test_series_wt_corr():
    idx = pd.Index([0, 1, 2, 3])
    s = series.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    other = pd.Series([2.0, 4.0, 6.0, 8.0], index=idx)
    weights = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    assert s.wt(weights).corr(other) == pytest.approx(1.0)


def test_series_wt_corr_alignment_and_min_periods():
    s = series.Series([1.0, 2.0, 3.0], index=pd.Index([0, 1, 2]))
    other = pd.Series([10.0, 20.0], index=pd.Index([0, 1]))
    weights = pd.Series([1.0, 1.0, 1.0], index=s.index)

    assert s.wt(weights).corr(other) == pytest.approx(1.0)
    assert np.isnan(s.wt(weights).corr(other, min_periods=3))


def test_series_wt_corr_unsupported_method():
    s = series.Series([1.0, 2.0, 3.0])
    other = pd.Series([3.0, 2.0, 1.0])
    weights = pd.Series([1.0, 1.0, 1.0])

    with pytest.raises(NotImplementedError):
        s.wt(weights).corr(other, method="kendall")


def test_series_wt_groupby_init():
    idx = pd.MultiIndex.from_arrays([["A", "A", "B", "B"]], names=["Group"])
    s = series.Series([10, 20, 30, 40], index=idx)
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group", observed=False)
    assert isinstance(grouped, series.WeightedSeriesGroupBy)
    assert np.array_equal(grouped.weights, weights)


def test_series_wt_groupby_iter():
    idx = pd.Index(["A", "A", "B", "B"], name="Group")
    s = series.Series([10, 20, 30, 40], index=idx)
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped: series.WeightedSeriesGroupBy = s.wt(weights).groupby("Group")
    groups = dict(iter(grouped))

    assert set(groups.keys()) == {"A", "B"}
    pd.testing.assert_series_equal(
        groups["A"].obj,
        s[s.index.get_level_values("Group") == "A"],
    )
    pd.testing.assert_series_equal(
        groups["B"].obj,
        s[s.index.get_level_values("Group") == "B"],
    )
    pd.testing.assert_series_equal(
        groups["A"].weights,
        weights[weights.index.get_level_values("Group") == "A"],
    )
    pd.testing.assert_series_equal(
        groups["B"].weights,
        weights[weights.index.get_level_values("Group") == "B"],
    )


def test_series_wt_groupby_count():
    idx = pd.MultiIndex.from_arrays([["A", "A", "B", "B"]], names=["Group"])
    s = series.Series([10, 20, None, 40], index=idx, name="test").astype(float)
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group")
    expected_count_skipna = pd.Series(
        [3.0, 2.5], index=pd.Index(["A", "B"], name="Group"), name="test"
    )
    expected_count_noskipna = pd.Series(
        [3.0, 4.0], index=pd.Index(["A", "B"], name="Group"), name="test"
    )
    pd.testing.assert_series_equal(grouped.count(), expected_count_skipna)
    pd.testing.assert_series_equal(grouped.count(skipna=False), expected_count_noskipna)


def test_series_wt_groupby_sum():
    idx = pd.MultiIndex.from_arrays([["A", "A", "B", "B"]], names=["Group"])
    s = series.Series([10, 20, 30, 40], index=idx, name="test")
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group")
    expected_sum = pd.Series(
        [50.0, 145.0], index=pd.Index(["A", "B"], name="Group"), name="test"
    )
    pd.testing.assert_series_equal(grouped.sum(), expected_sum, check_series_type=False)


def test_series_wt_groupby_mean():
    idx = pd.MultiIndex.from_arrays([["A", "A", "B", "B"]], names=["Group"])
    s = series.Series([10, 20, 30, 40], index=idx, name="test")
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group")
    expected_mean = pd.Series(
        [16.666666666666668, 36.25],
        index=pd.Index(["A", "B"], name="Group"),
        name="test",
    )
    pd.testing.assert_series_equal(grouped.mean(), expected_mean)


def test_series_wt_groupby_var():
    idx = pd.MultiIndex.from_arrays([["A", "A", "B", "B"]], names=["Group"])
    s = series.Series([10, 20, 30, 40], index=idx, name="test")
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group")
    expected_var = pd.Series(
        [33.333333333333314, 31.25],
        index=pd.Index(["A", "B"], name="Group"),
        name="test",
    )
    pd.testing.assert_series_equal(grouped.var(), expected_var)


def test_series_wt_groupby_std():
    idx = pd.MultiIndex.from_arrays([["A", "A", "B", "B"]], names=["Group"])
    s = series.Series([10, 20, 30, 40], index=idx, name="test")
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group")
    expected_std = pd.Series(
        [5.773502691896255, 5.5901699437494745],
        index=pd.Index(["A", "B"], name="Group"),
        name="test",
    )
    pd.testing.assert_series_equal(grouped.std(), expected_std)


def test_series_wt_groupby_corr():
    idx = pd.Index(["A", "A", "B", "B"], name="Group")
    s = series.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="test")
    other = pd.Series([2.0, 4.0, 8.0, 6.0], index=idx)
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)

    grouped = s.wt(weights).groupby("Group")
    expected = pd.Series(
        [0.0, 0.0],
        index=pd.Index(["A", "B"], name="Group"),
        name="test",
    )
    pd.testing.assert_series_equal(grouped.corr(other), expected)


def test_series_wt_groupby_corr_alignment_and_min_periods():
    idx = pd.Index(["A", "A", "B", "B"], name="Group")
    s = series.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="test")
    other = pd.Series([10.0, 20.0], index=pd.Index(["A", "A"]))
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)

    grouped = s.wt(weights).groupby("Group")
    expected_default = pd.Series(
        [1.0, np.nan],
        index=pd.Index(["A", "B"], name="Group"),
        name="test",
    )
    expected_min_periods = pd.Series(
        [np.nan, np.nan],
        index=pd.Index(["A", "B"], name="Group"),
        name="test",
    )

    pd.testing.assert_series_equal(grouped.corr(other), expected_default)
    pd.testing.assert_series_equal(
        grouped.corr(other, min_periods=3), expected_min_periods
    )


def test_series_wt_groupby_corr_unsupported_method():
    idx = pd.Index(["A", "A", "B", "B"], name="Group")
    s = series.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    other = pd.Series([4.0, 3.0, 2.0, 1.0], index=idx)
    weights = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)

    with pytest.raises(NotImplementedError):
        s.wt(weights).groupby("Group").corr(other, method="kendall")


def test_series_wt_apply():
    s = pd.Series([10, 20, 30, 40], name="Value")
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=s.index)

    def add_two(value: float) -> float:
        return value + 2

    expected_apply_scalar = pd.Series([12.0, 42.0, 47.0, 102.0])
    pd.testing.assert_series_equal(s.wt(weights).apply(add_two), expected_apply_scalar)


def test_series_wt_groupby_apply():
    s = pd.Series([10, 20, 30, 40], index=pd.Index(["A", "A", "B", "B"], name="Group"))
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=s.index)

    def weighted_minmax(series: pd.Series) -> pd.Series:
        return pd.Series({"min": series.min(), "max": series.max()}, name="Value")

    expected_apply_array = pd.Series(
        [10.0, 40.0, 45.0, 100.0],
        index=pd.MultiIndex.from_tuples(
            [("A", "min"), ("A", "max"), ("B", "min"), ("B", "max")],
            names=["Group", None],
        ),
    )
    pd.testing.assert_series_equal(
        s.wt(weights).groupby("Group").apply(weighted_minmax),
        expected_apply_array,
    )


def test_df_wt_groupby_multiple_groupings():
    idx = pd.MultiIndex.from_arrays(
        [["A", "A", "B", "B"], ["A", "B", "A", "B"]], names=["Group", "Subgroup"]
    )
    s = series.Series([10, 20, 30, 40], index=idx)
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby(["Group", "Subgroup"])
    assert isinstance(grouped._group_keys(), pd.MultiIndex)


def test_series_wt_resample_sum_count_mean():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    s = series.Series([1.0, 2.0, np.nan, 4.0], index=idx, name="Value")
    weights = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    resampled = s.wt(weights).resample("2D")

    expected_sum = pd.Series(
        [5.0, 16.0],
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
        name="Value",
    )
    expected_count_skipna = pd.Series(
        [3.0, 4.0],
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
        name="Value",
    )
    expected_count_noskipna = pd.Series(
        [3.0, 7.0],
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
        name="Value",
    )
    expected_mean = pd.Series(
        [5.0 / 3.0, 4.0],
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
        name="Value",
    )

    pd.testing.assert_series_equal(resampled.sum(), expected_sum)
    pd.testing.assert_series_equal(resampled.count(), expected_count_skipna)
    pd.testing.assert_series_equal(
        resampled.count(skipna=False), expected_count_noskipna
    )
    pd.testing.assert_series_equal(resampled.mean(), expected_mean)


def test_series_wt_resample_var_std():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    s = series.Series([1.0, 2.0, np.nan, 4.0], index=idx, name="Value")
    weights = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    resampled = s.wt(weights).resample("2D")

    expected_var = pd.Series(
        [1.0 / 3.0, 0.0],
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
        name="Value",
    )
    expected_std = pd.Series(
        [np.sqrt(1.0 / 3.0), 0.0],
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
        name="Value",
    )

    pd.testing.assert_series_equal(resampled.var(), expected_var)
    pd.testing.assert_series_equal(resampled.std(), expected_std)
