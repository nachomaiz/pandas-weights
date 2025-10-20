import numpy as np
import pandas as pd
import pytest

from pandas_weights import series


def test_series_wt_error_on_no_weights_set():
    s = series.Series()
    with pytest.raises(ValueError, match="Weights have not been set"):
        s.wt.weighted()


def test_series_wt_error_on_weights_length_mismatch():
    s = series.Series([1, 2, 3])
    with pytest.raises(
        ValueError, match="Length of weights must match number of rows in the data."
    ):
        s.wt(np.array([1, 2]))


def test_series_wt_error_on_weights_not_1d():
    s = series.Series([1, 2, 3])
    with pytest.raises(ValueError, match="weights must be one-dimensional"):
        s.wt(np.array([[1], [2], [3]]))


@pytest.mark.parametrize(
    "weights_types",
    [[0.5, 1.5, 2.0], pd.Series([0.5, 1.5, 2.0]), np.array([0.5, 1.5, 2.0])],
)
def test_series_wt_weighted(weights_types: list[float] | pd.Series | np.ndarray):
    s = series.Series([1, 2, 3])
    s_wt = s.wt(weights_types)
    assert np.array_equal(s_wt.weights, weights_types)
    assert np.array_equal(s_wt.weighted(), s * weights_types)


def test_series_wt_init_weight():
    s = series.Series([1, 2, 3])
    weights_series = pd.Series([0.5, 1.5, 2.0])
    s_wt = series.WeightedSeriesAccessor._init_validated(s, weights_series)
    assert np.array_equal(s_wt.weights, weights_series)
    assert np.array_equal(s_wt.weighted(), s * weights_series)


def test_series_wt_T():
    s = series.Series([1, 2])
    weights_series = pd.Series([0.5, 1.5])
    expected = s.mul(weights_series, axis=0).T
    np.array_equal(s.wt(weights_series).T, expected)


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

    expected_var = 0.8072916666666666

    assert s_wt.var() == expected_var


def test_series_wt_std():
    s = series.Series([1, 2, 3])
    weights_series = pd.Series([0.5, 1.5, 2.0])
    s_wt = s.wt(weights_series)

    expected_std = 0.898494110535326

    assert s_wt.std() == expected_std


def test_series_wt_groupby_init():
    idx = pd.MultiIndex.from_arrays([["A", "A", "B", "B"]], names=["Group"])
    s = series.Series([10, 20, 30, 40], index=idx)
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group")
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
    s = series.Series([10, 20, None, 40], index=idx).astype(float)
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group")
    expected_count_skipna = pd.Series(
        [3.0, 2.5], index=pd.Index(["A", "B"], name="Group")
    )
    expected_count_noskipna = pd.Series(
        [3.0, 4.0], index=pd.Index(["A", "B"], name="Group")
    )
    pd.testing.assert_series_equal(grouped.count(), expected_count_skipna)
    pd.testing.assert_series_equal(grouped.count(skipna=False), expected_count_noskipna)


def test_series_wt_groupby_sum():
    idx = pd.MultiIndex.from_arrays([["A", "A", "B", "B"]], names=["Group"])
    s = series.Series([10, 20, 30, 40], index=idx)
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group")
    expected_sum = pd.Series(
        [50.0, 145.0], index=pd.Index(["A", "B"], name="Group"), name=None
    )
    pd.testing.assert_series_equal(grouped.sum(), expected_sum, check_series_type=False)


def test_series_wt_groupby_mean():
    idx = pd.MultiIndex.from_arrays([["A", "A", "B", "B"]], names=["Group"])
    s = series.Series([10, 20, 30, 40], index=idx)
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group")
    expected_mean = pd.Series(
        [16.666666666666668, 36.25], index=pd.Index(["A", "B"], name="Group"), name=None
    )
    pd.testing.assert_series_equal(grouped.mean(), expected_mean)


def test_series_wt_groupby_var():
    idx = pd.MultiIndex.from_arrays([["A", "A", "B", "B"]], names=["Group"])
    s = series.Series([10, 20, 30, 40], index=idx)
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group")
    expected_var = pd.Series(
        [294.4444444444444, 1380.2083333333333],
        index=pd.Index(["A", "B"], name="Group"),
        name=None,
    )
    pd.testing.assert_series_equal(grouped.var(), expected_var)


def test_series_wt_groupby_std():
    idx = pd.MultiIndex.from_arrays([["A", "A", "B", "B"]], names=["Group"])
    s = series.Series([10, 20, 30, 40], index=idx)
    weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
    grouped = s.wt(weights).groupby("Group")
    expected_std = pd.Series(
        [17.159383568311664, 37.151155208597935],
        index=pd.Index(["A", "B"], name="Group"),
        name=None,
    )
    pd.testing.assert_series_equal(grouped.std(), expected_std)
