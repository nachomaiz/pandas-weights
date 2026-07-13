import numpy as np
import pandas as pd
import pytest

from pandas_weights import frame, series


@pytest.fixture
def df():
    return frame.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "weights": [0.5, 1.5, 2.0]})


@pytest.fixture
def grouped_df():
    return frame.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "Value": [10, 20, 30, 40],
            "weights": [1.0, 2.0, 1.5, 2.5],
        }
    )


@pytest.mark.parametrize(
    "weights",
    [[0.5, 1.5, 2.0], pd.Series([0.5, 1.5, 2.0]), np.array([0.5, 1.5, 2.0])],
)
def test_df_wt_weighted(df: frame.DataFrame, weights):
    df_wt = df.wt(weights)
    assert np.array_equal(df_wt.weights, weights)
    assert np.array_equal(df_wt.weighted()["A"], df["A"] * weights)


def test_df_wt_weighted_na_weight(df: frame.DataFrame):
    weights_with_na = pd.Series([0.5, None, 2.0])
    na_weight = 1.0
    df_wt = df.wt(weights_with_na, na_weight=na_weight)
    expected_weights = pd.Series([0.5, na_weight, 2.0])
    assert np.array_equal(df_wt.weights, expected_weights)
    assert np.array_equal(df_wt.weighted()["A"], df["A"] * expected_weights)


def test_df_wt_weighted_column(df: frame.DataFrame):
    assert np.array_equal(
        df.wt("weights").weighted(), df[["A", "B"]].mul(df["weights"], axis=0)
    )


def test_df_wt_init_weight(df: frame.DataFrame):
    weights_series = pd.Series([0.5, 1.5, 2.0])
    df_wt = frame.WeightedDataFrameAccessor._init_validated(df, weights_series)
    assert np.array_equal(df_wt.weights, weights_series)
    assert np.array_equal(df_wt.weighted()["A"], df["A"] * weights_series)


def test_df_wt_count():
    df = frame.DataFrame({"A": [1, 2, np.nan], "B": [4, np.nan, 6]})
    weights_series = pd.Series([0.5, 1.5, 2.0])
    df_wt = df.wt(weights_series)

    expected_count_skipna = pd.Series({"A": 2.0, "B": 2.5})
    expected_count_noskipna = pd.Series({"A": 4.0, "B": 4.0})

    pd.testing.assert_series_equal(
        df_wt.count(axis=0, skipna=True), expected_count_skipna
    )
    pd.testing.assert_series_equal(
        df_wt.count(axis=0, skipna=False), expected_count_noskipna
    )


def test_df_wt_sum(df: frame.DataFrame):
    df_wt = df.wt("weights")

    expected_sum = pd.Series({"A": 9.5, "B": 21.5})

    pd.testing.assert_series_equal(df_wt.sum(axis=0), expected_sum)


def test_df_wt_sum_min_count():
    df = frame.DataFrame({"A": [1, 2, None], "B": [None, None, None]}).astype(float)
    weights_series = pd.Series([0.5, 1.5, 2.0])
    df_wt = df.wt(weights_series)

    expected_sum = pd.Series({"A": 3.5, "B": None})

    pd.testing.assert_series_equal(df_wt.sum(axis=0, min_count=1), expected_sum)


def test_df_wt_mean(df: frame.DataFrame):
    df_wt = df.wt("weights")

    expected_mean = pd.Series({"A": 2.375, "B": 5.375})

    pd.testing.assert_series_equal(df_wt.mean(axis=0), expected_mean)


def test_df_wt_var(df: frame.DataFrame):
    df_wt = df.wt("weights")

    expected_var = pd.Series({"A": 0.6458333333333334, "B": 0.6458333333333334})

    pd.testing.assert_series_equal(df_wt.var(axis=0), expected_var)


def test_df_wt_std(df: frame.DataFrame):
    df_wt = df.wt("weights")

    expected_std = pd.Series({"A": 0.8036375634160796, "B": 0.8036375634160796})

    pd.testing.assert_series_equal(df_wt.std(axis=0), expected_std)


def test_df_wt_groupby_init(grouped_df: frame.DataFrame):
    grouped = grouped_df.wt("weights").groupby("Group", observed=False)
    assert isinstance(grouped, frame.WeightedFrameGroupBy)
    assert np.array_equal(grouped.weights, grouped_df["weights"])


def test_df_wt_groupby_iter(grouped_df: frame.DataFrame):
    grouped = grouped_df.wt("weights").groupby("Group")
    groups = dict(iter(grouped))
    assert set(groups.keys()) == {"A", "B"}
    pd.testing.assert_frame_equal(
        groups["A"].obj,
        grouped_df[grouped_df["Group"] == "A"][["Group", "Value"]],
    )
    pd.testing.assert_frame_equal(
        groups["B"].obj,
        grouped_df[grouped_df["Group"] == "B"][["Group", "Value"]],
    )
    pd.testing.assert_series_equal(
        groups["A"].weights,
        grouped_df[grouped_df["Group"] == "A"]["weights"],
    )
    pd.testing.assert_series_equal(
        groups["B"].weights,
        grouped_df[grouped_df["Group"] == "B"]["weights"],
    )


def test_df_wt_groupby_iter_idx_group(grouped_df: frame.DataFrame):
    grouped_df = grouped_df.set_index("Group")
    grouped = grouped_df.wt("weights").groupby("Group")
    groups = dict(iter(grouped))
    assert set(groups.keys()) == {"A", "B"}
    pd.testing.assert_frame_equal(
        groups["A"].obj,
        grouped_df.query("Group == 'A'")[["Value"]],
    )
    pd.testing.assert_frame_equal(
        groups["B"].obj,
        grouped_df.query("Group == 'B'")[["Value"]],
    )
    pd.testing.assert_series_equal(
        groups["A"].weights,
        grouped_df.query("Group == 'A'")["weights"],
    )
    pd.testing.assert_series_equal(
        groups["B"].weights,
        grouped_df.query("Group == 'B'")["weights"],
    )


def test_df_wt_groupby_count():
    df = frame.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "Value": [10, 20, np.nan, 40],
            "weights": [1.0, 2.0, 1.5, 2.5],
        }
    )
    grouped = df.wt("weights").groupby("Group")
    expected_count_skipna = pd.DataFrame(
        {"Value": [3.0, 2.5]}, index=pd.Index(["A", "B"], name="Group")
    )
    expected_count_noskipna = pd.DataFrame(
        {"Value": [3.0, 4.0]}, index=pd.Index(["A", "B"], name="Group")
    )
    pd.testing.assert_frame_equal(grouped.count(), expected_count_skipna)
    pd.testing.assert_frame_equal(grouped.count(skipna=False), expected_count_noskipna)


def test_df_wt_groupby_sum(grouped_df: frame.DataFrame):
    grouped = grouped_df.wt("weights").groupby("Group")
    expected_sum = pd.DataFrame(
        {"Value": [50.0, 145.0]}, index=pd.Index(["A", "B"], name="Group")
    )
    pd.testing.assert_frame_equal(grouped.sum(), expected_sum)


def test_df_wt_groupby_mean(grouped_df: frame.DataFrame):
    grouped = grouped_df.wt("weights").groupby("Group")
    expected_mean = pd.DataFrame(
        {"Value": [16.666666666666668, 36.25]}, index=pd.Index(["A", "B"], name="Group")
    )
    pd.testing.assert_frame_equal(grouped.mean(), expected_mean)


def test_df_wt_groupby_column_mean(grouped_df: frame.DataFrame):
    grouped = grouped_df.wt("weights").groupby("Group")["Value"]
    expected_mean = pd.Series(
        {"A": 16.666666666666668, "B": 36.25},
        index=pd.Index(["A", "B"], name="Group"),
        name="Value",
    )
    pd.testing.assert_series_equal(grouped.mean(), expected_mean)


def test_df_wt_groupby_numeric_groups_mean():
    df = frame.DataFrame(
        {
            "Group": [1, 1, 2, 2],
            "Value": [10, 20, np.nan, 40],
            "weights": [1.0, 2.0, 1.5, 2.5],
        }
    )
    grouped = df.wt("weights").groupby("Group")
    expected_mean_skipna = pd.DataFrame(
        {"Value": [16.666666666666668, 40]}, index=pd.Index([1, 2], name="Group")
    )
    expected_mean_noskipna = pd.DataFrame(
        {"Value": [16.666666666666668, 25.0]}, index=pd.Index([1, 2], name="Group")
    )
    pd.testing.assert_frame_equal(grouped.mean(), expected_mean_skipna)
    pd.testing.assert_frame_equal(grouped.mean(skipna=False), expected_mean_noskipna)


def test_df_wt_groupby_var(grouped_df: frame.DataFrame):
    grouped = grouped_df.wt("weights").groupby("Group")
    expected_var = pd.DataFrame(
        {"Value": [33.333333333333314, 31.25]},
        index=pd.Index(["A", "B"], name="Group"),
    )
    pd.testing.assert_frame_equal(grouped.var(), expected_var)


def test_df_wt_groupby_std(grouped_df: frame.DataFrame):
    grouped = grouped_df.wt("weights").groupby("Group")
    expected_std = pd.DataFrame(
        {"Value": [5.773502691896255, 5.5901699437494745]},
        index=pd.Index(["A", "B"], name="Group"),
    )
    pd.testing.assert_frame_equal(grouped.std(), expected_std)


def test_df_wt_apply():
    df = frame.DataFrame(
        {
            "Value": [10, 20, 30, 40],
            "weights": [1.0, 2.0, 1.5, 2.5],
        }
    )

    def weighted_range(series: pd.Series) -> float:
        return series.max() - series.min()

    expected_apply_scalar = pd.Series({"Value": 90.0})
    pd.testing.assert_series_equal(
        df.wt("weights").apply(weighted_range), expected_apply_scalar
    )


def test_df_wt_groupby_apply(grouped_df: frame.DataFrame):
    df = grouped_df.set_index("Group")

    def weighted_minmax(dataframe: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {"min": dataframe.min(axis=None), "max": dataframe.max(axis=None)}
        )

    expected_apply_array = pd.DataFrame(
        {"A": {"min": 10.0, "max": 40.0}, "B": {"min": 45.0, "max": 100.0}}
    ).T.rename_axis("Group")
    pd.testing.assert_frame_equal(
        df.wt("weights").groupby("Group").apply(weighted_minmax),
        expected_apply_array,
    )


def test_df_wt_getitem_column(df: frame.DataFrame):
    df_wt = df.wt("weights")
    pd.testing.assert_series_equal(df_wt["A"].obj, df["A"])
    assert isinstance(df_wt["A"], series.WeightedSeriesAccessor)


def test_df_wt_getitem_columns(df: frame.DataFrame):
    df_wt = df.wt("weights")
    pd.testing.assert_frame_equal(df_wt[["A", "B"]].obj, df[["A", "B"]])
    assert isinstance(df_wt[["A", "B"]], frame.WeightedDataFrameAccessor)


def test_df_wt_groupby_multiple_groupings(df: frame.DataFrame):
    df = df.assign(Group=["X", "X", "Y"])
    grouped = df.wt("weights").groupby(["A", "Group"])
    assert isinstance(grouped._group_keys(), pd.MultiIndex)


def test_df_wt_groupby_select_columns(grouped_df: frame.DataFrame):
    grouped = grouped_df.wt("weights").groupby("Group")
    assert isinstance(grouped["Value"], series.WeightedSeriesGroupBy)
    assert isinstance(grouped[["Value"]], frame.WeightedFrameGroupBy)


def test_df_wt_resample_sum_count_mean():
    df = frame.DataFrame(
        {
            "Value": [1.0, 2.0, np.nan, 4.0],
            "weights": [1.0, 2.0, 3.0, 4.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )

    resampled = df.wt("weights").resample("2D")

    expected_sum = pd.DataFrame(
        {"Value": [5.0, 16.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
    )
    expected_count_skipna = pd.DataFrame(
        {"Value": [3.0, 4.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
    )
    expected_count_noskipna = pd.DataFrame(
        {"Value": [3.0, 7.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
    )
    expected_mean = pd.DataFrame(
        {"Value": [5.0 / 3.0, 4.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
    )

    pd.testing.assert_frame_equal(resampled.sum(), expected_sum)
    pd.testing.assert_frame_equal(resampled.count(), expected_count_skipna)
    pd.testing.assert_frame_equal(
        resampled.count(skipna=False), expected_count_noskipna
    )
    pd.testing.assert_frame_equal(resampled.mean(), expected_mean)


def test_df_wt_resample_var_std():
    df = frame.DataFrame(
        {
            "Value": [1.0, 2.0, np.nan, 4.0],
            "weights": [1.0, 2.0, 3.0, 4.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )

    resampled = df.wt("weights").resample("2D")

    expected_var = pd.DataFrame(
        {"Value": [1.0 / 3.0, 0.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
    )
    expected_std = pd.DataFrame(
        {"Value": [np.sqrt(1.0 / 3.0), 0.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="2D"),
    )

    pd.testing.assert_frame_equal(resampled.var(), expected_var)
    pd.testing.assert_frame_equal(resampled.std(), expected_std)


def test_df_wt_corr_perfect_linear_relationships():
    df = frame.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [2.0, 4.0, 6.0, 8.0],
            "C": [4.0, 3.0, 2.0, 1.0],
            "weights": [1.0, 2.0, 3.0, 4.0],
        }
    )

    corr = df.wt("weights").corr()
    expected = pd.DataFrame(
        {
            "A": [1.0, 1.0, -1.0],
            "B": [1.0, 1.0, -1.0],
            "C": [-1.0, -1.0, 1.0],
        },
        index=["A", "B", "C"],
    )

    pd.testing.assert_frame_equal(corr, expected)


def test_df_wt_corr_min_periods_and_numeric_only():
    df = frame.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [2.0, 4.0, np.nan, 8.0],
            "C": [4.0, 3.0, 2.0, 1.0],
            "Label": ["x", "y", "z", "w"],
            "weights": [1.0, 1.0, 1.0, 1.0],
        }
    )

    corr = df.wt("weights").corr(min_periods=4)
    expected = pd.DataFrame(
        {
            "A": [1.0, np.nan, -1.0],
            "B": [np.nan, np.nan, np.nan],
            "C": [-1.0, np.nan, 1.0],
        },
        index=["A", "B", "C"],
    )

    pd.testing.assert_frame_equal(corr, expected)


def test_df_wt_corr_unsupported_method():
    df = frame.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0], "weights": [1.0, 1.0]})

    with pytest.raises(NotImplementedError):
        df.wt("weights").corr(method="kendall")  # type: ignore[arg-type]


def test_df_wt_groupby_corr_matrices():
    df = frame.DataFrame(
        {
            "Group": ["A", "A", "A", "B", "B", "B"],
            "X": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "Y": [2.0, 4.0, 6.0, 3.0, 2.0, 1.0],
            "Z": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            "weights": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        }
    )

    corr = df.wt("weights").groupby("Group").corr()

    expected_a = pd.DataFrame(
        {
            "X": [1.0, 1.0, -1.0],
            "Y": [1.0, 1.0, -1.0],
            "Z": [-1.0, -1.0, 1.0],
        },
        index=["X", "Y", "Z"],
    )
    expected_b = pd.DataFrame(
        {
            "X": [1.0, -1.0, 1.0],
            "Y": [-1.0, 1.0, -1.0],
            "Z": [1.0, -1.0, 1.0],
        },
        index=["X", "Y", "Z"],
    )
    expected = pd.concat({"A": expected_a, "B": expected_b}, names=["Group"])

    pd.testing.assert_frame_equal(corr, expected)


def test_df_wt_groupby_corr_unsupported_method():
    df = frame.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [2.0, 1.0, 4.0, 3.0],
            "weights": [1.0, 1.0, 1.0, 1.0],
        }
    )

    with pytest.raises(NotImplementedError):
        df.wt("weights").groupby("Group").corr(method="kendall")  # type: ignore[arg-type]


def test_df_wt_groupby_corr_min_periods_pairwise_complete():
    df = frame.DataFrame(
        {
            "Group": ["A", "A", "A", "B", "B", "B"],
            "X": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "Y": [2.0, 4.0, np.nan, 3.0, np.nan, 1.0],
            "Z": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            "Label": ["u", "v", "w", "x", "y", "z"],
            "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    corr = df.wt("weights").groupby("Group").corr(min_periods=3)

    expected_a = pd.DataFrame(
        {
            "X": [1.0, np.nan, -1.0],
            "Y": [np.nan, np.nan, np.nan],
            "Z": [-1.0, np.nan, 1.0],
        },
        index=["X", "Y", "Z"],
    )
    expected_b = pd.DataFrame(
        {
            "X": [1.0, np.nan, 1.0],
            "Y": [np.nan, np.nan, np.nan],
            "Z": [1.0, np.nan, 1.0],
        },
        index=["X", "Y", "Z"],
    )
    expected = pd.concat({"A": expected_a, "B": expected_b}, names=["Group"])

    pd.testing.assert_frame_equal(corr, expected)
