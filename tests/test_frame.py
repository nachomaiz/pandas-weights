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
    "weights_types",
    [[0.5, 1.5, 2.0], pd.Series([0.5, 1.5, 2.0]), np.array([0.5, 1.5, 2.0])],
)
def test_df_wt_weighted(
    df: frame.DataFrame, weights_types: list[float] | pd.Series | np.ndarray
):
    df_wt = df.wt(weights_types)
    assert np.array_equal(df_wt.weights, weights_types)
    assert np.array_equal(df_wt.weighted()["A"], df["A"] * weights_types)


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

    expected_var = pd.Series({"A": 0.8072916666666666, "B": 0.8072916666666666})

    pd.testing.assert_series_equal(df_wt.var(axis=0), expected_var)


def test_df_wt_std(df: frame.DataFrame):
    df_wt = df.wt("weights")

    expected_std = pd.Series({"A": 0.898494110535326, "B": 0.898494110535326})

    pd.testing.assert_series_equal(df_wt.std(axis=0), expected_std)


def test_df_wt_groupby_init(grouped_df: frame.DataFrame):
    grouped = grouped_df.wt("weights").groupby("Group", observed=False, axis=0)
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


def test_df_wt_groupby_var(grouped_df: frame.DataFrame):
    grouped = grouped_df.wt("weights").groupby("Group")
    expected_var = pd.DataFrame(
        {"Value": [294.4444444444444, 1380.2083333333333]},
        index=pd.Index(["A", "B"], name="Group"),
    )
    pd.testing.assert_frame_equal(grouped.var(), expected_var)


def test_df_wt_groupby_std(grouped_df: frame.DataFrame):
    grouped = grouped_df.wt("weights").groupby("Group")
    expected_std = pd.DataFrame(
        {"Value": [17.159383568311664, 37.151155208597935]},
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
