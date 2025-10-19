import numpy as np
import pandas as pd
import pytest

from pandas_weights import frame


def test_df_wt_error_on_no_weights_set():
    df = frame.DataFrame()
    with pytest.raises(ValueError, match="Weights have not been set"):
        df.wt.weighted()


def test_df_wt_error_on_weights_length_mismatch():
    df = frame.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="Length of weights must match number of rows in DataFrame"
    ):
        df.wt(np.array([1, 2]))


def test_df_wt_error_on_weights_not_1d():
    df = frame.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError, match="weights must be one-dimensional"):
        df.wt(np.array([[1], [2], [3]]))


@pytest.mark.parametrize(
    "weights_types",
    [[0.5, 1.5, 2.0], pd.Series([0.5, 1.5, 2.0]), np.array([0.5, 1.5, 2.0])],
)
def test_df_wt_weighted(weights_types: list[float] | pd.Series | np.ndarray):
    df = frame.DataFrame({"A": [1, 2, 3]})
    df_wt = df.wt(weights_types)
    assert np.array_equal(df_wt.weights, weights_types)
    assert np.array_equal(df_wt.weighted()["A"], df["A"] * weights_types)


def test_df_wt_weighted_column():
    df = frame.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "weights": [0.5, 1.5, 2.0]})
    assert np.array_equal(
        df.wt("weights").weighted(), df[["A", "B"]].mul(df["weights"], axis=0)
    )


def test_df_wt_init_weight():
    df = frame.DataFrame({"A": [1, 2, 3]})
    weights_series = pd.Series([0.5, 1.5, 2.0])
    df_wt = frame.WeightedDataFrameAccessor._init_validated(df, weights_series)
    assert np.array_equal(df_wt.weights, weights_series)
    assert np.array_equal(df_wt.weighted()["A"], df["A"] * weights_series)


def test_df_wt_T():
    df = frame.DataFrame({"A": [1, 2], "B": [3, 4]})
    weights_series = pd.Series([0.5, 1.5])
    expected = df[["A", "B"]].mul(weights_series, axis=0).T
    pd.testing.assert_frame_equal(df.wt(weights_series).T, expected)


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


def test_df_wt_sum():
    df = frame.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights_series = pd.Series([0.5, 1.5, 2.0])
    df_wt = df.wt(weights_series)

    expected_sum = pd.Series({"A": 9.5, "B": 21.5})

    pd.testing.assert_series_equal(df_wt.sum(axis=0), expected_sum)


def test_df_wt_sum_min_count():
    df = frame.DataFrame({"A": [1, 2, None], "B": [None, None, None]}).astype(float)
    weights_series = pd.Series([0.5, 1.5, 2.0])
    df_wt = df.wt(weights_series)

    expected_sum = pd.Series({"A": 3.5, "B": None})

    pd.testing.assert_series_equal(df_wt.sum(axis=0, min_count=1), expected_sum)


def test_df_wt_mean():
    df = frame.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights_series = pd.Series([0.5, 1.5, 2.0])
    df_wt = df.wt(weights_series)

    expected_mean = pd.Series({"A": 2.375, "B": 5.375})

    pd.testing.assert_series_equal(df_wt.mean(axis=0), expected_mean)


def test_df_wt_var():
    df = frame.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights_series = pd.Series([0.5, 1.5, 2.0])
    df_wt = df.wt(weights_series)

    expected_var = pd.Series({"A": 0.8072916666666666, "B": 0.8072916666666666})

    pd.testing.assert_series_equal(df_wt.var(axis=0), expected_var)


def test_df_wt_std():
    df = frame.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights_series = pd.Series([0.5, 1.5, 2.0])
    df_wt = df.wt(weights_series)

    expected_std = pd.Series({"A": 0.898494110535326, "B": 0.898494110535326})

    pd.testing.assert_series_equal(df_wt.std(axis=0), expected_std)


def test_df_wt_groupby_init():
    df = frame.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "Value": [10, 20, 30, 40],
            "weights": [1.0, 2.0, 1.5, 2.5],
        }
    )
    grouped = df.wt("weights").groupby("Group")
    assert isinstance(grouped, frame.WeightedFrameGroupBy)
    assert np.array_equal(grouped.weights, df["weights"])


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


def test_df_wt_groupby_sum():
    df = frame.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "Value": [10, 20, 30, 40],
            "weights": [1.0, 2.0, 1.5, 2.5],
        }
    )
    grouped = df.wt("weights").groupby("Group")
    expected_sum = pd.DataFrame(
        {"Value": [50.0, 145.0]}, index=pd.Index(["A", "B"], name="Group")
    )
    pd.testing.assert_frame_equal(grouped.sum(), expected_sum)


def test_df_wt_groupby_mean():
    df = frame.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "Value": [10, 20, 30, 40],
            "weights": [1.0, 2.0, 1.5, 2.5],
        }
    )
    grouped = df.wt("weights").groupby("Group")
    expected_mean = pd.DataFrame(
        {"Value": [16.666666666666668, 36.25]}, index=pd.Index(["A", "B"], name="Group")
    )
    pd.testing.assert_frame_equal(grouped.mean(), expected_mean)


def test_df_wt_groupby_var():
    df = frame.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "Value": [10, 20, 30, 40],
            "weights": [1.0, 2.0, 1.5, 2.5],
        }
    )
    grouped = df.wt("weights").groupby("Group")
    expected_var = pd.DataFrame(
        {"Value": [294.4444444444444, 1380.2083333333333]},
        index=pd.Index(["A", "B"], name="Group"),
    )
    pd.testing.assert_frame_equal(grouped.var(), expected_var)


def test_df_wt_groupby_std():
    df = frame.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "Value": [10, 20, 30, 40],
            "weights": [1.0, 2.0, 1.5, 2.5],
        }
    )
    grouped = df.wt("weights").groupby("Group")
    expected_std = pd.DataFrame(
        {"Value": [17.159383568311664, 37.151155208597935]},
        index=pd.Index(["A", "B"], name="Group"),
    )
    pd.testing.assert_frame_equal(grouped.std(), expected_std)
