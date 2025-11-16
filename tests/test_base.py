from typing import Union

import numpy as np
import pandas as pd
import pytest

from pandas_weights.base import BaseWeightedAccessor


def test_get_weights():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    weights_series = pd.Series([0.5, 1.5, 2.0])

    accessor = BaseWeightedAccessor._init_validated(df, weights_series)

    pd.testing.assert_series_equal(accessor.weights, weights_series)


def test_get_weights_not_set():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    accessor = BaseWeightedAccessor(df)

    with pytest.raises(ValueError) as exc_info:
        _ = accessor.weights

    assert (
        str(exc_info.value) == "Weights have not been set. Set weights with `.wt(...)`."
    )


@pytest.mark.parametrize(
    "weights", [[0.5, 1.5, 2.0], pd.Series([0.5, 1.5, 2.0]), np.array([0.5, 1.5, 2.0])]
)
def test_set_weights_various_types(weights: Union[list[float], pd.Series, np.ndarray]):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    accessor = BaseWeightedAccessor(df)
    accessor.weights = weights
    pd.testing.assert_series_equal(accessor.weights, pd.Series(weights, index=df.index))
