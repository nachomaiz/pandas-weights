# `pandas-weights`

[![CI status](https://github.com/nachomaiz/pandas-weights/actions/workflows/ci.yml/badge.svg)](https://github.com/nachomaiz/pandas-weights/actions/workflows/ci.yml)
[![GitHub Release](https://img.shields.io/github/v/release/nachomaiz/pandas-weights)](https://github.com/nachomaiz/pandas-weights/releases)
[![License](https://img.shields.io/github/license/nachomaiz/pandas-weights)](https://github.com/nachomaiz/pandas-weights/blob/main/LICENSE)

`pandas-weights` is a Python library that extends the functionality of [`pandas`](https://pandas.pydata.org/) by providing tools to handle weighted data in DataFrames. It allows users to easily apply weights to their data for statistical analysis and aggregation.

The library introduces a new accessor, `wt`, which can be used on `pandas.DataFrame` and `pandas.Series` objects to perform weighted operations such as weighted mean, weighted sum, and more.

The project utilizes native `pandas` vectorized operations as much as possible to ensure efficiency while maintaining readability, but it is not optimized for performance. It rather focuses on providing a simple and intuitive interface for working with weighted data in `pandas`. Contributions are welcome to enhance its functionality and performance.

## Installation

`pandas-weights` requires Python 3.11 or higher and depends on `pandas` and `numpy`. This library will target [`pandas` supported Python versions](https://pandas.pydata.org/docs/getting_started/install.html#python-version-support).

Due to a change in the `pandas` API for custom accessors, `pandas-weights` is not compatible with `pandas` versions prior to 3.0.0.

Future releases will be able to be installed using `pip` or `uv`:

```shell
pip install pandas-weights
```

For now, you can install it directly from the GitHub repository:

```shell
pip install git+https://github.com/nachomaiz/pandas-weights.git
```

## Usage

To use `pandas-weights`, you need to import it alongside `pandas`.

Then, you can create a DataFrame and use the `wt` accessor to apply weights to your data. This accessor is available on both `pandas.DataFrame` as well as `pandas.Series` objects.

You will need to define the weights for the aggregation or analysis you want to perform. To do so, call the `wt` accessor on your DataFrame and pass the name of the column containing the weights, or an array of weights.

If a column name is provided, it must exist in the `DataFrame`, and the results from aggregation functions will not include the weights column. In groupby operations, the weights column will not be included in the data.

> [!important]
> Only numeric columns (`int`, `float`, `bool`) are supported when using weights. Non-numeric columns will be ignored during weighted operations. If using a `pandas.Series`, only numeric data is supported.

### Supported Methods

Currently, only the following functionality is implemented (both for `pandas.DataFrame` and `pandas.Series`):

- Using the `wt` accessor:
  - `count()`
  - `mean()`
  - `sum()`
  - `var()`
  - `std()`
  - `corr(...)`
  - `resample(...)` (returns a weighted resampler object)
  - `apply(func, ...)`
  - `groupby(...)` (returns a weighted groupby object)
  - `df.wt(...)[col]` (returns a weighted Series accessor for the specified column)
  - `df.wt(...)[[col1, col2, ...]]` (returns a weighted DataFrame accessor for the specified columns)
- Using `groupby` with the `wt` accessor:
  - `count()`
  - `mean()`
  - `sum()`
  - `var()`
  - `std()`
  - `corr(...)` for DataFrame groupby
  - `corr(other, ...)` for Series groupby
  - `apply(func, ...)`
  - `df.wt(...).groupby(...)[col]` (returns a weighted Series groupby object for the specified column)
  - `df.wt(...).groupby(...)[[col1, col2, ...]]` (returns a weighted DataFrame groupby object for the specified columns)
  - `for group_name, group_data in df.wt(...).groupby(...): ...` (iterating over groups)
    - where `group_data` is a weighted DataFrame accessor to the group's data.
- Using `resample` with the `wt` accessor:
  - `count()`
  - `mean()`
  - `sum()`
  - `var()`
  - `std()`

> [!warning]
> `.wt`,  `.wt.groupby` and `.wt.resample` do not support all weighted aggregation functions that `pandas` provides. If you attempt to use an unsupported function, it will raise an `AttributeError`.

> [!note]
> For weighted Series groupby correlation (`series.wt(...).groupby(...).corr(other)`), index alignment follows `pandas` semantics.
> The grouped Series and `other` are aligned by index within each group before correlation is computed.
> When duplicate index labels are present, this can produce repeated pairings and results that may be surprising at first glance.

Example with duplicate index labels:

```python
import pandas as pd
import pandas_weights

idx = pd.Index(["A", "A", "B", "B"], name="Group")
s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="values")
weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
other = pd.Series([2.0, 4.0, 8.0, 6.0], index=idx)

# Correlation is computed per group after index alignment with `other`.
# With duplicate labels, alignment can create repeated pairings.
result = s.wt(weights).groupby("Group").corr(other)
print(result)

# Output
# Group
# A    0.0
# B    0.0
# Name: values, dtype: float64
```

If you want intuitive row-by-row pairing inside each group, make sure indices are unique within each group. For example, using a `MultiIndex`:

```python
import pandas as pd
import pandas_weights

idx = pd.MultiIndex.from_tuples(
    [("A", 0), ("A", 1), ("B", 0), ("B", 1)],  # each row has a unique index within its group
    names=["Group", "row_id"],
)
s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="values")
weights = pd.Series([1.0, 2.0, 1.5, 2.5], index=idx)
other = pd.Series([2.0, 4.0, 8.0, 6.0], index=idx)

# Now alignment is one-to-one by (Group, row_id), so results match
# the intuitive within-group pairings.
result = s.wt(weights).groupby("Group").corr(other)
print(result)

# Output
# Group
# A    1.0
# B   -1.0
# Name: values, dtype: float64
```

### Examples

```python
import pandas as pd
import pandas_weights

# Create a sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'weights': [0.1, 0.2, 0.3, 0.4, 0.5]
}
df = pd.DataFrame(data)  # or use `pandas_weights.DataFrame(data)` for typed `wt` accessor

# perform a weighted average
result = df.wt("weights").mean()
print(result)

# On a DataFrame you can also access specific columns after applying weights
result_A = df.wt("weights")['A'].mean()
result_AB = df.wt("weights")[['A', 'B']].mean()
```

You can also pass an array of weights directly:

```python
import pandas as pd
import pandas_weights

# Create a sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1]
}
df = pd.DataFrame(data)

# Define weights
weights = [0.1, 0.2, 0.3, 0.4, 0.5]  # can be a list, numpy array, or pandas Series
# Must be the same length as the DataFrame.
# If using a pandas Series, as long as the length matches, the index will be overwritten
# with the DataFrame's index.

# Apply weights and perform a weighted average
result = df.wt(weights).mean()
print(result)
```

There's an option to set a default weight for missing weight values using the `na_weight` parameter:

```python
import pandas as pd
import pandas_weights

# Create a sample DataFrame
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}
df = pd.DataFrame(data)

# Define weights with a missing value
weights_with_na = [0.5, None, 2.0]

# Apply weights with a default weight for missing values
df.wt(weights_with_na, na_weight=1.0)
```

The same operations are available when using a `pandas.Series`:

```python
import pandas as pd
import pandas_weights

# Create a sample Series
data = [1, 2, 3, 4, 5]
weights = [0.1, 0.2, 0.3, 0.4, 0.5]
series = pd.Series(data)  # or use `pandas_weights.Series(data)` for typed `wt` accessor

# Apply weights and perform a weighted average
result = series.wt(weights).mean()
print(result)
```

Finally, you can use `apply` to apply custom functions to weighted data:

> [!tip]
> `apply` works similarly to `pandas.DataFrame.apply` and `pandas.DataFrame.groupby.apply`, but the function will receive pre-weighted data instead.

```python
import pandas as pd
import pandas_weights

# Create a sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'weights': [0.1, 0.2, 0.3, 0.4, 0.5]
}
df = pd.DataFrame(data)

# Define a custom function
def value_range(x: pd.Series) -> float:
    return x.max() - x.min()

# Apply weights and use the custom function
result = df.wt("weights").apply(value_range)
print(result)
```

### GroupBy Functionality

Use the `.groupby` method on `pandas.Series` and `pandas.DataFrame` to perform weighted aggregations on grouped data:

```python
import pandas as pd
import pandas_weights

# Create a sample DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'C'],
    'Values': [10, 20, 30, 40, 50],
    'weights': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# Perform weighted groupby operations
result = df.wt("weights").groupby('Category').mean()
print(result)
```

Groupby iteration is also supported:

```python
import pandas as pd
import pandas_weights

# Create a sample DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'C'],
    'Values': [10, 20, 30, 40, 50],
    'weights': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# iterate over groups (group_data is a weighted DataFrame accessor)
for group_name, group_data in df.wt("weights").groupby("Category"):
    print(f"Group: {group_name}", f"Weighted Mean: {group_data.mean()}")
```

It's also possible to access specific columns after applying weights in a groupby operation:

```python
import pandas as pd
import pandas_weights

# Create a sample DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'C'],
    'Values': [10, 20, 30, 40, 50],
    'weights': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# Perform weighted groupby operations on specific column(s)
result_values = df.wt("weights").groupby('Category')['Values'].mean()
result_values_df = df.wt("weights").groupby('Category')[['Values']].mean()
```

## Contributing

Contributions are welcome! If you would like to contribute to `pandas-weights`, please fork the repository and create a pull request with your changes. Make sure to include tests for any new functionality you add.

`pandas` functionality is very extensive, so if you would like to see support for additional *weighted* aggregation functions or features, please open an issue or submit a pull request.

Some areas for potential contributions include:

- Implementing additional weighted aggregation functions (e.g., weighted median, weighted quantiles, etc.)
- Adding support for weighted correlation and covariance.
- Adding support for weighted rolling and expanding window functions.
- Improving performance of existing functions.
- More aggregation functions for weighted and weighted groupby accessors.

### Development Environment

`pandas-weights` uses `uv` to manage the development environment and dependencies. You can use `uv` to create a virtual environment and install the required dependencies for development.

To set up a development environment for `pandas-weights`, you can follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/nachomaiz/pandas-weights.git
    cd pandas-weights
    ```

1. Create a virtual environment and activate it:

    ```shell
    uv venv
    uv activate
    ```

1. Install the development dependencies:

    ```shell
    uv sync --group dev --group test
    ```

1. You can now run tests and make changes to the code. To run tests, use:

    ```shell
    uv run pytest
    ```

## License

`pandas-weights` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
