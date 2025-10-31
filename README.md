# `pandas-weights`

[![cov](https://nachomaiz.github.io/pandas-weights/badges/coverage.svg)](https://github.com/nachomaiz/pandas-weights/actions)

`pandas-weights` is a Python library that extends the functionality of `pandas` by providing tools to handle weighted data in DataFrames. It allows users to easily apply weights to their data for statistical analysis and aggregation.

The library introduces a new accessor, `wt`, which can be used on `pandas.DataFrame` and `pandas.Series` objects to perform weighted operations such as weighted mean, weighted sum, and more.

The project utilizes native `pandas` vectorized operations as much as possible to ensure efficiency while maintaining readability, but it is not optimized for performance. It rather focuses on providing a simple and intuitive interface for working with weighted data in `pandas`. Contributions are welcome to enhance its functionality and performance.

## Installation

`pandas-weights` requires Python 3.9 or higher and depends on `pandas` and `numpy`. This library will target `pandas` supported Python versions.

You can install `pandas-weights` using pip:

```bash
pip install pandas-weights
```

## Usage

To use `pandas-weights`, you need to import it alongside `pandas`.

Then, you can create a DataFrame and use the `wt` accessor to apply weights to your data. This accessor is available on both `pandas.DataFrame` as well as `pandas.Series` objects.

You will need to define the weights for the aggregation or analysis you want to perform. To do so, call the `wt` accessor on your DataFrame and pass the name of the column containing the weights, or an array of weights.

If a column name is provided, it must exist in the `DataFrame`, and the results from aggregation functions will not include the weights column. In groupby operations, the weights column will not be included in the data.

> [!important]
> Only numeric columns are supported when using weights. Non-numeric columns will be ignored during weighted operations. If using a `pandas.Series`, only numeric data is supported.

### Supported Methods

Currently, only the following functionality is implemented:

- Using the `wt` accessor:
  - `count()`
  - `mean()`
  - `sum()`
  - `var()`
  - `std()`
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
  - `apply(func, ...)`
  - `df.wt(...).groupby(...)[col]` (returns a weighted Series groupby object for the specified column)
  - `df.wt(...).groupby(...)[[col1, col2, ...]]` (returns a weighted DataFrame groupby object for the specified columns)
  - `for group_name, group_data in df.wt(...).groupby(...): ...` (iterating over groups)
    - where `group_data` is a weighted DataFrame accessor to the group's data.

> [!warning]
> `.wt` and `.wt.groupby` do not support all weighted aggregation functions that `pandas` provides. If you attempt to use an unsupported function, it will raise an `AttributeError`.

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

## License

`pandas-weights` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgements

We would like to thank the developers of the `pandas` library for their incredible work, which serves as the foundation for `pandas-weights`.
