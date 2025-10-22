# `pandas-weights`

`pandas-weights` is a Python library that extends the functionality of `pandas` by providing tools to handle weighted data in DataFrames. It allows users to easily apply weights to their data for statistical analysis and aggregation.

The library introduces a new accessor, `wt`, which can be used on `pandas.DataFrame` and `pandas.Series` objects to perform weighted operations such as weighted mean, weighted sum, and more.

The project utilizes native `pandas` vectorized operations as much as possible to ensure efficiency while maintaining readability, but it is not optimized for performance. It rather focuses on providing a simple and intuitive interface for working with weighted data in `pandas`. Contributions are welcome to enhance its functionality and performance.

## Installation

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

Here is a simple example:

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

Use the `.groupby` method to perform weighted aggregations on grouped data:

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

You can also perform the same operation using a `pandas.Series`:

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

## Supported Functions

Currently, only the following aggregation functions are implemented:

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
  - `for group_name, group_data in df.wt(...).groupby(...): ...` (iterating over groups)
    - where `group_data` is a weighted DataFrame accessor to the group's data.

> [!warning]
> `.wt` and `.wt.groupby` do not support all weighted aggregation functions that `pandas` provides. If you attempt to use an unsupported function, it will raise an `AttributeError`.

## Contributing

Contributions are welcome! If you would like to contribute to `pandas-weights`, please fork the repository and create a pull request with your changes. Make sure to include tests for any new functionality you add.

`pandas` functionality is very extensive, so if you would like to see support for additional *weighted* aggregation functions or features, please open an issue or submit a pull request.

## License

`pandas-weights` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgements

We would like to thank the developers of the `pandas` library for their incredible work, which serves as the foundation for `pandas-weights`.
