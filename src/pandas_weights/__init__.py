import importlib.metadata

from pandas_weights import frame as frame
from pandas_weights import series as series
from pandas_weights.frame import DataFrame as DataFrame
from pandas_weights.series import Series as Series

__all__ = ("DataFrame", "Series", "frame", "series")

__version__ = importlib.metadata.version("pandas-weights")
