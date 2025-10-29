from typing import TypeAlias, Union

import numpy as np
import pandas as pd

D1NumericArray: TypeAlias = Union[list[Union[bool, int, float]], pd.Series, np.ndarray]
