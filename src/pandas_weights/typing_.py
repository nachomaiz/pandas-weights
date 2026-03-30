from typing import TypeAlias, Union

import numpy as np
import pandas as pd

Number: TypeAlias = Union[bool, int, float]
D1NumericArray: TypeAlias = Union[list[Number], pd.Series, np.ndarray]
