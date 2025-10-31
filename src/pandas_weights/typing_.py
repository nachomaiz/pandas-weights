from typing import Union

import numpy as np
import pandas as pd

Number = Union[bool, int, float]
D1NumericArray = Union[list[Number], pd.Series, np.ndarray]
