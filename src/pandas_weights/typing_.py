from typing import TypeAlias

import numpy as np
import pandas as pd

D1NumericArray: TypeAlias = list[bool | int | float] | pd.Series | np.ndarray
