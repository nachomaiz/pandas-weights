from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

Number: TypeAlias = bool | int | float
D1NumericArray: TypeAlias = (
    list[Number]
    | pd.Series
    | np.ndarray[tuple[Literal[1]], np.dtype[np.number | np.bool]]
)
