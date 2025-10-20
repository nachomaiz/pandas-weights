import numpy as np
import pandas as pd

D1NumericArray = list[bool | int | float] | pd.Series | np.ndarray
D2NumericArray = list[D1NumericArray] | pd.DataFrame | np.ndarray
