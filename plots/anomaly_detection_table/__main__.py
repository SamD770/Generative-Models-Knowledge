import pandas as pd
import numpy as np

x = np.random.rand(25).reshape(5, 5)
print(pd.DataFrame(x).to_latex())
