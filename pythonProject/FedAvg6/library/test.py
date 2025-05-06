import numpy as np
from statsmodels.tools import add_constant
x_vec = np.array([1,2,3,4])
print(x_vec)
add_constant(x_vec)
print(add_constant(x_vec))