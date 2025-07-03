import numpy as np
import torch

#Make random 10,000*10,000 matrix
n = 10000
m = 10000

mat1 = np.random.random_integers(-5,5,(n,m))
mat2 = np.random.random_integers(-5,5,(n,m))

mat = mat1@mat2
