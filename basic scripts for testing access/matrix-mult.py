import numpy as np
import time
from timeit import default_timer as timer

#Make random 10,000*10,000 matrix
start = timer() #Start of timer
n = 10000
m = 10000

mat1 = np.random.randint(-5,5,(n,m))
mat2 = np.random.randint(-5,5,(n,m))

mat = mat1@mat2
end = timer() #end of timer

print( mat[:5], (end-start))

