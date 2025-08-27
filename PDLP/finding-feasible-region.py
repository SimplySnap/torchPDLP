''' 
Load MPS files and find bounds for pattern spotting
l is clipped to 0 if no region. Can go below but likely will not
What happens if we get u = -3, l = None (0)??
Then LP infeasible, detected earlier so not deep

 '''
import pandas as pd
import torch
from .util import mps_to_standard_form


#  Load files
mps_file_path = 'datasets/netlib/feasible' #  Use netlib feasible datasets
mps_files = sorted([f for f in os.listdir(mps_folder_path) if f.endswith('.mps')])

#  Init df
df = {}
df = pd.DataFrame

#  Scan each file
for file in mps_files
    c, K, q, m_ineq, l, u= mps_to_standard_form(file, support_sparse=False, verbose=False)

    #  Count number of NaN entries TODO


    #  Count how entries are positive, how many are negative TODO
    l_pos = None
    l_neg = None
    u_pos = None
    u_neg = None

    #  Append to dataframe - NaN's (unbounded), l_neg, l_pos, u_neg, u_pos, percent positive primal-space TODO

#  Sum columns and calculate total stats TODO

