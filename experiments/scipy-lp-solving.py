import numpy as np
import pandas as pd
import time
import scipy.optimize as opt
from collections import defaultdict
import torch
import os
import sys

def mps_to_standard_form(mps_file, device='cpu', verbose=True):
    """
    Parses an MPS file and returns the standard form LP components as numpy arrays:

    Returns: G,A,h,A,b,l,u as numpy arrays
    """

    #Read MPS file
    with open(mps_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('*')]

    section = None
    row_types = {}
    row_indices = {}
    col_data = defaultdict(list)
    rhs_data = {}
    range_data = {}
    bound_data = defaultdict(dict)

    row_counter = 0
    var_names = []
    seen_vars = set()
    obj_row_name = None

    for line in lines:
        if line == 'NAME' or line == 'ENDATA':
            continue
        elif line == 'ROWS':
            section = 'ROWS'
            continue
        elif line == 'COLUMNS':
            section = 'COLUMNS'
            continue
        elif line == 'RHS':
            section = 'RHS'
            continue
        elif line == 'RANGES':
            section = 'RANGES'
            continue
        elif line == 'BOUNDS':
            section = 'BOUNDS'
            continue

        tokens = line.split()
        if section == 'ROWS':
            sense, row_name = tokens
            row_types[row_name] = sense
            row_indices[row_name] = row_counter
            if sense == 'N':
                obj_row_name = row_name
            row_counter += 1

        elif section == 'COLUMNS':
            var_name = tokens[0]
            if var_name not in seen_vars:
                var_names.append(var_name)
                seen_vars.add(var_name)
            for i in range(1, len(tokens), 2):
                row, val = tokens[i], float(tokens[i + 1])
                col_data[var_name].append((row, val))

        elif section == 'RHS':
            for i in range(1, len(tokens), 2):
                row, val = tokens[i], float(tokens[i + 1])
                rhs_data[row] = val

        elif section == 'RANGES':
            for i in range(1, len(tokens), 2):
                row, val = tokens[i], float(tokens[i + 1])
                range_data[row] = val

        elif section == 'BOUNDS':
            bound_type, _, var_name = tokens[:3]
            val = float(tokens[3]) if len(tokens) > 3 else None
            if bound_type == 'LO':
                bound_data[var_name]['lo'] = val
            elif bound_type == 'UP':
                bound_data[var_name]['up'] = val
            elif bound_type == 'FX':
                bound_data[var_name]['lo'] = val
                bound_data[var_name]['up'] = val
            elif bound_type == 'FR':
                bound_data[var_name]['lo'] = 0.0
                bound_data[var_name]['up'] = float('inf')

    # Final variable ordering and index mapping
    var_index = {v: i for i, v in enumerate(var_names)}
    num_vars = len(var_names)

    # Build objective vector c
    c = np.zeros(num_vars)
    for var, entries in col_data.items():
        col_idx = var_index[var]
        for row_name, val in entries:
            if row_name == obj_row_name:
                c[col_idx] = val

    # Build row vectors from col_data
    row_vectors = {row: np.zeros(num_vars) for row in row_types}
    for var, entries in col_data.items():
        col_idx = var_index[var]
        for row_name, val in entries:
            row_vectors[row_name][col_idx] = val

     # Build A (equality) and G (inequality)
    A_rows, b_eq = [], []
    G_rows, h_ineq = [], []

    for row_name, sense in row_types.items():
        if row_name == obj_row_name:
            continue

        row_vec = row_vectors[row_name]
        rhs_val = rhs_data.get(row_name, 0.0)
        range_val = range_data.get(row_name, None)

        if range_val is not None:
            if sense == 'G':
                lb = rhs_val
                ub = rhs_val + abs(range_val)
            elif sense == 'L':
                ub = rhs_val
                lb = rhs_val - abs(range_val)
            elif sense == 'E':
                if range_val > 0:
                    lb = rhs_val
                    ub = rhs_val + range_val
                else:
                    ub = rhs_val
                    lb = rhs_val + range_val
            else:
                raise ValueError(f"Unsupported ranged sense: {sense}")

            G_rows.append(row_vec)
            h_ineq.append(lb)
            G_rows.append(-row_vec)
            h_ineq.append(-ub)

        else:
            if sense == 'E':
                A_rows.append(row_vec)
                b_eq.append(rhs_val)
            elif sense == 'G':
                G_rows.append(row_vec)
                h_ineq.append(rhs_val)
            elif sense == 'L':
                G_rows.append(-row_vec)
                h_ineq.append(-rhs_val)

    # Bounds
    l = []
    u = []
    for var in var_names:
        lo = bound_data[var].get('lo', 0)
        up = bound_data[var].get('up', float('inf'))
        l.append(lo)
        u.append(up)

    # Convert all to torch
    A = np.array(A_rows)
    b = np.array(b_eq)
    G = np.array(G_rows)
    h = np.array(h_ineq)
    c = np.array(c)
    l = np.array(l)
    u = np.array(u)
    #Format empty arrays
    if A is not None and (A.size == 0 or A.shape[0] == 0):
        A = None
        b = None
    if G is not None and (G.size == 0 or G.shape[0] == 0):
        G = None
        h_ineq = None
    if c is not None and c.size == 0:
        c = None
    if l is not None and l.size == 0:
        l = None
    if u is not None and u.size == 0:
        u = None
        
    
    return G,A,h,A,b,l,u,c


results = []

mps_folder = 'datasets/Netlib/feasible'

#  List all files in the folder (adjust extension if needed)
mps_files = [f for f in os.listdir(mps_folder) if f.endswith('.mps')]  

for mps_file in mps_files:
    mps_path = os.path.join(mps_folder, mps_file) #  Get directory path of the mps file

    #  Extract problem data
    G,A,h,A,b,l,u,c = mps_to_standard_form(mps_path)

    
    start_time = time.time()
    #  Solve the LP
    result = opt.linprog(c, A_ub=G, b_ub=h, A_eq=A, b_eq=b, 
                bounds=list(zip(l, u)),
                method='highs', 
                options={'maxiter': 1000000, 'disp': True})
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    results.append({
            'mps_file': mps_file,
            'time_taken_sec': elapsed_time,
            'solution_found': result.success
        })
    #  Convert results to a DataFrame
df = pd.DataFrame(results)

#  Save results to CSV
df.to_csv('lp_solve_results.csv', index=False)
