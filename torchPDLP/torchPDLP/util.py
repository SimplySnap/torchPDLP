import torch
import numpy as np
from collections import defaultdict
import time

class Timer:
  """
  Timer class to measure execution time of code blocks.
  Usage:
  
    with Timer("Label"):
        # Code block to be timed

  Output:
    Label: <time in seconds> seconds
  """
  def __init__(self, label="Elapsed time"):
      self.label = label

  def __enter__(self):
      self.start = time.perf_counter()
      return self

  def __exit__(self, *args):
      self.end = time.perf_counter()
      self.elapsed = self.end - self.start
      print(f"{self.label}: {self.elapsed:.6f} seconds")

def sparse_vs_dense(A, device='cpu', kkt_passes=10):
    """
    Benchmarks matrix-vector multiplication using dense and sparse formats.
    
    Parameters:
        A (torch.Tensor): 2D matrix (dense tensor)
        device (str): 'cpu' or 'cuda'
        kkt_passes (even int): Number of repetitions for matrix multiplication

    Returns:
        tensor A as either sparse or dense, which ever is faster
    """
    assert A.dim() == 2, "Input must be a 2D matrix"
    m, n = A.shape
    A = A.to(device)
    
     # Precompute random vectors for fair timing
    vecs_n = [torch.randn(n, 1, device=device) for _ in range(kkt_passes // 2)]
    vecs_m = [torch.randn(m, 1, device=device) for _ in range(kkt_passes // 2)]

    # Dense timing
    A_transpose = A.t()
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for vec_n, vec_m in zip(vecs_n, vecs_m):
        _ = A @ vec_n
        _ = A_transpose @ vec_m
    if device == 'cuda':
        torch.cuda.synchronize()
    dense_time = time.time() - start

    # Sparse timing
    A_sparse = A.to_sparse()
    A_sparse_transpose = A_sparse.t()
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for vec_n, vec_m in zip(vecs_n, vecs_m):
        _ = torch.sparse.mm(A_sparse, vec_n)
        _ = torch.sparse.mm(A_sparse_transpose, vec_m)
    if device == 'cuda':
        torch.cuda.synchronize()
    sparse_time = time.time() - start

    return A_sparse if sparse_time < dense_time else A

def mps_to_standard_form(mps_file, device='cpu', support_sparse=False, verbose=False):
    """
    Parses an MPS file and returns the standard form LP components as PyTorch tensors:
        minimize     cᵀx
        subject to   G x ≥ h
                     A x = b
                     l ≤ x ≤ u

    Returns: c, G, h, A, b, l, u
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
    A_tensor = torch.tensor(np.array(A_rows), dtype=torch.float32, device=device)
    b_tensor = torch.tensor(np.array(b_eq), dtype=torch.float32, device=device).view(-1, 1)
    G_tensor = torch.tensor(np.array(G_rows), dtype=torch.float32, device=device)
    h_tensor = torch.tensor(np.array(h_ineq), dtype=torch.float32, device=device).view(-1, 1)
    c_tensor = torch.tensor(c, dtype=torch.float32, device=device).view(-1, 1)
    l_tensor = torch.tensor(l, dtype=torch.float32, device=device).view(-1, 1)
    u_tensor = torch.tensor(u, dtype=torch.float32, device=device).view(-1, 1)

    m_ineq = G_tensor.shape[0] if G_tensor.numel() > 0 else 0
        
    # Combine original constraints into K and q
    combined_matrix_list = []
    rhs = []
    if m_ineq > 0:
        combined_matrix_list.append(G_tensor)
        rhs.append(h_tensor)
    if A_tensor.numel() > 0:
        combined_matrix_list.append(A_tensor)
        rhs.append(b_tensor)
    
    K_tensor = torch.vstack(combined_matrix_list)
    q_tensor = torch.vstack(rhs)
    
    if support_sparse:
        # Check if sparse operations are faster
        K_tensor = sparse_vs_dense(K_tensor, device=device, kkt_passes=10)
        if verbose:
            print("Using Sparse operations") if K_tensor.is_sparse else print("Using Dense operations")
        
    return c_tensor, K_tensor, q_tensor, m_ineq, l_tensor, u_tensor
