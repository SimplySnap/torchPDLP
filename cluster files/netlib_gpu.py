import torch
import numpy as np
from time import perf_counter
from collections import defaultdict
import os
import pandas as pd

def mps_to_standard_form_torch(mps_file, device='cpu'):
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
    var_names = set()
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
            var_names.add(var_name)
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
                # bound_data[var_name]['lo'] = -float('inf')
                bound_data[var_name]['lo'] = 0.0
                bound_data[var_name]['up'] = float('inf')
                
    # Check correct information loaded
    # print(row_types, col_data,rhs_data,range_data,bound_data)
    
    # Final variable ordering and index mapping
    var_names = sorted(var_names)
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
            # Ranged constraint → convert to two inequalities
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
        # lo = bound_data[var].get('lo', -float('inf'))
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

    return c_tensor, G_tensor, h_tensor, A_tensor, b_tensor, l_tensor, u_tensor

def project_lambda_box(grad, is_neg_inf, is_pos_inf):
    """
    Projects the gradient onto the normal cone of the feasible region defined by bounds l and u.

    For each i:
      - If l[i] == -inf and u[i] == +inf: projection is 0
      - If l[i] == -inf and u[i] is real: clamp to ≤ 0 (R⁻)
      - If l[i] is real and u[i] == +inf: clamp to ≥ 0 (R⁺)
      - If both are finite: no projection (keep full value)

    Args:
        grad: (n, 1) gradient vector (torch tensor)
        l: (n, 1) lower bounds (torch tensor)
        u: (n, 1) upper bounds (torch tensor)

    Returns:
        projected: (n, 1) projected gradient (interpreted as λ)
    """
    projected = torch.zeros_like(grad)

    # Case 1: (-inf, +inf) → {0}
    unconstrained = is_neg_inf & is_pos_inf
    projected[unconstrained] = 0.0

    # Case 2: (-inf, real) → R⁻ → clamp at 0 from above
    neg_only = is_neg_inf & ~is_pos_inf
    projected[neg_only] = torch.clamp(grad[neg_only], max=0.0)

    # Case 3: (real, +inf) → R⁺ → clamp at 0 from below
    pos_only = ~is_neg_inf & is_pos_inf
    projected[pos_only] = torch.clamp(grad[pos_only], min=0.0)

    # Case 4: (real, real) → full space → keep gradient
    fully_bounded = ~is_neg_inf & ~is_pos_inf
    projected[fully_bounded] = grad[fully_bounded]

    return projected


def spectral_norm_estimate_torch(A, num_iters=10):
  """
  Estimates the spectral norm of a matrix A with enough acuracy to use in
  setting the step size of the PDHG algorithm.
  """

  b = torch.randn(A.shape[1], 1, device=A.device)
  for _ in range(num_iters):
      b = A.T @ (A @ b)
      b /= torch.norm(b)
  return torch.norm(A @ b)

def pdhg_torch(c, G, h, A, b, l, u, is_neg_inf, is_pos_inf, l_dual, u_dual, device, max_iter=100000, tol=1e-2, verbose=True, term_period=1000):
    """
    Solves:
        min cᵀx s.t. Gx ≥ h, Ax = b, l ≤ x ≤ u
    using the Primal-Dual Hybrid Gradient (PDHG) algorithm.

    Args:
      c, G, h, A, b, l, u: torch tensors representing the problem data
      is_pos_inf: torch tensor indicating which elements of u are +inf
      is_neg_inf: torch tensor indicating which elements of l are -inf
      l_dual: torch tensor representing the lower bounds of the dual variables
      u_dual: torch tensor representing the upper bounds of the dual variables
      device: torch device (cpu or cuda)
      max_iter: maximum number of iterations
      tol: tolerance for convergence
      verbose: whether to print termination information
      term_period: period for termination checks

    Returns:
      minimizer, objective value, and number of iterations for convergence
    """    
    n = c.shape[0]
    m_ineq = G.shape[0] if G.numel() > 0 else 0
    m_eq = A.shape[0] if A.numel() > 0 else 0

    # Combine constraints
    combined_matrix_list = []
    rhs = []
    if m_ineq > 0:
        combined_matrix_list.append(G)
        rhs.append(h)
    if m_eq > 0:
        combined_matrix_list.append(A)
        rhs.append(b)

    if not combined_matrix_list:
        raise ValueError("Both G and A matrices are empty.")
  
    K = torch.vstack(combined_matrix_list).to(device)           # Combined constraint matrix
    q = torch.vstack(rhs).to(device)                            # Combined right-hand side
    c = c.to(device)
  
    q_norm = torch.linalg.norm(q, 2)
    c_norm = torch.linalg.norm(c, 2)
  
    eta = 0.9 / spectral_norm_estimate_torch(K, num_iters=100)

    if q_norm > 0 and c_norm > 0:
        omega = c_norm / q_norm
    else:
        omega = torch.tensor(1.0)

    tau = eta / omega
    sigma = eta * omega

    theta = 1.0
  
    # Initialize primal and dual
    x = torch.zeros((n, 1), device=device)
    x_old = x.clone()
    y = torch.zeros((K.shape[0], 1), device=device)
  
    for k in range(max_iter):
        x_old.copy_(x)
    
        # Compute gradient and primal update
        Kt_y = K.T @ y
        grad = c - Kt_y
        x = torch.clamp(x - tau * grad, min=l, max=u)

        # Extrapolate
        x_bar = x + theta * (x - x_old)

        # Dual update
        K_xbar = K @ x_bar
        y += sigma * (q - K_xbar)

        # Project duals:
        if m_ineq > 0:
            y[:m_ineq] = torch.clamp(y[:m_ineq], min=0.0)

        # --- Check Termination Every term_period Iterations ---
        if k % term_period == 0:
            # Primal and dual objective
            prim_obj = (c.T @ x)[0][0]
            dual_obj = (q.T @ y)[0][0]

            # Lagrange multipliers from box projection
            lam = project_lambda_box(grad, is_neg_inf, is_pos_inf)
            lam_pos = (l_dual.T @ torch.clamp(lam, min=0.0))[0][0]
            lam_neg = (u_dual.T @ torch.clamp(lam, max=0.0))[0][0]

            adjusted_dual = dual_obj + lam_pos + lam_neg
            duality_gap = abs(adjusted_dual - prim_obj)

            # Primal residual (feasibility)
            residual_eq = A @ x - b if m_eq > 0 else torch.zeros(1, device=device)
            residual_ineq = torch.clamp(h - G @ x, min=0.0) if m_ineq > 0 else torch.zeros(1, device=device)
            primal_residual = torch.norm(torch.vstack([residual_eq, residual_ineq]), p=2).item()

            # Dual residual (change in x)
            dual_residual = torch.norm(grad - lam, p=2).item()

            if verbose:
                print(f"[{k}] Primal Obj: {prim_obj:.4f}, Adjusted Dual Obj: {adjusted_dual:.4f}, "
                      f"Gap: {duality_gap:.2e}, Prim Res: {primal_residual:.2e}, Dual Res: {dual_residual:.2e}")

            # Termination condition
            if (primal_residual <= tol * (1 + q_norm) and
                dual_residual <= tol * (1 + c_norm) and
                duality_gap <= tol * (1 + abs(prim_obj) + abs(adjusted_dual))):
                if verbose:
                    print(f"Converged at iteration {k}")
                break
            
    return x, prim_obj.cpu().numpy(), k

class Timer:
  """
  Timer class to measure execution time of code blocks.
  Usage:
  
    with Timer("Label"):
        # Code block to be timed

  Output:
    Label: <time in seconds> seconds
  """
    # ChatGPT wrote this and I don't know how it works
  def __init__(self, label="Elapsed time"):
      self.label = label

  def __enter__(self):
      self.start = perf_counter()
      return self

  def __exit__(self, *args):
      self.end = perf_counter()
      self.elapsed = self.end - self.start
      print(f"{self.label}: {self.elapsed:.6f} seconds")

if __name__ == '__main__':
    # --- Device Selection ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"PyTorch is using ROCm/CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("ROCm/CUDA not available. PyTorch is using CPU.")

    # --- Configuration ---
    mps_folder_path = 'feasible'
    max_iter = 100000
    tol = 1e-2
    results = []

    num_threads = torch.get_num_threads()
    print(f"PyTorch is running on {num_threads} threads.")

    # --- Configuration ---
    mps_folder_path = 'feasible'
    max_iter = 100000
    tol = 1e-2
    results = []

    # --- Get all MPS files from the folder ---
    mps_files = sorted([f for f in os.listdir(mps_folder_path) if f.endswith('.mps')])

    for mps_file in mps_files:
        mps_file_path = os.path.join(mps_folder_path, mps_file)
        print(f"\nProcessing {mps_file_path}...")

        try:
            # --- Load problem ---
            c, G, h, A, b, l, u = mps_to_standard_form_torch(mps_file_path, device=device)
        except Exception as e:
            print(f"Failed to load MPS file: {mps_file_path}. Error: {e}")
            results.append({
                'File': mps_file,
                'Objective': 'N/A',
                'Iterations (k)': 'N/A',
                'Time (s)': 'N/A',
                'Status': f'Failed to load: {e}'
            })
            continue

        is_neg_inf = torch.isinf(l) & (l < 0)
        is_pos_inf = torch.isinf(u) & (u > 0)

        l_dual = l.clone()
        u_dual = u.clone()
        l_dual[is_neg_inf] = 0
        u_dual[is_pos_inf] = 0

        # --- Solve ---
        try:
            with Timer("Solve time") as t:
                x, obj, k = pdhg_torch(c, G, h, A, b, l, u, is_neg_inf, is_pos_inf,
                                       l_dual, u_dual, device=device,
                                       max_iter=max_iter, tol=tol, verbose=False)
            time_elapsed = t.elapsed

            status = "Solved"
            if k == max_iter - 1:
                status = "Unsolved (max iterations reached)"

            results.append({
                'File': mps_file,
                'Objective': obj,
                'Iterations (k)': k,
                'Time (s)': time_elapsed,
                'Status': status
            })

            print(f"Finished: {mps_file}, Time: {time_elapsed:.2f}s, Iter: {k}, Obj: {obj:.4f}, Status: {status}")

        except Exception as e:
            print(f"Solver failed for {mps_file}. Error: {e}")
            results.append({
                'File': mps_file,
                'Objective': 'N/A',
                'Iterations (k)': 'N/A',
                'Time (s)': 'N/A',
                'Status': f'Solver failed: {e}'
            })

    # --- Save results to Excel ---
    df = pd.DataFrame(results)
    df.to_excel('pdhg_results_gpu.xlsx', index=False)
    print("\nAll done. Results saved to pdhg_results_gpu.xlsx.")