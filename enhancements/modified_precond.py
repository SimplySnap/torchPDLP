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

def ruiz_precondition_gpu(c, G, h, A, b, l, u, max_iter=20):
    """
    (GPU-ONLY) Performs Ruiz scaling on the problem data.
    Takes original problem tensors and returns scaled problem tensors.
    """
    device = c.device
    m_ineq = G.shape[0] if G.numel() > 0 else 0
    
    # Combine original constraints into K and q
    combined_matrix_list = []
    rhs = []
    if m_ineq > 0:
        combined_matrix_list.append(G)
        rhs.append(h)
    if A.numel() > 0:
        combined_matrix_list.append(A)
        rhs.append(b)
    
    K = torch.vstack(combined_matrix_list)
    q = torch.vstack(rhs)

    # --- Scaling Loop ---
    K_s, c_s, q_s, l_s, u_s = K.clone(), c.clone(), q.clone(), l.clone(), u.clone()
    m, n = K_s.shape
    eps = 1e-6

    D_row = torch.ones((m, 1), dtype=K.dtype, device=device)
    D_col = torch.ones((n, 1), dtype=K.dtype, device=device)

    for i in range(max_iter):
        row_norms = torch.sqrt(torch.linalg.norm(K_s, ord=torch.inf, dim=1, keepdim=True))
        row_norms[row_norms < eps] = 1.0
        D_row /= row_norms
        K_s /= row_norms

        col_norms = torch.sqrt(torch.linalg.norm(K_s, ord=torch.inf, dim=0, keepdim=True))
        col_norms[col_norms < eps] = 1.0
        D_col /= col_norms.T
        K_s /= col_norms

        if (torch.max(torch.abs(1 - row_norms)) < eps and
            torch.max(torch.abs(1 - row_norms)) < eps):
            break
    
    c_s *= D_col
    q_s *= D_row
    l_s /= D_col
    u_s /= D_col

    return K_s, c_s, q_s, l_s, u_s, D_col, m_ineq

def pdhg_scaled_solver_gpu(K_s, c_s, q_s, l_s, u_s, m_ineq,
                           is_neg_inf, is_pos_inf, 
                           max_iter=100000, tol=1e-2, verbose=True, term_period=1000):
    """
    (GPU-ONLY) Solves the scaled problem and returns the scaled solution.
    All termination checks are performed on the (scaled) data it receives.
    """
    device = K_s.device
    n = c_s.shape[0]

    # Calculate norms of SCALED data for termination criteria
    q_s_norm = torch.linalg.norm(q_s, 2)
    c_s_norm = torch.linalg.norm(c_s, 2)

    eta = 0.9 / spectral_norm_estimate_torch(K_s, num_iters=100)
    omega = c_s_norm / q_s_norm if (q_s_norm > 0 and c_s_norm > 0) else torch.tensor(1.0, device=device)

    tau, sigma, theta = eta / omega, eta * omega, 1.0
    
    x_s = torch.zeros((n, 1), device=device)
    x_s_old = x_s.clone()
    y = torch.zeros((K_s.shape[0], 1), device=device)

    l_dual = l_s.clone()
    u_dual = u_s.clone()
    l_dual[is_neg_inf] = 0
    u_dual[is_pos_inf] = 0

    
    for k in range(max_iter):
        x_s_old.copy_(x_s)
    
        grad_s = c_s - (K_s.T @ y)
        x_s = torch.clamp(x_s - tau * grad_s, min=l_s, max=u_s)

        x_s_bar = x_s + theta * (x_s - x_s_old)

        y += sigma * (q_s - (K_s @ x_s_bar))

        if m_ineq > 0:
            y[:m_ineq] = torch.clamp(y[:m_ineq], min=0.0)

         # --- Check Termination Every term_period Iterations on GPU---
        if k % term_period == 0:
            # --- Calculate all residuals and gaps as GPU tensors ---
            prim_obj = c_s.T @ x_s
            dual_obj = q_s.T @ y

            lam = project_lambda_box(grad_s, is_neg_inf, is_pos_inf)
            lam_pos = l_dual.T @ torch.clamp(lam, min=0.0)
            lam_neg = u_dual.T @ torch.clamp(lam, max=0.0)

            adjusted_dual = dual_obj + lam_pos + lam_neg
            duality_gap = torch.abs(adjusted_dual - prim_obj)

            full_residual_s = K_s @ x_s - q_s
            residual_ineq = torch.clamp(full_residual_s[:m_ineq], max=0.0)
            residual_eq = full_residual_s[m_ineq:]

            primal_residual = torch.linalg.norm(torch.vstack([residual_ineq, residual_eq]))
            
            dual_residual = torch.norm(grad_s - lam, p=2)

            # --- Check convergence condition on GPU ---
            cond1 = primal_residual <= tol * (1 + q_s_norm)
            cond2 = dual_residual <= tol * (1 + c_s_norm)
            cond3 = duality_gap <= tol * (1 + torch.abs(prim_obj) + torch.abs(adjusted_dual))
            
            converged = cond1 & cond2 & cond3

            # --- Optional Verbose Logging (this part transfers data for printing) ---
            if verbose:
                print(f"[{k}] Primal Obj: {prim_obj.item():.4f}, Adjusted Dual Obj: {adjusted_dual.item():.4f}, "
                      f"Gap: {duality_gap.item():.2e}, Prim Res: {primal_residual.item():.2e}, Dual Res: {dual_residual.item():.2e}")

            # --- Check for termination (1 boolean transfer from GPU to CPU) ---
            if converged.item():
                if verbose:
                    print(f"Converged at iteration {k}")
                break
            
    return x_s, k

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
    mps_folder_path = 'feasibleT'
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

        # --- Solve ---
        try:
            with Timer("Solve time") as t:
                # PRECONDITION: Perform scaling entirely on GPU
                K_s, c_s, q_s, l_s, u_s, D_col, m_ineq = ruiz_precondition_gpu(c, G, h, A, b, l, u)

                # SOLVE: Run the solver on the scaled problem
                x_s, k = pdhg_scaled_solver_gpu(K_s, c_s, q_s, l_s, u_s, m_ineq,
                                                is_neg_inf, is_pos_inf,
                                                max_iter=max_iter, tol=tol, verbose=False)

                # UN-SCALE & TRANSFER
                x_final = x_s * D_col
                obj_final = (c.T @ x_final).item()

            time_elapsed = t.elapsed

            status = "Solved"
            if k == max_iter - 1:
                status = "Unsolved (max iterations reached)"

            results.append({
                'File': mps_file,
                'Objective': obj_final,
                'Iterations (k)': k,
                'Time (s)': time_elapsed,
                'Status': status
            })

            print(f"Finished: {mps_file}, Time: {time_elapsed:.2f}s, Iter: {k}, Obj: {obj_final:.4f}, Status: {status}")

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