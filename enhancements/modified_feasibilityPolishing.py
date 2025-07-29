import torch
import numpy as np
from time import perf_counter
from collections import defaultdict
import os
import pandas as pd

def mps_to_standard_form_torch(mps_file, device='cpu'):
    """
    Parses an MPS file and returns the standard form LP components as PyTorch tensors:
        minimize      cᵀx
        subject to    G x ≥ h
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
                bound_data[var_name]['lo'] = -float('inf')
                bound_data[var_name]['up'] = float('inf')
                
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
        lo = bound_data[var].get('lo', 0.0)
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
    projected = torch.zeros_like(grad)
    unconstrained = is_neg_inf & is_pos_inf
    projected[unconstrained] = 0.0
    neg_only = is_neg_inf & ~is_pos_inf
    projected[neg_only] = torch.clamp(grad[neg_only], max=0.0)
    pos_only = ~is_neg_inf & is_pos_inf
    projected[pos_only] = torch.clamp(grad[pos_only], min=0.0)
    fully_bounded = ~is_neg_inf & ~is_pos_inf
    projected[fully_bounded] = grad[fully_bounded]
    return projected

def spectral_norm_estimate_torch(A, num_iters=10):
    if A.numel() == 0:
        return 1.0
    b = torch.randn(A.shape[1], 1, device=A.device)
    b /= torch.norm(b)
    for _ in range(num_iters):
        b_new = A.T @ (A @ b)
        b = b_new / torch.norm(b_new)
    return torch.norm(A @ b)

def check_termination(x, y, c, G, h, A, b, l_dual, u_dual, is_neg_inf, is_pos_inf, K, q):
    """Calculates all residuals and gaps."""
    m_eq = A.shape[0] if A.numel() > 0 else 0
    m_ineq = G.shape[0] if G.numel() > 0 else 0
    
    # Primal and Dual Objectives
    prim_obj = c.T @ x
    dual_obj_base = q.T @ y

    # Gradient for dual residual calculation
    grad = c - K.T @ y
    
    # Dual objective adjustment for box constraints
    lam = project_lambda_box(grad, is_neg_inf, is_pos_inf)
    lam_pos = l_dual.T @ torch.clamp(lam, min=0.0)
    lam_neg = u_dual.T @ torch.clamp(lam, max=0.0)
    adjusted_dual_obj = dual_obj_base + lam_pos + lam_neg
    
    # Duality Gap
    duality_gap = torch.abs(adjusted_dual_obj - prim_obj)
    
    # Primal Residual
    res_eq = A @ x - b if m_eq > 0 else torch.tensor([], device=x.device).view(0, 1)
    res_ineq = torch.clamp(h - G @ x, min=0.0) if m_ineq > 0 else torch.tensor([], device=x.device).view(0, 1)
    primal_residual = torch.norm(torch.vstack([res_eq, res_ineq]), p=2)
    
    # Dual Residual
    dual_residual = torch.norm(grad - lam, p=2)

    return prim_obj, adjusted_dual_obj, duality_gap, primal_residual, dual_residual

def _pdhg_core_loop(c, K, q, l, u, m_ineq, x_start, y_start, tau, sigma, max_iter, term_tol, early_exit_check=None):
    """Core PDHG loop for main solve and polishing."""
    x = x_start.clone()
    y = y_start.clone()
    x_old = x.clone()
    theta = 1.0
    
    for k_inner in range(max_iter):
        x_old.copy_(x)
        grad = c - K.T @ y
        x = torch.clamp(x - tau * grad, min=l, max=u)
        x_bar = x + theta * (x - x_old)
        K_xbar = K @ x_bar
        y += sigma * (q - K_xbar)
        if m_ineq > 0:
            y[:m_ineq] = torch.clamp(y[:m_ineq], min=0.0)

        # Optional early exit for polishing sub-problems
        if early_exit_check:
            primal_res, dual_res = early_exit_check(x)
            if primal_res < term_tol or dual_res < term_tol:
                break
                
    return x, y

def pdhg_torch(c, G, h, A, b, l, u, is_neg_inf, is_pos_inf, l_dual, u_dual, device, 
               max_iter=100000, tol=1e-2, verbose=True, term_period=1000,
               polish=True, rel_gap_polish_tol=1e-1):
    """
    Solves an LP using PDHG with optional feasibility polishing.
    """
    n = c.shape[0]
    m_ineq = G.shape[0] if G.numel() > 0 else 0
    m_eq = A.shape[0] if A.numel() > 0 else 0

    # --- Problem Setup ---
    combined_matrix_list = []
    rhs_list = []
    if m_ineq > 0:
        combined_matrix_list.append(G)
        rhs_list.append(h)
    if m_eq > 0:
        combined_matrix_list.append(A)
        rhs_list.append(b)
    if not combined_matrix_list:
        raise ValueError("Both G and A matrices are empty.")

    K = torch.vstack(combined_matrix_list).to(device)
    q = torch.vstack(rhs_list).to(device)
    c_zero = torch.zeros_like(c)

    # --- Step Size Calculation ---
    eta = 0.9 / spectral_norm_estimate_torch(K, num_iters=100)
    c_norm = torch.linalg.norm(c, 2)
    q_norm = torch.linalg.norm(q, 2)
    omega = c_norm / q_norm if q_norm > 0 and c_norm > 0 else torch.tensor(1.0)
    tau = eta / omega
    sigma = eta * omega

    # --- Initialization ---
    x = torch.zeros((n, 1), device=device)
    y = torch.zeros((K.shape[0], 1), device=device)
    x_old = x.clone()
    x_avg = x.clone()
    y_avg = y.clone()
    theta = 1.0
    
    next_polish_k = 100

    # --- Main PDHG Loop ---
    for k in range(1, max_iter + 1):
        # Primal update
        x_old.copy_(x)
        grad = c - K.T @ y
        x = torch.clamp(x - tau * grad, min=l, max=u)

        # Extrapolation
        x_bar = x + theta * (x - x_old)

        # Dual update
        y += sigma * (q - (K @ x_bar))
        if m_ineq > 0:
            y[:m_ineq] = torch.clamp(y[:m_ineq], min=0.0)

        # Update average iterates
        x_avg = (x_avg * (k - 1) + x) / k
        y_avg = (y_avg * (k - 1) + y) / k
        
        # --- Regular Termination Check ---
        if k % term_period == 0:
            prim_obj, dual_obj, gap, prim_res, dual_res = check_termination(
                x_avg, y_avg, c, G, h, A, b, l_dual, u_dual, is_neg_inf, is_pos_inf, K, q
            )
            if verbose:
                print(f"[{k}] Primal Obj: {prim_obj.item():.4f}, Dual Obj: {dual_obj.item():.4f}, "
                      f"Gap: {gap.item():.2e}, Prim Res: {prim_res.item():.2e}, Dual Res: {dual_res.item():.2e}")
            
            cond1 = prim_res <= tol * (1 + q_norm)
            cond2 = dual_res <= tol * (1 + c_norm)
            cond3 = gap <= tol * (1 + torch.abs(prim_obj) + torch.abs(dual_obj))
            
            if cond1 and cond2 and cond3:
                if verbose: print(f"Converged at iteration {k}")
                return x_avg, prim_obj.item(), k

        # --- Feasibility Polishing ---
        if polish and k == next_polish_k:
            prim_obj, dual_obj, gap, _, _ = check_termination(
                x_avg, y_avg, c, G, h, A, b, l_dual, u_dual, is_neg_inf, is_pos_inf, K, q
            )
            relative_gap = gap / (1 + torch.abs(prim_obj) + torch.abs(dual_obj))
            
            if relative_gap < rel_gap_polish_tol:
                if verbose: print(f"--- [{k}] Polishing triggered (Rel Gap: {relative_gap.item():.2e}) ---")
                polish_iters = k // 8

                # --- 2. Primal Feasibility Polish---
                x_tilde, _ = _pdhg_core_loop(c_zero, K, q, l, u, m_ineq, x_avg, torch.zeros_like(y), 
                                            tau, sigma, polish_iters, tol)
                
                # --- 3. Dual Feasibility Polish---
                
                # FIX: Create homogenized bounds
                l_homog = l.clone()
                l_homog[~is_neg_inf] = 0.0
                u_homog = u.clone()
                u_homog[~is_pos_inf] = 0.0
                
                # In this simple formulation, we assume q (rhs) is also zeroed out for homogenization.
                q_zero = torch.zeros_like(q)

                # FIX: Call with original 'c' and homogenized bounds 'l_homog', 'u_homog'
                _, y_tilde = _pdhg_core_loop(c, K, q_zero, l_homog, u_homog, m_ineq, torch.zeros_like(x), y_avg,
                                            tau, sigma, polish_iters, tol)
                
                # --- 4. Check if polished solution converges ---
                p_obj, d_obj, p_gap, p_prim_res, p_dual_res = check_termination(
                    x_tilde, y_tilde, c, G, h, A, b, l_dual, u_dual, is_neg_inf, is_pos_inf, K, q
                )
                
                p_cond1 = p_prim_res <= tol * (1 + q_norm)
                p_cond2 = p_dual_res <= tol * (1 + c_norm)
                p_cond3 = p_gap <= tol * (1 + torch.abs(p_obj) + torch.abs(d_obj))
                
                if p_cond1 and p_cond2 and p_cond3:
                    if verbose: 
                        print(f"--- Converged after polishing at iteration {k} ---")
                        print(f"    Final Obj: {p_obj.item():.4f}, Gap: {p_gap.item():.2e}, Prim Res: {p_prim_res.item():.2e}, Dual Res: {p_dual_res.item():.2e}")
                    return x_tilde, p_obj.item(), k
            
            next_polish_k *= 2
            
    # Fallback return if max_iter is reached
    final_obj, _, _, _, _ = check_termination(x_avg, y_avg, c, G, h, A, b, l_dual, u_dual, is_neg_inf, is_pos_inf, K, q)
    return x_avg, final_obj.item(), max_iter

class Timer:
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

    # --- Get all MPS files from the folder ---
    mps_files = sorted([f for f in os.listdir(mps_folder_path) if f.endswith('.mps')])

    for mps_file in mps_files:
        mps_file_path = os.path.join(mps_folder_path, mps_file)
        print(f"\nProcessing {mps_file_path}...")

        try:
            c, G, h, A, b, l, u = mps_to_standard_form_torch(mps_file_path, device=device)
        except Exception as e:
            print(f"Failed to load MPS file: {mps_file_path}. Error: {e}")
            results.append({'File': mps_file, 'FP_Status': f'Failed to load: {e}'})
            continue

        is_neg_inf = torch.isinf(l) & (l < 0)
        is_pos_inf = torch.isinf(u) & (u > 0)
        l_dual = l.clone(); l_dual[is_neg_inf] = 0
        u_dual = u.clone(); u_dual[is_pos_inf] = 0

        # --- Solve ---
        try:
            with Timer("Solve time") as t:
                x, obj, k = pdhg_torch(c, G, h, A, b, l, u, is_neg_inf, is_pos_inf,
                                       l_dual, u_dual, device=device,
                                       max_iter=max_iter, tol=tol, verbose=True, polish=True)
            time_elapsed = t.elapsed
            
            status = "Solved"
            if k == max_iter:
                status = "Unsolved (max iterations reached)"

            results.append({
                'File': mps_file, 'FP_Objective': obj, 'FP_Iterations (k)': k,
                'FP_Time (s)': time_elapsed, 'FP_Status': status
            })
            print(f"Finished: {mps_file}, Time: {time_elapsed:.2f}s, Iter: {k}, Obj: {obj:.4f}, Status: {status}")

        except Exception as e:
            print(f"Solver failed for {mps_file}. Error: {e}")
            results.append({'File': mps_file, 'FP_Status': f'Solver failed: {e}'})

    # --- Save results to Excel ---
    df = pd.DataFrame(results)
    df.to_excel('pdhg_results_FP.xlsx', index=False)
    print("\nAll done. Results saved to pdhg_results_FP.xlsx.")