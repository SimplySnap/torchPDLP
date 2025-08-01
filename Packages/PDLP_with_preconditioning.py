def ruiz_precondition(c, K, q, l, u, device='cpu', max_iter=20, eps=1e-6):
    """
    Performs Ruiz equilibration (scaling) on the standard-form linear program using GPU tensors.

    This is done to improve the numerical stability of iterative solvers, especially for
    ill-conditioned problems.

    Standard form of the LP:
        minimize     cᵀx
        subject to   Gx ≥ h
                     Ax = b
                     l ≤ x ≤ u

    Inputs:
    -------
    c  : (n x 1) torch tensor — objective function vector
    K  : ((m_ineq + m_eq) x n) torch tensor — constraint matrix (stacked G and A)
    q  : ((m_ineq + m_eq) x 1) torch tensor — RHS vector (stacked h and b)
    l  : (n x 1) torch tensor — lower bounds on variables
    u  : (n x 1) torch tensor — upper bounds on variables
    max_iter : int — number of scaling iterations to perform (default: 20)

    Outputs:
    --------
    K_s : ((m_ineq + m_eq) x n) torch tensor — scaled constraint matrix (stacked G and A)
    c_s : (n x 1) torch tensor — scaled objective vector
    q_s : ((m_ineq + m_eq) x 1) torch tensor — scaled RHS vector (stacked h and b)
    l_s : (n x 1) torch tensor — scaled lower bounds
    u_s : (n x 1) torch tensor — scaled upper bounds
    D_col : (n x 1) torch tensor — final column scaling factors (for rescaling solution)
    m_ineq : int — number of inequality constraints (used for slicing G vs A in K_s if needed)

    Notes:
    ------
    - This works whether K is sparse or dense and outputs K_s in the same format
    - The scaling preserves feasibility and optimality but improves numerical conditioning.
    - You must rescale your solution after solving using D_col (and D_row if needed).
    """
    import torch
    
    is_sparse = K.is_sparse

    # Clone inputs
    K_s = K.clone()
    c_s, q_s, l_s, u_s = c.clone(), q.clone(), l.clone(), u.clone()
    m, n = K_s.shape

    D_row = torch.ones((m, 1), dtype=K.dtype, device=device)
    D_col = torch.ones((n, 1), dtype=K.dtype, device=device)

    for _ in range(max_iter):
        # --- Row norm ---
        if is_sparse:
            row_indices = K_s.indices()[0]
            abs_vals = K_s.values().abs()
            row_max = torch.zeros(m, device=device)
            row_max.scatter_reduce_(0, row_indices, abs_vals, reduce="amax", include_self=True)
            row_norms = torch.sqrt(torch.clamp(row_max, min=eps)).view(-1, 1)

            # Update sparse values
            new_vals = K_s.values() / row_norms[row_indices, 0]
            K_s = torch.sparse_coo_tensor(K_s.indices(), new_vals, K_s.shape, device=K_s.device)

        else:
            row_norms = torch.sqrt(torch.linalg.norm(K_s, ord=float('inf'), dim=1, keepdim=True))
            row_norms = torch.clamp(row_norms, min=eps)
            K_s = K_s / row_norms

        D_row = D_row / row_norms

        # --- Column norm ---
        if is_sparse:
            col_indices = K_s.indices()[1]
            abs_vals = K_s.values().abs()
            col_max = torch.zeros(n, device=device)
            col_max.scatter_reduce_(0, col_indices, abs_vals, reduce="amax", include_self=True)
            col_norms = torch.sqrt(torch.clamp(col_max, min=eps)).view(1, -1)

            # Update sparse values
            new_vals = K_s.values() / col_norms[0, col_indices]
            K_s = torch.sparse_coo_tensor(K_s.indices(), new_vals, K_s.shape, device=K_s.device)

        else:
            col_norms = torch.sqrt(torch.linalg.norm(K_s, ord=float('inf'), dim=0, keepdim=True))
            col_norms = torch.clamp(col_norms, min=eps)
            K_s = K_s / col_norms

        D_col = D_col / col_norms.T

        # --- Convergence check ---
        if torch.max(torch.abs(1 - row_norms)) < eps and torch.max(torch.abs(1 - col_norms)) < eps:
            break

    # --- Scale other vectors ---
    c_s = c_s * D_col
    q_s = q_s * D_row
    l_s = l_s / D_col
    u_s = u_s / D_col

    return c_s, K_s, q_s, l_s, u_s, D_col, D_row


def sparse_vs_dense(A, device='cpu', kkt_passes=10):
    """
    Benchmarks matrix-vector multiplication using dense and sparse formats.
    
    Parameters:
        A (torch.Tensor): 2D matrix (dense tensor)
        device (str): 'cpu' or 'cuda'
        num_trials (int): Number of repetitions for timing

    Returns:
        dict: {
            'preferred': 'sparse' or 'dense',
            'dense_time': float (seconds),
            'sparse_time': float (seconds)
        }
    """
    import torch
    import time
  
    assert A.dim() == 2, "Input must be a 2D matrix"
    m, n = A.shape
    A = A.to(device)

    # Dense timing
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    A_transpose = A.t()
    for _ in range(kkt_passes // 2):
        vec = torch.randn(n, 1, device=device)
        _ = A @ vec
        vec = torch.randn(m, 1, device=device)
        _ = A_transpose @ vec
    torch.cuda.synchronize() if device == 'cuda' else None
    dense_time = time.time() - start

    # Convert to sparse
    A_sparse = A.to_sparse()
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    A_sparse_transpose = A_sparse.t()
    for _ in range(kkt_passes // 2):
        vec = torch.randn(n, 1, device=device)
        _ = torch.sparse.mm(A_sparse, vec)
        vec = torch.randn(m, 1, device=device)
        _ = torch.sparse.mm(A_sparse_transpose, vec)
    torch.cuda.synchronize() if device == 'cuda' else None
    sparse_time = time.time() - start

    return A_sparse if sparse_time < dense_time else A

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
    import torch
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


def spectral_norm_estimate_torch(A, num_iters=20):
  """
  Estimates the spectral norm of a matrix A with enough acuracy to use in
  setting the step size of the PDHG algorithm.
  """
  import torch

  b = torch.randn(A.shape[1], 1, device=A.device)
  for _ in range(num_iters):
      b = A.T @ (A @ b)
      b /= torch.norm(b)
  return torch.norm(A @ b)

def compute_residuals_and_duality_gap(x, y, c, q, K, m_ineq, is_neg_inf, is_pos_inf, l_dual, u_dual):
  """
  Computes the primal and dual residuals, duality gap, and KKT error.
  
  Args:
      x (torch.Tensor): Primal variable.
      y (torch.Tensor): Dual variable.
      c (torch.Tensor): Coefficients for the primal objective.
      q (torch.Tensor): Right-hand side vector for the constraints.
      K (torch.Tensor): Constraint matrix.
      m_ineq (int): Number of inequality constraints.
      omega (float): Scaling factor for the dual update.
      is_neg_inf (torch.Tensor): Boolean mask for negative infinity lower bounds.
      is_pos_inf (torch.Tensor): Boolean mask for positive infinity upper bounds.
      l_dual (torch.Tensor): Lower bounds for the dual variables.
      u_dual (torch.Tensor): Upper bounds for the dual variables.
  Returns:
      primal_residual (torch.Tensor): Norm of the primal residual.
      dual_residual (torch.Tensor): Norm of the dual residual.
      duality_gap (torch.Tensor): Duality gap.
  """
  import torch
  
  # Primal and dual objective
  grad = c - K.T @ y
  prim_obj = (c.T @ x).flatten()
  dual_obj = (q.T @ y).flatten()

  # Lagrange multipliers from box projection
  lam = project_lambda_box(grad, is_neg_inf, is_pos_inf)
  lam_pos = (l_dual.T @ torch.clamp(lam, min=0.0)).flatten()
  lam_neg = (u_dual.T @ torch.clamp(lam, max=0.0)).flatten()

  adjusted_dual = dual_obj + lam_pos + lam_neg
  duality_gap = adjusted_dual - prim_obj
    
  # Primal residual (feasibility)
  full_residual = K @ x - q
  residual_ineq = torch.clamp(full_residual[:m_ineq], max=0.0)
  residual_eq = full_residual[m_ineq:]
  primal_residual = torch.norm(torch.vstack([residual_eq, residual_ineq]), p=2).flatten()

  # Dual residual (change in x)
  dual_residual = torch.norm(grad - lam, p=2).flatten()
  
  return primal_residual, dual_residual, duality_gap, prim_obj, adjusted_dual
    
def KKT_error(x, y, c, q, K, m_ineq, omega, is_neg_inf, is_pos_inf, l_dual, u_dual, device):
      """
      Computes the KKT error using global variables.
      """
      import torch
  
      omega_sqrd = omega ** 2
      # Compute primal and dual residuals, and duality gap
      primal_residual, dual_residual, duality_gap, _ , _ = compute_residuals_and_duality_gap(x, y, c, q, K, m_ineq, is_neg_inf, is_pos_inf, l_dual, u_dual)
      # Compute the error
      KKT = torch.sqrt(omega_sqrd * primal_residual ** 2 + (dual_residual ** 2) / omega_sqrd + duality_gap ** 2)

      return KKT
  
def check_termination(primal_residual, dual_residual, duality_gap, prim_obj, adjusted_dual, q_norm, c_norm, tol):
    """
    Checks the termination conditions for the PDHG algorithm.
    Args:
        primal_residual (torch.Tensor): Norm of the primal residual.
        dual_residual (torch.Tensor): Norm of the dual residual.
        duality_gap (torch.Tensor): Duality gap.
        prim_obj (torch.Tensor): Primal objective value.
        adjusted_dual (torch.Tensor): Adjusted dual objective value.
        q_norm (float): Norm of the right-hand side vector q.
        c_norm (float): Norm of the coefficients vector c.
        tol (float): Tolerance for stopping criterion.
    Returns:
        bool: True if termination conditions are met, False otherwise.
    """
    cond1 = primal_residual <= tol * (1 + q_norm) 
    cond2 = dual_residual <= tol * (1 + c_norm)
    cond3 = duality_gap <= tol * (1 + abs(prim_obj) + abs(adjusted_dual))
    return cond1 and cond2 and cond3

def primal_weight_update(x_prev, x, y_prev, y, omega, smooth_theta):
    import torch
  
    diff_y_norm = torch.linalg.norm(y_prev - y, 2)
    diff_x_norm = torch.linalg.norm(x_prev - x, 2)
    if diff_x_norm > 0 and diff_y_norm > 0:
        omega = torch.exp(smooth_theta * (torch.log(diff_y_norm/diff_x_norm)) + (1-smooth_theta)*torch.log(omega))
    return omega

def one_step_pdhg(x, y, c, q, K, l, u, m_ineq, eta, omega, theta, j):
  import torch

  x_old = x.clone()

  # Compute gradient and primal update
  Kt_y = K.T @ y
  grad = c - Kt_y
  x = torch.clamp(x - eta / omega * grad, min=l, max=u) #Project

  # Extrapolate
  x_bar = x + theta * (x - x_old)

  # Dual update
  K_xbar = K @ x_bar
  y += eta * omega * (q - K_xbar)

  # Project dual:
  if m_ineq > 0:
      y[:m_ineq] = torch.clamp(y[:m_ineq], min=0.0)

  return x, y, eta, eta, j + 1 # Add one kkt pass

def adaptive_one_step_pdhg(x, y, c, q, K, l, u, m_ineq, eta, omega, theta, k, j):
    """
    Perform one step of the Primal-Dual Hybrid Gradient (PDHG) algorithm with adaptive stepsize.
    Args:
        x (torch.Tensor): Current primal variable.
        y (torch.Tensor): Current dual variable.
        c (torch.Tensor): Coefficients for the primal objective.
        q (torch.Tensor): Right-hand side vector for the constraints.
        K (torch.Tensor): Constraint matrix.
        l (torch.Tensor): Lower bounds for the primal variable.
        u (torch.Tensor): Upper bounds for the primal variable.
        m_ineq (int): Number of inequality constraints.
        eta (float): Step size for the primal update.
        omega (float): Scaling factor for the dual update.
        theta (float): Extrapolation parameter.
        k (int): Current iteration number.
        j (int): Current KKT pass number.
    Returns:
        x (torch.Tensor): Updated primal variable.
        y (torch.Tensor): Updated dual variable.
    """ 
    import torch
    x_old = x.clone()
    y_old = y.clone()
    
    # Primal update
    Kt_y = K.T @ y_old
    grad = c - Kt_y

    for i in range(200):
        
        # --- CURRENT STEPSIZE ---
        tau = eta / omega
        sigma = eta * omega

        x = torch.clamp(x_old - tau * grad, min=l, max=u)

        # Extrapolate
        diff_x = x - x_old
        x_bar = x + theta * diff_x

        # Dual update
        K_xbar = K @ x_bar
        y = y_old + sigma * (q - K_xbar)

        # Project duals:
        if m_ineq > 0:
            y[:m_ineq] = torch.clamp(y[:m_ineq], min=0.0)
            
        diff_y = y - y_old
        
        j += 1

        # Calculate the denominator for the eta_bar update
        denominator = 2 * (diff_y.T @ K @ diff_x)

        # --- CALCULATE NEW STEP SIZES ---
        if denominator != 0:
            numerator = omega * (torch.linalg.norm(diff_x)**2) + (torch.linalg.norm(diff_y)**2) / omega
            eta_bar = numerator / abs(denominator)
            eta_prime_term1 = (1 - (k + 1)**(-0.3)) * eta_bar
        else:
            eta_bar = torch.tensor(float('inf'))
            eta_prime_term1 = torch.tensor(float('inf'))
            
        eta_prime_term2 = (1 + (k + 1)**(-0.6)) * eta
        eta_prime = torch.min(eta_prime_term1, eta_prime_term2)

        if eta <= eta_bar:
            return x, y, eta.squeeze(), eta_prime.squeeze(), j

        eta = eta_prime
        
        return x, y, eta.squeeze(), eta.squeeze(), j

def mps_to_standard_form(mps_file, device='cpu', verbose=True):
    """
    Parses an MPS file and returns the standard form LP components as PyTorch tensors:
        minimize     cᵀx
        subject to   G x ≥ h
                     A x = b
                     l ≤ x ≤ u

    Returns: c, G, h, A, b, l, u
    """
    import numpy as np
    from collections import defaultdict
    import torch

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

    # Check if using sparse matrix is faster
    K_tensor = sparse_vs_dense(K_tensor, device=device, kkt_passes = 10)
    if verbose:
        print("Using sparse operations") if K_tensor.is_sparse else print("Using dense operations")
    
    return c_tensor, K_tensor, q_tensor, m_ineq, l_tensor, u_tensor

def pdlp_algorithm(K, m_ineq, c, q, l, u, device, max_iter=100_000, tol=1e-4, verbose=True, restart_period=40, precondition=False, primal_update=False, adaptive_step=False):
    
    import torch
    #from primal_dual_hybrid_gradient_step import adaptive_one_step_pdhg, fixed_one_step_pdhg
    #from helpers import spectral_norm_estimate_torch, KKT_error, compute_residuals_and_duality_gap, check_termination
    #from enhancements import primal_weight_update

    is_neg_inf = torch.isinf(l) & (l < 0)
    is_pos_inf = torch.isinf(u) & (u > 0)

    l_dual = l.clone()
    u_dual = u.clone()
    l_dual[is_neg_inf] = 0
    u_dual[is_pos_inf] = 0

    # -------------------- Preconditioning --------------------  
    
    # Save unconditioned tensors for terminiation checks
    c_og, K_og, q_og, l_og, u_og = c.clone(), K.clone(), q.clone(), l.clone(), u.clone
    if precondition: 
        # Precondition tensors and output D_col and D_row to uncondition the primal and dual for termination checks
        c, K, q, l, u, D_col, D_row = ruiz_precondition(c, K, q, l, u, device=device, max_iter=20, eps=1e-6)
    else:
        D_row = torch.ones((K.shape[0], 1), dtype=K.dtype, device=device)
        D_col = torch.ones((K.shape[1], 1), dtype=K.dtype, device=device)

    # ------------------- End Preconditioning ------------------
        
    q_norm = torch.linalg.norm(q, 2)
    c_norm = torch.linalg.norm(c, 2)
    q_norm_og = torch.linalg.norm(q_og, 2)
    c_norm_og = torch.linalg.norm(c_og, 2)

    eta = 0.9 / spectral_norm_estimate_torch(K, num_iters=100)
    omega = c_norm / q_norm if q_norm > 1e-6 and c_norm > 1e-6 else torch.tensor(1.0, device=device)

    theta = 1.0

    # Restart Parameters [Sufficient, Necessary, Artificial]
    beta = [0.2, 0.8, 0.36]

    # Initialize primal and dual
    x = torch.zeros((c.shape[0], 1), device=device)
    y = torch.zeros((K.shape[0], 1), device=device)

    # Counters
    n = 0 # Outer Loop Counter
    k = 0 # Total Iteration Counter
    j = 0 # Kkt pass Counter

    # Initialize Previous KKT Error
    KKT_first = 0 # The actual KKT error of the very first point doesn't matter since the artificial criteria will always hit anyway
    
    # -------------- Outer Loop --------------
    while k < max_iter:
        t = 0 # Initialize inner iteration counter
        
        # Initialize/Reset sums for averaging
        x_eta_total = torch.zeros_like(x)
        y_eta_total = torch.zeros_like(y)
        eta_total = 0
        
        # Initialize/Reset Previous restart point for primal weighting
        x_last_restart = x.clone()
        y_last_restart = y.clone()
        
        # --------- Inner Loop ---------
        while k < max_iter:
            
            k += 1
            x_previous = x.clone() # For checking necessary criteria
            y_previous = y.clone()

            # Adaptive step of pdhg
            x, y, eta, eta_hat, j = adaptive_one_step_pdhg(x, y, c, q, K, l, u, m_ineq, eta, omega, theta, k, j) if adaptive_step else one_step_pdhg(x, y, c, q, K, l, u, m_ineq, eta, omega, theta, j)
            
            # Increase iteration counters
            t += 1

            # Update totals
            x_eta_total += eta * x
            y_eta_total += eta * y
            eta_total += eta
        
            # Update eta
            eta = eta_hat

            # Check Restart Criteria Every restart_period iterations
            if t % restart_period == 0:
                
                # Compute averages
                x_avg = x_eta_total / eta_total
                y_avg = y_eta_total / eta_total

                # Compute KKT errors
                KKT_current = KKT_error(x, y, c, q, K, m_ineq, omega, is_neg_inf, is_pos_inf, l_dual, u_dual, device)
                KKT_average = KKT_error(x_avg, y_avg, c, q, K, m_ineq, omega, is_neg_inf, is_pos_inf, l_dual, u_dual, device)
                KKT_min = min(KKT_current, KKT_average)
                KKT_previous = KKT_error(x_previous, y_previous, c, q, K, m_ineq, omega, is_neg_inf, is_pos_inf, l_dual, u_dual, device) # For checking necessary criteria

                # Add three kkt passes
                j += 3
            
                # Check Restart Criteria and update with Restart Candidate
                if KKT_min <= beta[0] * KKT_first: # Sufficient Criteria
                    print(f"Sufficient restart at iteration {t} using the", "Average iterate." if KKT_current >= KKT_average else "Current iterate.") if verbose else None
                    (x, y) = (x_avg, y_avg) if KKT_current >= KKT_average else (x, y)
                    break
                elif KKT_min <= beta[1] * KKT_first and KKT_min > KKT_previous: # Necessary Criteria
                    print(f"Necessary restart at iteration {t} using the", "Average iterate." if KKT_current >= KKT_average else "Current iterate.") if verbose else None
                    (x, y) = (x_avg, y_avg) if KKT_current >= KKT_average else (x, y)
                    break
                elif t >= beta[2] * k: # Artificial Criteria
                    print(f"Artificial restart at iteration {t} using the", "Average iterate." if KKT_current >= KKT_average else "Current iterate.") if verbose else None
                    (x, y) = (x_avg, y_avg) if KKT_current >= KKT_average else (x, y)
                    break
        # ------------- End Inner Loop ------------
        n += 1 # Increase restart loop counter

        if primal_update: # Primal weight update
            omega = primal_weight_update(x_last_restart, x, y_last_restart, y, omega, 0.5)

        KKT_first = KKT_error(x, y, c, q, K, m_ineq, omega, is_neg_inf, is_pos_inf, l_dual, u_dual, device) # Update KKT_first for next iteration (incase primal weight updates)
        j += 1 # Add one kkt pass
      
        # Compute primal and dual residuals, and duality gap with unconditioned tensors
        primal_residual, dual_residual, duality_gap, prim_obj, adjusted_dual = compute_residuals_and_duality_gap(x * D_col, y * D_row, c_og, q_og, K_og, m_ineq, is_neg_inf, is_pos_inf, l_dual, u_dual)

        # Add one kkt pass
        j += 1

        if verbose:
            print(f"[{k}] Primal Obj: {prim_obj.item():.4f}, Adjusted Dual Obj: {adjusted_dual.item():.4f}, "
                    f"Gap: {duality_gap.item()/ (1 + abs(prim_obj.item()) + abs(adjusted_dual.item())):.2e}, Prim Res: {primal_residual.item() / (1 + q_norm_og):.2e}, Dual Res: {dual_residual.item() / (1 + c_norm_og):.2e}")
            print("")

        # Termination conditions
        if check_termination(primal_residual, dual_residual, duality_gap, prim_obj, adjusted_dual, q_norm_og, c_norm_og, tol):
            if verbose:
              print(f"Converged at iteration {k} restart loop {n}")
            break
    # ------------------- End Outer Loop ------------------------
    
    return x * D_col, prim_obj.cpu().item(), k, n, j

def pdlp_solver(mps_file_path, tol=1e-4, restart_period=40, verbose=True, max_iter=100_000, precondition=False, adaptive_step=False, primal_update=False):
    """
    Full Restarted PDHG solver implementation using PyTorch.

    Args:
      mps_file_path (str): Path to the MPS file.
      tol (float, optional): Tolerance for convergence. Defaults to 1e-4. Use 1e-8 for high accuracy
      term_period (int, optional): Period for termination checks. Defaults to 1000.
      verbose (bool, optional): Whether to print termination check information. Defaults to True.

    Returns:
      The minimizer, objective value, and number of iterations for convergence.
    """
    import torch

    # --- Device Selection ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"PyTorch is using ROCm/CUDA device: {torch.cuda.get_device_name(0)}") if verbose else None
    else:
        device = torch.device('cpu')
        print("ROCm/CUDA not available. PyTorch is using CPU.") if verbose else None

    # --- Parameter Loading ---
    try:
          c, K, q, m_ineq, l, u = mps_to_standard_form(mps_file_path, device=device, verbose=verbose)
    except Exception as e:
        print(f"Failed to load MPS file: {e}")
        exit(1)

    # --- Run PDHG Solver on the GPU or CPU ---
    minimizer, objective_value, total_iterations, total_restarts, kkt_passes = pdlp_algorithm(K, m_ineq, c, q, l, u, device, max_iter=max_iter, tol=tol, verbose=verbose, restart_period=restart_period, precondition=precondition, primal_update=primal_update, adaptive_step=adaptive_step)

    if verbose:
      print("Objective Value:", objective_value)
      print("Iterations:", total_iterations)
      print("Restarts:", total_restarts)
      print("KKT Passes:", kkt_passes)
      print("\nMinimizer (first 10 variables):")
      print(minimizer[:10].cpu().numpy())

    return minimizer, objective_value, total_iterations, total_restarts, kkt_passes

