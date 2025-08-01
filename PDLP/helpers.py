import torch

def sparse_vs_dense(A, device='cpu', kkt_passes=10):
    """
    Benchmarks matrix-vector multiplication using dense and sparse formats.
    
    Parameters:
        A (torch.Tensor): 2D matrix (dense tensor)
        device (str): 'cpu' or 'cuda'
        num_trials (int): Number of repetitions for timing

    Returns:
        tensor A as either sparse or dense, which ever is faster
    """
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

def spectral_norm_estimate_torch(K, num_iters=10):
  """
  Estimates the spectral norm of a matrix K with enough acuracy to use in
  setting the step size of the PDHG algorithm.
  """

  b = torch.randn(K.shape[1], 1, device=K.device)
  for _ in range(num_iters):
      b = K.T @ (K @ b)
      b /= torch.norm(b)
  return torch.norm(K @ b)

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
