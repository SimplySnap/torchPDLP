import torch

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

def KKT_error(x, y, c, q, K, m_ineq, omega, is_neg_inf, is_pos_inf, l_dual, u_dual, device):
      """
      Computes the KKT error using global variables.
      """
      omega_sqrd = omega ** 2
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

      # Compute the error
      KKT = torch.sqrt(omega_sqrd * primal_residual ** 2 + (dual_residual ** 2) / omega_sqrd + duality_gap ** 2)

      return KKT