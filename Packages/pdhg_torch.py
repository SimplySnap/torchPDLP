import torch
import numpy as np

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

def pdhg_torch(c, G, h, A, b, l, u, is_neg_inf, is_pos_inf, l_dual, u_dual, device, max_iter=1000, tol=1e-4, verbose=True, term_period=1000):
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
  
  
    eta = 1 / spectral_norm_estimate_torch(K, num_iters=100)
    omega = 1.0

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
