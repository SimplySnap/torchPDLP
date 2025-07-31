import torch
from primal_dual_hybrid_gradient_step import adaptive_one_step_pdhg

def restarted_pdhg(c, G, h, A, b, l, u, device, max_iter=100_000, tol=1e-4, verbose=True, restart_period=40):
    
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
    omega = c_norm / q_norm if q_norm > 0 and c_norm > 0 else 1.0

    theta = 1.0

    # Restart Parameters [Sufficient, Necessary, Artificial]
    beta = [0.2, 0.8, 0.36]
  
    # Initialize primal and dual
    x = torch.zeros((n, 1), device=device)
    x_old = x.clone()
    y = torch.zeros((K.shape[0], 1), device=device)

    # Counters
    n = 0 # Outer Loop Counter
    k = 0 # Total Iteration Counter

    # Initialize Previous KKT Error
    KKT_first = 0 # The actual KKT error of the very first point doesn't matter since the artificial criteria will always hit anyway
  
    # -------------- Outer Loop --------------
    while k < max_iter:
      t = 0 # Initialize inner iteration counter

      # Initialize/Reset sums for averaging
      x_eta_total = torch.zeros_like(x)
      y_eta_total = torch.zeros_like(y)
      eta_total = 0
      # --------- Inner Loop ---------
      while k < max_iter:
        x_old = x.clone()
        y_old = y.clone()
        x, y = one_step_pdhg(x, y, c, q, K, l, u, m_ineq, eta, omega, theta)

        # Increase iteration counters
        k += 1
        t += 1

        # Update totals
        x_eta_total += eta * x
        y_eta_total += eta * y
        eta_total += eta

        # Check Restart Criteria Every restart_period iterations
        if t % restart_period == 0:
          # Compute averages
          x_avg = x_eta_total / eta_total
          y_avg = y_eta_total / eta_total

          # Compute KKT errors
          omega_sqrd = omega ** 2
          KKT_current = KKT_error(x, y, omega)
          KKT_average = KKT_error(x_avg, y_avg, omega)

          KKT_min = min(KKT_current, KKT_average)

          KKT_previous = KKT_error(x_old, y_old, omega) # For checking necessary criteria

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
      KKT_first = KKT_error(x, y, omega) # Update KKT_first for next iteration (incase primal weight updates)
      
      # Check Termination Criteria Every Restart

      # Primal and dual objective
      grad = c - K.T @ y
      prim_obj = (c.T @ x).flatten()
      dual_obj = (q.T @ y).flatten()

      # Lagrange multipliers from box projection
      lam = project_lambda_box(grad, is_neg_inf, is_pos_inf)
      lam_pos = (l_dual.T @ torch.clamp(lam, min=0.0)).flatten()
      lam_neg = (u_dual.T @ torch.clamp(lam, max=0.0)).flatten()

      # Duality Gap (optimality)
      adjusted_dual = dual_obj + lam_pos + lam_neg
      duality_gap = abs(adjusted_dual - prim_obj)

      # Primal residual (feasibility)
      residual_eq = A @ x - b if m_eq > 0 else torch.zeros(1, device=device)
      residual_ineq = torch.clamp(h - G @ x, min=0.0) if m_ineq > 0 else torch.zeros(1, device=device)
      primal_residual = torch.norm(torch.vstack([residual_eq, residual_ineq]), p=2).flatten()

      # Dual residual (feasibility)
      dual_residual = torch.norm(grad - lam, p=2).flatten()

      if verbose:
          print(f"[{k}] Primal Obj: {prim_obj.item():.4f}, Adjusted Dual Obj: {adjusted_dual.item():.4f}, "
                f"Gap: {duality_gap.item()/ (1 + abs(prim_obj.item()) + abs(adjusted_dual.item())):.2e}, Prim Res: {primal_residual.item() / (1 + q_norm):.2e}, Dual Res: {dual_residual.item() / (1 + c_norm):.2e}")
          print("")

      # Termination conditions
      if (primal_residual <= tol * (1 + q_norm) and
          dual_residual <= tol * (1 + c_norm) and
          duality_gap <= tol * (1 + abs(prim_obj) + abs(adjusted_dual))):
          if verbose:
              print(f"Converged at iteration {k} restart loop {n}")
          break
    # ------------------- End Outer Loop ------------------------          

    return x, prim_obj.cpu().item(), k, n