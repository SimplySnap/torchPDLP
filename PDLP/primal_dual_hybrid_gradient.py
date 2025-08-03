import torch
from primal_dual_hybrid_gradient_step import adaptive_one_step_pdhg, fixed_one_step_pdhg
from helpers import spectral_norm_estimate_torch, KKT_error, compute_residuals_and_duality_gap, check_termination
from enhancements import primal_weight_update

def pdlp_algorithm(K, m_ineq, c, q, l, u, device, max_iter=100_000, tol=1e-4, verbose=True, restart_period=40, precondition=False, primal_update=False, adaptive=False, data_precond=None):
    
    is_neg_inf = torch.isinf(l) & (l < 0)
    is_pos_inf = torch.isinf(u) & (u > 0)

    l_dual = l.clone()
    u_dual = u.clone()
    l_dual[is_neg_inf] = 0
    u_dual[is_pos_inf] = 0
        
    q_norm = torch.linalg.norm(q, 2)
    c_norm = torch.linalg.norm(c, 2)

    eta = 0.9 / spectral_norm_estimate_torch(K, num_iters=100)
    omega = c_norm / q_norm if q_norm > 1e-6 and c_norm > 1e-6 else 1.0

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

            if adaptive:
                # Adaptive step of pdhg
                x, y, eta, eta_hat, j = adaptive_one_step_pdhg(x, y, c, q, K, l, u, m_ineq, eta, omega, theta, k, j)
            else:
                # Fixed step of pdhg
                x, y, eta, eta_hat= fixed_one_step_pdhg(x, y, c, q, K, l, u, m_ineq, eta, omega, theta)
                j += 1
           
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
      
        # Compute primal and dual residuals, and duality gap
        if precondition:
            D_col, D_row, K_unscaled, c_unscaled, q_unscaled, l_unscaled, u_unscaled = data_precond
            l_unscaled[is_neg_inf] = 0
            u_unscaled[is_pos_inf] = 0
            primal_residual, dual_residual, duality_gap, prim_obj, adjusted_dual = compute_residuals_and_duality_gap(D_col * x, D_row * y, c_unscaled, q_unscaled, K_unscaled, m_ineq, is_neg_inf, is_pos_inf, l_unscaled, u_unscaled)
        else:
            primal_residual, dual_residual, duality_gap, prim_obj, adjusted_dual = compute_residuals_and_duality_gap(x, y, c, q, K, m_ineq, is_neg_inf, is_pos_inf, l_dual, u_dual)
        # Add one kkt pass
        j += 1

        if verbose:
            print(f"[{k}] Primal Obj: {prim_obj.item():.4f}, Adjusted Dual Obj: {adjusted_dual.item():.4f}, "
                    f"Gap: {duality_gap.item()/ (1 + abs(prim_obj.item()) + abs(adjusted_dual.item())):.2e}, Prim Res: {primal_residual.item() / (1 + q_norm):.2e}, Dual Res: {dual_residual.item() / (1 + c_norm):.2e}")
            print("")

        # Termination conditions
        if check_termination(primal_residual, dual_residual, duality_gap, prim_obj, adjusted_dual, q_norm, c_norm, tol):
            if verbose:
              print(f"Converged at iteration {k} restart loop {n}")
            break
    # ------------------- End Outer Loop ------------------------
    
    return x, prim_obj.cpu().item(), k, n, j
