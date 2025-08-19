import torch
import time

def preconditioning(c, K, q, l, u, device='cpu', max_iter=20, eps=1e-6):
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
    - The scaling preserves feasibility and optimality but improves numerical conditioning.
    - You must rescale your solution after solving using D_col (and D_row if needed).
    """
    time_start = time.time()
    # --- Scaling Loop ---
    K_s, c_s, q_s, l_s, u_s = K.clone(), c.clone(), q.clone(), l.clone(), u.clone()
    m, n = K_s.shape

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
    
    time_used = time.time() - time_start

    return K_s, c_s, q_s, l_s, u_s, (D_col, D_row, K, c, q, l, u), time_used

def primal_weight_update(x_prev, x, y_prev, y, omega, smooth_theta):
    diff_y_norm = torch.linalg.norm(y_prev - y, 2)
    diff_x_norm = torch.linalg.norm(x_prev - x, 2)
    if diff_x_norm > 0 and diff_y_norm > 0:
        omega = torch.exp(smooth_theta * (torch.log(diff_y_norm/diff_x_norm)) + (1-smooth_theta)*torch.log(omega))
    return omega

def detect_infeasibility(x, y, x_prev, y_prev, lam, lam_prev, c, q, K, l, u, m_ineq, device, tol=1e-2):
    """
    Detect primal and dual infeasibility using current and previous iterates.
    
    Args:
        x: Current primal variable
        y: Current dual variable  
        x_prev: Previous primal variable
        y_prev: Previous dual variable
        lam_prev: Previous lambda (projected gradient)
        c: Primal objective coefficients
        q: Dual objective coefficients (RHS)
        K: Constraint matrix
        l: Lower bounds
        u: Upper bounds
        m_ineq: Number of inequality constraints
        is_neg_inf: Boolean mask for negative infinite bounds
        is_pos_inf: Boolean mask for positive infinite bounds
        device: Torch device
        tol: Combined tolerance for all infeasibility checks
    
    Returns:
        str or None: "DUAL_INFEASIBLE", "PRIMAL_INFEASIBLE", or None if feasible
    """
    
    # Compute current lambda and differences
    dx = x - x_prev
    dy = y - y_prev
    dlam = lam - lam_prev
    
    # Dual infeasibility (primal unbounded) detection
    dlam_plus = (-dlam).clamp(min=0)
    dlam_minus = dlam.clamp(min=0)
    
    # Check dual infeasibility conditions
    # Check equality constraint: A @ dx = 0 (using norm < tol)
    equality_ok = (K[m_ineq:] @ dx).norm() < tol
    
    # Check inequality constraint: G @ dx >= -tol  
    inequality_ok = torch.all(K[:m_ineq] @ dx >= -tol)
    
    # Check objective condition: c^T @ dx < 0
    objective_ok = (c.T @ dx) < tol
    
    if equality_ok and inequality_ok and objective_ok:
        bounds_ok = True
        for i in range(x.shape[0]):
            dx_i = dx[i]
            c_i = c[i]
            l_i = l[i]
            u_i = u[i]
            
            if not(
                (not torch.isinf(l[i]) and not torch.isinf(u[i]) and torch.abs(dx_i) <= tol) or
                (u_i == float('inf') and c_i >= 0 and dx_i >= -tol) or
                (l_i == -float('inf') and c_i <= 0 and dx_i <= tol)
            ):
                bounds_ok = False
                break
        
        if bounds_ok:
            return "DUAL_INFEASIBLE"
    
    # Primal infeasibility (dual unbounded) detection
    # Compute dual residual: G^T @ dy_in + A^T @ dy_eq - dlam
    dual_res = K[:m_ineq].T @ dy[:m_ineq] + K[m_ineq:].T @ dy[m_ineq:] - dlam
    
    if dual_res.norm() < tol and torch.all(dy[:m_ineq] >= -tol):
        dual_combo = (q[:m_ineq].T @ dy[:m_ineq]).item() + (q[m_ineq:].T @ dy[m_ineq:]).item()
        
        finite_l = (~torch.isinf(l).view(-1)) & (l.view(-1) != 0)
        finite_u = (~torch.isinf(u).view(-1)) & (u.view(-1) != 0)
        
        if finite_l.any():
            dual_combo -= (l[finite_l].view(1, -1) @ dlam_minus[finite_l].view(-1, 1)).item()
        if finite_u.any():
            dual_combo -= (u[finite_u].view(1, -1) @ dlam_plus[finite_u].view(-1, 1)).item()
        
        if dual_combo > -tol:
            return "PRIMAL_INFEASIBLE"
    
    return None
