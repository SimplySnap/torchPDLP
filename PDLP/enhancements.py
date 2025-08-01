import torch

def ruiz_precondition(c, G, h, A, b, l, u, device='cpu', max_iter=20, eps=1e-6):
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
    G  : (m_ineq x n) torch tensor — inequality constraint matrix
    h  : (m_ineq x 1) torch tensor — inequality RHS vector
    A  : (m_eq x n) torch tensor — equality constraint matrix
    b  : (m_eq x 1) torch tensor — equality RHS vector
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

    return K_s, m_ineq, c_s, q_s, l_s, u_s, D_col

def primal_weight_update(x_prev, x, y_prev, y, omega, smooth_theta):
    diff_y_norm = torch.linalg.norm(y_prev - y, 2)
    diff_x_norm = torch.linalg.norm(x_prev - x, 2)
    if diff_x_norm > 0 and diff_y_norm > 0:
        omega = torch.exp(smooth_theta * (torch.log(diff_y_norm/diff_x_norm)) + (1-smooth_theta)*torch.log(omega))
    return omega
