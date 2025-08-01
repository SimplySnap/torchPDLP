import torch

def fixed_one_step_pdhg(x, y, c, q, K, l, u, m_ineq, eta, omega, theta):
    """
    Perform one step of the Primal-Dual Hybrid Gradient (PDHG) algorithm.
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
    Returns:
        x (torch.Tensor): Updated primal variable.
        y (torch.Tensor): Updated dual variable.
    """ 
    x_old = x.clone()
    
    # Compute gradient and primal update
    Kt_y = K.T @ y
    grad = c - Kt_y
    x = torch.clamp(x - eta / omega * grad, min=l, max=u)

    # Extrapolate
    x_bar = x + theta * (x - x_old)

    # Dual update
    K_xbar = K @ x_bar
    y += eta * omega * (q - K_xbar)

    # Project dual:
    if m_ineq > 0:
        y[:m_ineq] = torch.clamp(y[:m_ineq], min=0.0)
    
    return x, y, eta, eta


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
        eta (float): Updated step size.
        eta_prime (float): New step size for the next iteration.
        j (int): Updated KKT pass number.
    """ 
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