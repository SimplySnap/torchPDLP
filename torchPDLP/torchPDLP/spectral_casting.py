import torch
import numpy as np
from .helpers import spectral_norm_estimate_torch, spectral_norm_estimate_torch, project_lambda_box

def spectral_cast(K,c,q,l,u,m_ineq,k,s=2,i=5,device="cpu"):
    '''
    Main function which takes as input needed data types, and i (exponent of points) to cast.
    Returns a starting point (x, y) that is a promising initial guess for the primal-dual problem,
    typically better than the zero vector.

    Args:
        K (torch.Tensor): Constraint matrix.
        c (torch.Tensor): Primal objective vector.
        q (torch.Tensor): Dual objective vector.
        l (torch.Tensor): Lower bound vector.
        u (torch.Tensor): Upper bound vector.
        m_ineq (int): Number of inequality constraints.
        k (int): Number of PDHG iterations to perform between point reduction steps.
        s (int, optional): Proportion to reduce points by at each step. Default 2.
        i (int, optional): Exponent for number of initial points (2^i points). Default 5.
        device (str, optional): Device to use. Default "cpu".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: x, y — promising primal and dual starting points.
    '''
    pts, radius = sample_points(K,i,device) #  Cast points randomly around primal-ball
    start_x, start_y = fishnet(pts,K,c,q,l,u,m_ineq,s,k,device)

    return start_x, start_y


def sample_points(K,i,device="cpu"):
    '''
    Generates 2^i random points in an n-dimensional ball, where n is the number of columns in K,
    and the ball's radius is estimated by the spectral norm of K.

    Args:
        K (torch.Tensor): Constraint matrix.
        i (int): Exponent for the number of points to generate (2^i points).
        device (str, optional): Device to use. Default "cpu".

    Returns:
        tuple[torch.Tensor, float]: A matrix of points (each column is a point),
                                    and the spectral radius (ball radius).
    '''

    r = spectral_norm_estimate_torch(K,25) #  25 because we want a tight ball, pause
    dim = K.shape[1] #Get num columns for casting
    j = 2**i #Number of points based on i

    points = torch.randn(size=(dim,j),device=device) #  j points, each of length dim in primal-space

    #  Center on the positive diagonal so the sphere touches axes but never crosses
    centre = (r / dim ** 0.5) * torch.ones((dim, 1), device=device)

    #  Cast points around ball
    #  Normalize points to fit unit n-sphere, then scale by spectral radius
    points = points*r / torch.norm(points, dim=0, keepdim=True) 
    
    points += centre #  Translate to centre AT END of scaling


    return points, r #  Returns vector of vector-primal-points and radius r

def fishnet(pts,K,c,q,l,u,m_ineq,s=2,k=32, device="cpu"):
    '''
    Iteratively reduces a set of primal and dual points by running PDHG, evaluating their
    duality gap, and "breeding" new points from the best candidates.
    Returns the single best primal-dual pair found.

    Inspired by evolutionary algorithms (population-based) and crossover method

    Args:
        pts (torch.Tensor): Initial primal points (n x 2^i).
        K (torch.Tensor): Constraint matrix.
        c (torch.Tensor): Primal objective vector.
        q (torch.Tensor): Dual objective vector.
        l (torch.Tensor): Lower bound vector.
        u (torch.Tensor): Upper bound vector.
        m_ineq (int): Number of inequality constraints.
        k (int): Number of PDHG iterations per reduction.
        s (int, optional): Proportion to reduce points by at each step. Default 2.
        device (str, optional): Device to use. Default "cpu".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The best primal and dual points found (flattened).
    '''

    j = pts.shape[1] #  number of points, power of 2, num cols
    k = k #  Number of iterations we do for single PDHG pass before checking optimality
    '''
    In this loop, we carry out PDHG on each of the points k times.
    Then, we check point optimality based on duality gap (versus KKT residual, which is too time intensive)
    Then, if i is even, we halve the points, and if i is odd, we halve them and breed a new half to get better choices of points.
    
    Breeding criteria: we randomly (normal) select weights for the remaining p points such that weights sum to 1. We do this p times to generate p more points, meaning each new point is a linear combination of the previous best half of points.
    '''
    #  Get starting y points and other vars
    pts_y = K @ pts
    count = pts.shape[1]
    eta,omega,is_pos_inf,is_neg_inf, = init_PDHG_vars(K,c,q,l,u)

    i = 0 #  Counter for while loop, also tracks parity for breeding
    while j > 1:
        #  Main loop that reduces the number of points until there is only one
        for _ in range(k):
            #  Run PDHG j times
            pts, pts_y = PDHG_step(K,pts,pts_y,c,q,l,u,eta,omega,m_ineq,device)
            
        #  Now, we evaluate top s proportion of points
        old_j = pts.shape[1]
        count += 2*pts.shape[1]*k #  For counting KKT passes - pts,pts_y KKT passes
        pts,pts_y = get_best_pts(K,pts,pts_y,c,q,l,u,is_pos_inf,is_neg_inf,s)
        new_j = pts.shape[1]

        #  Parity - if i odd then repopulate fully, if i odd, then don't
        if i % 2 == 1 and new_j > 1:
            #  Repopulate with midpoint
            
            midpoint = pts.mean(dim=1, keepdim=True)      # shape: (n, 1)
            midpoint_y = pts_y.mean(dim=1, keepdim=True)      # shape: (n, 1)
            
            #  We append these later so they don't interfere with breeding 
            #  (both shape and information interference)

            #  Create lists to collect new points
            new_pts = []
            new_pts_y = []


            #  No we cast random combinations of our remaining points 
            for t in range(old_j-new_j-1):

                #  Sample random positive weights and normalize to sum to 1
                weights = torch.rand(new_j, device=pts.device)
                weights = weights / weights.sum()

                #  Compute convex combinations for primal and dual points
                new_pt = (pts @ weights.view(-1,1)).squeeze(1)      #  shape (n)
                new_pt_y = (pts_y @ weights.view(-1,1)).squeeze(1)  #  shape (m)

                #  Add as new column
                # Collect the new points
                new_pts.append(new_pt.unsqueeze(1))
                new_pts_y.append(new_pt_y.unsqueeze(1))

            #  Concatenate all new points at once
            if new_pts:  #  Only if we have new points to add
                all_new_pts = torch.cat(new_pts, dim=1)  #  shape: (n, num_new_points)
                all_new_pts_y = torch.cat(new_pts_y, dim=1)  #  shape: (m, num_new_points)
                
                #  Append all new points and midpoint
                pts = torch.cat([pts, all_new_pts, midpoint], dim=1)
                pts_y = torch.cat([pts_y, all_new_pts_y, midpoint_y], dim=1)
            
        j = pts.shape[1]
        i+=1 #  Add to index

    return pts.flatten(), pts_y.flatten()

def init_PDHG_vars(K,c,q,l,u):
    '''
    Initialize step size (eta), primal weight (omega), and other variables for PDHG.

    Args:
        K (torch.Tensor): Constraint matrix.
        c (torch.Tensor): Primal objective vector.
        q (torch.Tensor): Dual objective vector.
        l (torch.Tensor): Lower bound vector.
        u (torch.Tensor): Upper bound vector.

    Returns:
        tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
            eta: PDHG step size
            omega: primal weight (c_norm / q_norm)
            is_pos_inf: boolean mask of where u is +infinity
            is_neg_inf: boolean mask of where l is -infinity
    '''

    q_norm = torch.linalg.norm(q, 2)
    c_norm = torch.linalg.norm(c, 2)

    eta = 0.9 / spectral_norm_estimate_torch(K, num_iters=50)
    omega = c_norm / q_norm if q_norm > 1e-6 and c_norm > 1e-6 else torch.tensor(1.0, device = q.device)

    is_neg_inf = torch.isinf(l) & (l < 0)
    is_pos_inf = torch.isinf(u) & (u > 0)

    return eta,omega,is_pos_inf,is_neg_inf

def get_best_pts(K,pts,pts_y,c,q,l,u,is_pos_inf,is_neg_inf,s=2):
    '''
    Selects the best s-proportion of points based on duality gap.
    Doesn't use KKT residual for efficiency.

    Args:
        K (torch.Tensor): Constraint matrix.
        pts (torch.Tensor): Primal points (n x num_pts).
        pts_y (torch.Tensor): Dual points (m x num_pts).
        c (torch.Tensor): Primal objective vector.
        q (torch.Tensor): Dual objective vector.
        l (torch.Tensor): Lower bound vector.
        u (torch.Tensor): Upper bound vector.
        is_pos_inf (torch.Tensor): Mask for +infinity upper bounds.
        is_neg_inf (torch.Tensor): Mask for -infinity lower bounds.
        s (int, optional): Proportion to select (1/s points survive). Default 2.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Primal and dual points (chopped)
    '''

    #  Step 1 : Duality gap
    l_dual = l.clone()
    u_dual = u.clone()
    l_dual[is_neg_inf] = 0
    u_dual[is_pos_inf] = 0

    #  Primal and dual objective 1D tensors
    grad = c - K.T @ pts_y
    prim_obj = torch.matmul(c.T, pts)      #  shape: (num_points,)
    dual_obj = torch.matmul(q.T, pts_y)    #  shape: (num_points,)

    duality_gap = [] #  initialize duality gap vector
    #  Loop over columns/points to create duality gap vector
    for i in range(grad.shape[1]):
        grad_column = grad[:,i].unsqueeze(1) #  Single grad column - corresponds to grad
        lam = project_lambda_box(grad_column, is_neg_inf, is_pos_inf)  # 1D
        lam_pos = (l_dual * torch.clamp(lam, min=0.0)).sum()
        lam_neg = (u_dual * torch.clamp(lam, max=0.0)).sum()
        adjusted_dual = dual_obj[0, i] + lam_pos + lam_neg
        gap = adjusted_dual - prim_obj[0, i]
        duality_gap.append(gap)

    duality_gap = torch.stack(duality_gap)  #  shape: (num_points,)
    

    #  Step 2 : Sort vectors
    sorted_indices = torch.argsort(duality_gap) #  Get indices in ascending order
    pts = pts[:,sorted_indices]
    pts_y = pts_y[:,sorted_indices]

    #  Step 3 : Chop vectors

    #  n - number of columns 
    _, n = pts.shape

    num_cols = max(1,n//s) #  s parameter controls how much we chop at each stage

    pts = pts[:,:num_cols] #  Chop s.t first 1/s proportion of points remain
    pts_y = pts_y[:,:num_cols]

    return pts,pts_y

def PDHG_step(K,pts,pts_y,c,q,l,u,eta,omega,m_ineq,device="cpu"):
    '''
    Performs one PDHG iteration on a matrix of primal and dual points.

    Args:
        K (torch.Tensor): Constraint matrix.
        pts (torch.Tensor): Primal points (n x num_pts).
        pts_y (torch.Tensor): Dual points (m x num_pts).
        c (torch.Tensor): Primal objective vector.
        q (torch.Tensor): Dual objective vector.
        l (torch.Tensor): Lower bound vector.
        u (torch.Tensor): Upper bound vector.
        eta (float): Step size.
        omega (float): Primal weight.
        m_ineq (int): Number of inequality constraints.
        device (str, optional): Device to use. Default "cpu".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Updated primal and dual matrices.
    '''

    pts_old = pts.clone()
    K_t_y_pts = K.T @ pts_y #  Matrix containing grads for all y points around k-sphere
    #  Calculate grad for all said points
    grad_y_pts = c - K_t_y_pts #  mrow

    #  Clamp new x points to get new vector
    pts = torch.clamp(pts - ((eta / omega) * grad_y_pts), min = l, max = u)

    x_bar_pts = 2*pts - pts_old #  Momentum for stability

    #  Get new y points
    K_x = K @ x_bar_pts
    pts_y += eta * omega * (q - K_x)

    #  Clamp the for inequality bounds in dual
    if m_ineq > 0:
        pts_y[:m_ineq, :] = torch.clamp(pts_y[:m_ineq, :], min=0.0)

    return pts, pts_y

