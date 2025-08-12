import torch
import numpy as np
from helpers import spectral_norm_estimate_torch


def sample_points(K,i,device="cpu"):
    '''
    This function does two things. The first, we create an n-dim ball whose radius is the size of the spectral norm. The second, we sample j points over this ball, and output their vector values.
    The dimension of x, denoted n, is the length of c AKA width of K

    Args:
        K : Full constraint matrix
        i : exponent of number of points we cast. EG if i = 4, we have 2**4 points on our sphere
    '''

    r = spectral_norm_estimate_torch(K,25) #  25 because we want a tight ball, pause
    dim = K.shape[1] #Get num columns for casting
    j = 2**i #Number of points based on i

    points = torch.normal(size=(dim,j),device=device) #  j points, each of length dim in primal-space

    #Now, we gotta plot around radius
    centre = torch.normal(size=(dim,1),device=device) #  random center TODO - change st sphere edge is origin

    #  Cast points around
    points = points*r / torch.norm(points, dim=0, keepdim=True, device=device) #  Normalize points to fit unit n-sphere, then scale by spectral radius
    
    points = points+centre #  Translate to centre AT END of scaling


    return points, r #  Returns vector of vector-primal-points and radius r

def init_PDHG_vars(K,c,q,l,u):
    '''
    Initialize step size, primal weights, m_ineq and other variables needed for PDHG

    Args:
        K (torch.Tensor) : Main constraint
        c (torch.Tensor) : primal objective variable
        q (torch.Tensor) : dual objective variable
        l (torch.Tensor) : lower something I forgor lmaoo
        u (torch.Tensor) : see above (im faded ngl)

    Returns:


    '''

    q_norm = torch.linalg.norm(q, 2)
    c_norm = torch.linalg.norm(c, 2)

    eta = 0.9 / spectral_norm_estimate_torch(K, num_iters=100)
    omega = c_norm / q_norm if q_norm > 1e-6 and c_norm > 1e-6 else torch.tensor(1.0)

    is_neg_inf = torch.isinf(l) & (l < 0)
    is_pos_inf = torch.isinf(u) & (u > 0)

    return eta,omega,is_pos_inf,is_neg_inf

def fishnet(pts,K,x,y,c,q,l,u,m_ineq,t,k=32, device="cpu"):
    '''ewewe'''

    t = t #  Exponent of points, so we 
    j = pts.shape[1] #  number of points, power of 2, num cols
    k = k #  Number of iterations we do for single PDHG pass step before checking optimality
    '''
    In this loop, we carry out PDHG on each of the points k times.
    Then, we check point optimality based on duality gap (versus KKT residual, which is too time intensive)
    Then, if i is even, we halve the points, and if i is odd, we halve them and breed a new half to get better choices of points.
    
    Breeding criteria: we randomly (normal) select weights for the remaining p points such that weights sum to 1. We do this p times to generate p more points, meaning each new point is a linear combination of the previous best half of points.
    '''
    #  Get starting y points and other vars
    pts_y = K @ pts
    eta,omega,is_pos_inf,is_neg_inf, = init_PDHG_vars(K,c,q,l,u)

    i = 0 #  Counter for while loop, also tracks parity for breeding
    while j != 1:
        #  Main loop that reduces the number of points until there is only one
        for j in range(k):
            #  Run PDHG j times
            pts, pts_y = PDHG_step(K,pts,pts_y,c,q,l,u,eta,omega,m_ineq,device)
        #  Now, we evaluate top half of points


        j = pts.shape[1]

    return


def get_best_pts(pts,pts_y,c,q,is_pos_inf,is_neg_inf,s=2):
    '''
    Takes as input x_points, y_points (both 2d tensors), computes duality gap, and 
    returns new tensors of points containing 'best' s-proportion of both tensors' points
    w.r.t duality gap
    Note we don't use KKT residual because it's too expensive when duality gaps [should] do the trick


    '''

    #  Step 1 : Duality gap
    l_dual = l.clone()
    u_dual = u.clone()
    l_dual[is_neg_inf] = 0
    u_dual[is_pos_inf] = 0


# TODO - adapt below to 2D tensor 
    #  Primal and dual objective
    grad = c - K.T @ y
    prim_obj = (c.T @ x).flatten()
    dual_obj = (q.T @ y).flatten()

    #  Lagrange multipliers from box projection
    lam = project_lambda_box(grad, is_neg_inf, is_pos_inf)
    lam_pos = (l_dual.T @ torch.clamp(lam, min=0.0)).flatten()
    lam_neg = (u_dual.T @ torch.clamp(lam, max=0.0)).flatten()
    #  Duality gap
    adjusted_dual = dual_obj + lam_pos + lam_neg
    duality_gap = adjusted_dual - prim_obj
    

    #  Step 2 : Sort vectors

    #  TODO : research tensor sorting algs to sort columns

    #  Step 3 : Chop vectors

    #  n - number of columns 
    _, n = pts.shape

    index = torch.floor(n / s).clamp(min=1) #  s parameter controls how much we chop at each stage

    pts = pts[:,:s]
    pts_y = pts_y[:,:s]






    
    

def PDHG_step(K,pts,pts_y,c,q,l,u,eta,omega,m_ineq,device="cpu"):
    '''
    This step, we take a matrix of x-points, a matrix of y-points and run PDHG on them to return a matrix of new x-points & y-points.

    Args: 

        K (torch.Tensor) : Matrix
        pts (torch.Tensor) : primal matrix where each column is a point in primal dimension
        pts_y (torch.Tensor) : dual matrix where each col is a point in dual dimension
        c (torch.Tensor) : objective variable primal
        q (torch.Tensor) : objective variable dual
        l (torch.Tensor) : lower bound
        u (torch.Tensor) : upper bound
        eta (float) : step size
        omega (float) : primal weight
        m_ineq (int) : Number of inequality constraints.
    
    Returns:
        pts (torch.Tensor) : updated primal variables
        pts_y (torch.Tensor) : updated dual variables
    '''

    # TODO check if our 'unsqueeze' should be a different dimension
    pts_old = pts.clone()
    K_t_y_pts = K.T @ pts_y #  Matrix containing grads for all y points around k-sphere
    #  Calculate grad for all said points
    grad_y_pts = K_t_y_pts + c.unsqueeze(1) #  mrow

    #  Clamp new x points to get new vector
    pts = torch.clamp(pts - ((eta / omega) * grad_y_pts), min = l.unsqueeze(1), max = u.unsqueeze(1))

    x_bar_pts = 2*pts - pts_old #  Momentum for stability

    #  Get new y points
    K_x = K @ x_bar_pts
    pts_y += eta * omega * (q.unsqueeze(1) - K_x)

    #  Clamp the for inequality bounds in dual
    if m_ineq > 0:
        pts_y[:m_ineq, :] = torch.clamp(pts_y[:m_ineq, :], min=0.0)

    return pts, pts_y

