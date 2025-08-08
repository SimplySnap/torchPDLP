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
    centre = torch.normal(size=(dim,1),device=device) #  random center TODO

    #  Cast points around
    points = points*r / torch.norm(points, dim=0, keepdim=True, device=device) #  Normalize points to fit unit n-sphere, then scale by spectral radius
    
    points = points+centre #  Translate to centre AT END of scaling


    return points, r

def scatter(pts,K,x,y,c,q,l,u,m_ineq,t,k=32, device="cpu"):
    '''ewewe'''

    t = t #  Exponent of points, so we 
    j = pts.shape[0] #  number of points, power of 2
    k = k #  Number of iterations we do for single PDHG pass step before checking optimality
    i = 0 #  Counts number of passes, and also tracks parity for breeding
    '''
    In this loop, we carry out PDHG on each of the points k times.
    Then, we check point optimality based on duality gap (versus KKT residual, which is too time intensive)
    Then, if i is even, we halve the points, and if i is odd, we halve them and breed a new half to get better choices of points.
    
    Breeding criteria: we randomly (normal) select weights for the remaining p points such that weights sum to 1. We do this p times to generate p more points, meaning each new point is a linear combination of the previous best half of points.
    '''

    return

def PDHG_step(K,pts,pts_y,c,q,l,u,m_ineq,device="cpu"):
    '''
    This step, we take a matrix of x-points, a matrix of y-points and run PDHG on them to return a matrix of new x-points & y-points.
    '''
    return


