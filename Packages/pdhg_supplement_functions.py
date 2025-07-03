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
