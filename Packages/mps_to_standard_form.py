import cplex
from cplex.exceptions import CplexError
import torch
import numpy as np

def mps_to_standard_form_torch(mps_file, device='cpu'):
    """
    Loads an MPS file using CPLEX and converts it into standard form LP in PyTorch:
    minimize     cᵀx
    subject to   G x ≥ h
                 A x = b
                 l ≤ x ≤ u

    Args:
        mps_file (str): Path to the MPS file.
        device (str): Device to use for PyTorch tensors ('cpu' or 'cuda').
    
    Returns as torch tensors: c, G, h, A, b, l, u 

    """
    try:
      # Parse .mps file using cplex parser
      cpx = cplex.Cplex(mps_file)
      cpx.set_results_stream(None)  # Mute output

      # Number of variables and constriants
      num_vars = cpx.variables.get_num()
      num_constraints = cpx.linear_constraints.get_num()

      # Upper and lower bounds and replace default max and min bounds with +-inf
      l = np.array(cpx.variables.get_lower_bounds())
      l = [x if x > -1.0000e+20 else -float('inf') for x in l]
      u = np.array(cpx.variables.get_upper_bounds())
      u = [x if x < 1.0000e+20 else float('inf') for x in u]

      # Objective vector
      c = np.array(cpx.objective.get_linear())

      # Extract Constraints row by row
      rows = cpx.linear_constraints.get_rows()
      senses = cpx.linear_constraints.get_senses()
      rhs = np.array(cpx.linear_constraints.get_rhs())

      A_rows, G_rows = [], []
      b_eq, h_ineq = [], []

      for row, sense, rhs_i in zip(rows, senses, rhs):
          row_vec = np.zeros(num_vars)
          for idx, val in zip(row.ind, row.val):
              row_vec[idx] = val

          if sense == 'E':
              A_rows.append(row_vec)
              b_eq.append(rhs_i)
          elif sense == 'G':  # ≥
              G_rows.append(row_vec)
              h_ineq.append(rhs_i)
          elif sense == 'L':  # ≤ → -row ≥ -rhs
              G_rows.append(-row_vec)
              h_ineq.append(-rhs_i)

      # Convert to numpy arrays for faster conversion to tensors
      A_rows = np.array(A_rows)
      b_eq = np.array(b_eq)
      G_rows = np.array(G_rows)
      h_ineq = np.array(h_ineq)

      # Convert to torch tensors
      c_tensor = torch.tensor(c, dtype=torch.float32, device=device).view(-1, 1)
      l_tensor = torch.tensor(l, dtype=torch.float32, device=device).view(-1, 1)
      u_tensor = torch.tensor(u, dtype=torch.float32, device=device).view(-1, 1)

      A_tensor = torch.tensor(A_rows, dtype=torch.float32, device=device) if A_rows.any() else torch.zeros((0, num_vars), device=device)
      b_tensor = torch.tensor(b_eq, dtype=torch.float32, device=device).view(-1, 1) if b_eq.any() else torch.zeros((0, 1), device=device)

      G_tensor = torch.tensor(G_rows, dtype=torch.float32, device=device) if G_rows.any() else torch.zeros((0, num_vars), device=device)
      h_tensor = torch.tensor(h_ineq, dtype=torch.float32, device=device).view(-1, 1) if h_ineq.any() else torch.zeros((0, 1), device=device)

      return c_tensor, G_tensor, h_tensor, A_tensor, b_tensor, l_tensor, u_tensor

    except CplexError as e:
        print("CPLEX Error:", e)
        return None
