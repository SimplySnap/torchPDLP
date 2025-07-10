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
    import cplex
    from cplex.exceptions import CplexError
    import torch
    import numpy as np
    
    try:
      # Parse .mps file using cplex parser
      cpx = cplex.Cplex(mps_file)
      cpx.set_results_stream(None)  # Mute output

      # Number of variables and constriants
      num_vars = cpx.variables.get_num()
      
     # Upper and lower bounds, replacing default max/min with infinity
      l = np.array(cpx.variables.get_lower_bounds())
      l[l <= -1e+20] = -float('inf')
      u = np.array(cpx.variables.get_upper_bounds())
      u[u >= 1e+20] = float('inf')
      
      # Objective vector
      c = np.array(cpx.objective.get_linear())

      # Extract Constraints row by row
      rows = cpx.linear_constraints.get_rows()
      senses = cpx.linear_constraints.get_senses()
      rhs = np.array(cpx.linear_constraints.get_rhs())
      ranges = np.array(cpx.linear_constraints.get_range_values())
      
      A_rows, G_rows = [], []
      b_eq, h_ineq = [], []

      for i in range(len(rows)):
            row = rows[i]
            sense = senses[i]
            rhs_i = rhs[i]
            range_val = ranges[i]

            row_vec = np.zeros(num_vars)
            for idx, val in zip(row.ind, row.val):
                row_vec[idx] = val
            
            # --- MODIFIED LOGIC: Handle ranged and non-ranged constraints ---
            if sense == 'R':
                lb = rhs_i
                ub = rhs_i + range_val

                if ub < lb:
                    lb, ub = ub, lb  # swap to ensure lb ≤ ub

                # 1. row ≥ lb
                G_rows.append(row_vec)
                h_ineq.append(lb)

                # 2. row ≤ ub --> -row ≥ -ub
                G_rows.append(-row_vec)
                h_ineq.append(-ub)
                
            else:
                # No range, handle as a simple constraint
                if sense == 'E':
                    A_rows.append(row_vec)
                    b_eq.append(rhs_i)
                elif sense == 'G':  # ≥
                    G_rows.append(row_vec)
                    h_ineq.append(rhs_i)
                elif sense == 'L':  # ≤  
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

      A_tensor = torch.tensor(A_rows, dtype=torch.float32, device=device)
      b_tensor = torch.tensor(b_eq, dtype=torch.float32, device=device).view(-1, 1)

      G_tensor = torch.tensor(G_rows, dtype=torch.float32, device=device)
      h_tensor = torch.tensor(h_ineq, dtype=torch.float32, device=device).view(-1, 1)

      return c_tensor, G_tensor, h_tensor, A_tensor, b_tensor, l_tensor, u_tensor

    except CplexError as e:
        print("CPLEX Error:", e)
        return None