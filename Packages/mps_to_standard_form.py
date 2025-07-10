def mps_to_standard_form_torch_cplex(mps_file, device='cpu'):
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
    
def mps_to_standard_form_torch(mps_file, device='cpu'):
    """
    Parses an MPS file and returns the standard form LP components as PyTorch tensors:
        minimize     cᵀx
        subject to   G x ≥ h
                     A x = b
                     l ≤ x ≤ u

    Returns: c, G, h, A, b, l, u
    """
    import torch
    import numpy as np
    from collections import defaultdict
    
    #Read MPS file
    with open(mps_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('*')]

    section = None
    row_types = {}
    row_indices = {}
    col_data = defaultdict(list)
    rhs_data = {}
    range_data = {}
    bound_data = defaultdict(dict)

    row_counter = 0
    var_names = set()
    obj_row_name = None

    for line in lines:
        
        if line == 'NAME' or line == 'ENDATA':
            continue
        elif line == 'ROWS':
            section = 'ROWS'
            continue
        elif line == 'COLUMNS':
            section = 'COLUMNS'
            continue
        elif line == 'RHS':
            section = 'RHS'
            continue
        elif line == 'RANGES':
            section = 'RANGES'
            continue
        elif line == 'BOUNDS':
            section = 'BOUNDS'
            continue
    
        tokens = line.split()
        if section == 'ROWS':
            sense, row_name = tokens
            row_types[row_name] = sense
            row_indices[row_name] = row_counter
            if sense == 'N':
                obj_row_name = row_name
            row_counter += 1

        elif section == 'COLUMNS':
            var_name = tokens[0]
            var_names.add(var_name)
            for i in range(1, len(tokens), 2):
                row, val = tokens[i], float(tokens[i + 1])
                col_data[var_name].append((row, val))

        elif section == 'RHS':
            for i in range(1, len(tokens), 2):
                row, val = tokens[i], float(tokens[i + 1])
                rhs_data[row] = val

        elif section == 'RANGES':
            for i in range(1, len(tokens), 2):
                row, val = tokens[i], float(tokens[i + 1])
                range_data[row] = val

        elif section == 'BOUNDS':
            bound_type, _, var_name = tokens[:3]
            val = float(tokens[3]) if len(tokens) > 3 else None
            if bound_type == 'LO':
                bound_data[var_name]['lo'] = val
            elif bound_type == 'UP':
                bound_data[var_name]['up'] = val
            elif bound_type == 'FX':
                bound_data[var_name]['lo'] = val
                bound_data[var_name]['up'] = val
            elif bound_type == 'FR':
                # bound_data[var_name]['lo'] = -float('inf')
                bound_data[var_name]['lo'] = 0.0
                bound_data[var_name]['up'] = float('inf')
                
    # Check correct information loaded
    # print(row_types, col_data,rhs_data,range_data,bound_data)
    
    # Final variable ordering and index mapping
    var_names = sorted(var_names)
    var_index = {v: i for i, v in enumerate(var_names)}
    num_vars = len(var_names)

    # Build objective vector c
    c = np.zeros(num_vars)
    for var, entries in col_data.items():
        col_idx = var_index[var]
        for row_name, val in entries:
            if row_name == obj_row_name:
                c[col_idx] = val

    # Build row vectors from col_data
    row_vectors = {row: np.zeros(num_vars) for row in row_types}
    for var, entries in col_data.items():
        col_idx = var_index[var]
        for row_name, val in entries:
            row_vectors[row_name][col_idx] = val

    # Build A (equality) and G (inequality)
    A_rows, b_eq = [], []
    G_rows, h_ineq = [], []

    for row_name, sense in row_types.items():
        if row_name == obj_row_name:
            continue

        row_vec = row_vectors[row_name]
        rhs_val = rhs_data.get(row_name, 0.0)
        range_val = range_data.get(row_name, None)

        if range_val is not None:
            # Ranged constraint → convert to two inequalities
            if sense == 'G':
                lb = rhs_val
                ub = rhs_val + abs(range_val)
            elif sense == 'L':
                ub = rhs_val
                lb = rhs_val - abs(range_val)
            elif sense == 'E':
                if range_val > 0:
                    lb = rhs_val
                    ub = rhs_val + range_val
                else:
                    ub = rhs_val
                    lb = rhs_val + range_val
            else:
                raise ValueError(f"Unsupported ranged sense: {sense}")

            G_rows.append(row_vec)
            h_ineq.append(lb)

            G_rows.append(-row_vec)
            h_ineq.append(-ub)

        else:
            if sense == 'E':
                A_rows.append(row_vec)
                b_eq.append(rhs_val)
            elif sense == 'G':
                G_rows.append(row_vec)
                h_ineq.append(rhs_val)
            elif sense == 'L':
                G_rows.append(-row_vec)
                h_ineq.append(-rhs_val)

    # Bounds
    l = []
    u = []
    for var in var_names:
        # lo = bound_data[var].get('lo', -float('inf'))
        lo = bound_data[var].get('lo', 0)
        up = bound_data[var].get('up', float('inf'))
        l.append(lo)
        u.append(up)

    # Convert all to torch
    A_tensor = torch.tensor(np.array(A_rows), dtype=torch.float32, device=device) 
    b_tensor = torch.tensor(np.array(b_eq), dtype=torch.float32, device=device).view(-1, 1) 

    G_tensor = torch.tensor(np.array(G_rows), dtype=torch.float32, device=device) 
    h_tensor = torch.tensor(np.array(h_ineq), dtype=torch.float32, device=device).view(-1, 1) 

    c_tensor = torch.tensor(c, dtype=torch.float32, device=device).view(-1, 1)

    l_tensor = torch.tensor(l, dtype=torch.float32, device=device).view(-1, 1)
    u_tensor = torch.tensor(u, dtype=torch.float32, device=device).view(-1, 1)

    return c_tensor, G_tensor, h_tensor, A_tensor, b_tensor, l_tensor, u_tensor
