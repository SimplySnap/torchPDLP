def generate_feasible_lp(num_vars=100, num_ineq=200, num_eq=50, density=0.05, mps_filename="generated_lp.mps"):
  """
  Generates large feasible LP problems to test and saves them in the .mps format.
  The number of variables in each constraint with nonzero coefficients will be roughly density * num_vars

  Args:
    num_vars: number of variables in each constraint
    num_ineq: number of inequality constraints
    num_eq: number of equality constraints
    density: density of nonzero coefficients in each constraint
    mps_filename: name of the .mps file to save the LP to
  """
    
  rng = np.random.default_rng(0)

  # Step 1: Feasible solution
  x_feas = rng.uniform(low=-10, high=10, size=(num_vars, 1))
  
  # Step 2: Sparse matrices (convert to dense)
  G_sparse = sparse_random(num_ineq, num_vars, density=density, format='csr', random_state=None)
  A_sparse = sparse_random(num_eq, num_vars, density=density, format='csr', random_state=None)

  G = G_sparse.toarray()
  A = A_sparse.toarray()

  # Step 3: RHS vectors
  h = G @ x_feas + rng.uniform(0.1, 5.0, size=(num_ineq, 1))
  b = A @ x_feas

  # Step 4: Bounds
  l = x_feas - rng.uniform(1, 5, size=(num_vars, 1))
  u = x_feas + rng.uniform(1, 5, size=(num_vars, 1))
  l = np.maximum(l, -1e4)
  u = np.minimum(u, 1e4)

  # Step 5: Objective
  c = rng.normal(size=(num_vars, 1))

  # Step 6: Write to MPS using pulp
  prob = pulp.LpProblem("Feasible_LP", pulp.LpMinimize)
  x_vars = [
      pulp.LpVariable(f"x{i}", lowBound=float(l[i]), upBound=float(u[i]))
      for i in range(num_vars)
  ]
  prob += pulp.lpDot(c.flatten(), x_vars)

  # Inequality constraints: Gx ≥ h
  for i in range(num_ineq):
      prob += pulp.lpDot(G[i], x_vars) <= float(h[i]), f"ineq_{i}"

  # Equality constraints: Ax = b
  for i in range(num_eq):
      prob += pulp.lpDot(A[i], x_vars) == float(b[i]), f"eq_{i}"

  prob.writeMPS(mps_filename)
  print(f"Done\n LP written to: {mps_filename}")
