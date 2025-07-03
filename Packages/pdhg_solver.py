def pdhg_solver(mps_file_path, max_iter=10000, tol=1e-4, term_period=1000, verbose=True):
  """
    Full PDHG solver implementation using PyTorch.

    Args:
      mps_file_path (str): Path to the MPS file.
      max_iter (int, optional): Maximum number of iterations. Defaults to 10000.
      tol (float, optional): Tolerance for convergence. Defaults to 1e-4. Use 1e-8 for high accuracy
      term_period (int, optional): Period for termination checks. Defaults to 1000.
      verbose (bool, optional): Whether to print termination check information. Defaults to True.

    Returns:
      The minimizer, objective value, and number of iterations for convergence.
    """
    # --- Device Selection ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"PyTorch is using ROCm/CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("ROCm/CUDA not available. PyTorch is using CPU.")

    # --- Parameter Loading ---
    try:
          c, G, h, A, b, l, u = mps_to_standard_form_torch(mps_file_path, device=device)
    except Exception as e:
        print(f"Failed to load MPS file: {e}")
        exit(1)

    is_neg_inf = torch.isinf(l) & (l < 0)
    is_pos_inf = torch.isinf(u) & (u > 0)

    l_dual = l.clone()
    u_dual = u.clone()

    l_dual[is_neg_inf] = 0
    u_dual[is_pos_inf] = 0

    # --- Run PDHG Solver on the GPU or CPU ---
    minimizer, obj_val, iterations = pdhg_torch(c, G, h, A, b, l, u, is_neg_inf, is_pos_inf, l_dual, u_dual, max_iter=max_iter, tol=tol, device=device, verbose=verbose, term_period=term_period)

    print("Objective Value:", obj_val)
    print("Iterations:", iterations)
    print("\nMinimizer (first 10 variables):")
    print(minimizer[:10].cpu().numpy())

    return minimizer, obj_val, iterations
