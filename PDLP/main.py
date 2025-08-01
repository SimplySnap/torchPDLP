import argparse
import torch
import os
from util import mps_to_standard_form, Timer
from enhancements import ruiz_precondition
from PDLP.primal_dual_hybrid_gradient import pdlp_algorithm

def parse_args():
    parser = argparse.ArgumentParser(description='Run LP solver with configuration options.')

    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'auto'], default='auto',
                        help="Device to run on: 'cpu', 'gpu', or 'auto' (default: auto)")
    parser.add_argument('--instance_path', type=str, default='feasible',
                        help="Path to folder containing MPS instances (default: 'feasible')")
    parser.add_argument('--tolerance', type=float, default=1e-2,
                        help="Tolerance for stopping criterion (default: 1e-2)")
    parser.add_argument('--output_path', type=str, default='output',
                        help="Directory where outputs will be saved (default: 'output')")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # --- Device Selection ---
    if args.device == 'auto' or args.device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"PyTorch is using ROCm/CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("ROCm/CUDA not available. PyTorch is using CPU.")
    else:
        device = torch.device(args.device)
        print(f"PyTorch is using device: {device}")

    # --- Configuration ---
    mps_folder_path = args.instance_path
    tol = args.tolerance
    output_path = args.output_path
    results = []
    
    # --- Get all MPS files from the folder ---
    mps_files = sorted([f for f in os.listdir(mps_folder_path) if f.endswith('.mps')])

    for mps_file in mps_files:
        mps_file_path = os.path.join(mps_folder_path, mps_file)
        print(f"\nProcessing {mps_file_path}...")
        try:
            # --- Load problem ---
            c, K, q, m_ineq, l, u= mps_to_standard_form(mps_file_path, device=device)
        except Exception as e:
            print(f"Failed to load MPS file: {mps_file_path}. Error: {e}")
            results.append({
                'File': mps_file,
                'Objective': 'N/A',
                'Iterations (k)': 'N/A',
                'Time (s)': 'N/A',
                'Status': f'Failed to load: {e}'
            })
            continue
        
        try:
            with Timer("Solve time") as t:
                # PRECONDITION
                K_s, c_s, q_s, l_s, u_s, D_col, D_row, old_vars = ruiz_precondition(c, K, q, l, u, device = device)
                #Wrap preconditioned variables for termination criteria checks
                precond_vars = (D_col, D_row, old_vars)
                x, prim_obj, k, n, j = pdlp_algorithm(K_s, m_ineq, c_s, q_s, l_s, u_s,precond_vars ,device, max_iter=100_000, tol=tol, verbose=True, restart_period=40, primal_update=True)
                
                # POSTPROCESSING
                x_final = x * D_col #Undo preconditioning
                obj_final = (c.T @ x_final).item()
                
                print(f"Objective value: {obj_final:.4f}")
                
            time_elapsed = t.elapsed
            print(f"Took {time_elapsed:.4f} seconds.")
        except Exception as e:
            print(f"Solver failed for {mps_file}. Error: {e}")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # --- Call your solver/main logic here ---
    print(f"Instance path: {mps_folder_path}")
    print(f"Tolerance: {tol}")
    print(f"Output path: {output_path}")