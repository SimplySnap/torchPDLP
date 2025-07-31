import argparse
import torch
import os
from util import mps_to_standard_form, Timer
from pre_post_primal_dual_hyrid_gradient import ruiz_precondition

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
            c, G, h, A, b, l, u = mps_to_standard_form(mps_file_path, device=device)
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
                # PRECONDITION: Perform scaling entirely on GPU
                K_s, c_s, q_s, l_s, u_s, D_col, m_ineq = ruiz_precondition(c, G, h, A, b, l, u, device = device)
                
            time_elapsed = t.elapsed
            print(f"Preconditioning took {time_elapsed:.4f} seconds.")
        except Exception as e:
            print(f"Solver failed for {mps_file}. Error: {e}")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # --- Call your solver/main logic here ---
    print(f"Instance path: {mps_folder_path}")
    print(f"Tolerance: {tol}")
    print(f"Output path: {output_path}")