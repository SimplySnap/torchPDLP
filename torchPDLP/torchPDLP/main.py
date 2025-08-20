import argparse
import torch
import os
import pandas as pd
from .spectral_casting import spectral_cast

import time
from .util import mps_to_standard_form
from .enhancements import preconditioning
from .primal_dual_hybrid_gradient import pdlp_algorithm

def parse_args():
    parser = argparse.ArgumentParser(description='Run LP solver with configuration options.')

    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'auto'], default='auto',
                        help="Device to run on: 'cpu', 'gpu', or 'auto' (default: auto)")
    parser.add_argument('--instance_path', type=str, default='feasible',
                        help="Path to folder containing MPS instances (default: 'feasible')")
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help="Tolerance for stopping criterion (default: 1e-4)")
    parser.add_argument('--output_path', type=str, default='output',
                        help="Directory where outputs will be saved (default: 'output')")
    parser.add_argument('--precondition', action='store_true',
                        help="Enable Ruiz preconditioning (default: False)")
    parser.add_argument('--primal_weight_update', action='store_true',
                        help="Enable primal weight update (default: False)")
    parser.add_argument('--adaptive_stepsize', action='store_true',
                        help="Enable adaptive stepsize for PDLP (default: False)")
    parser.add_argument('--infeasibility_detect', action='store_true',
                        help="Enable infeasibility detection (default: False)")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose output (default: False)")
    parser.add_argument('--support_sparse', action='store_true',
                        help="Support sparse matrices operations(default: False)")
    parser.add_argument('--max_kkt', type=int, default=None,
                        help="Maximum number of KKT passes (default: None)")
    parser.add_argument('--time_limit', type=int, default=None,
                        help="Time limit for the solver in seconds (default: None)")
    parser.add_argument('--fishnet', action='store_true',help="Use fishnet alg for better startpoint",default=False)
    return parser.parse_args()

def main():
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
    precondition = args.precondition
    primal_weight_update = args.primal_weight_update
    adaptive_stepsize = args.adaptive_stepsize
    infeasibility_detect = args.infeasibility_detect
    verbose=args.verbose
    support_sparse = args.support_sparse
    max_kkt = args.max_kkt
    time_limit = args.time_limit
    
    print(f"\nConfiguration:")
    print(f"Instance path: {mps_folder_path}")
    print(f"Tolerance: {tol}")
    print(f"Output path: {output_path}")
    print(f"Time limit: {time_limit} seconds")
    print(f"Max KKT passes: {max_kkt}")
    print(f"Preconditioning: {precondition}")
    print(f"Primal weight update: {primal_weight_update}")
    print(f"Adaptive stepsize: {adaptive_stepsize}")
    print(f"Infeasibility detection: {infeasibility_detect}")
    
    results = []
    
    # --- Get all MPS files from the folder ---
    mps_files = sorted([f for f in os.listdir(mps_folder_path) if f.endswith('.mps')])

    for mps_file in mps_files:
        mps_file_path = os.path.join(mps_folder_path, mps_file)
        print(f"\nProcessing {mps_file_path}...")
        try:
            # --- Load problem ---
            c, K, q, m_ineq, l, u= mps_to_standard_form(mps_file_path, device=device, support_sparse=support_sparse, verbose=verbose)
        except Exception as e:
            print(f"Failed to load MPS file: {mps_file_path}. Error: {e}")
            results.append({
                'File': mps_file,
                'Objective': 'N/A',
                'Iterations (k)': 'N/A',
                'Restarts (n)': 'N/A',
                'KKT Passes (j)': 'N/A',
                'Time (s)': 'N/A',
                'Status': f'Failed to load: {str(e)[:50]}...' if len(str(e)) > 50 else str(e)
            })
            continue
        
        try:
            # PRECONDITION
            if precondition:
                K, c, q, l, u, dt_precond, time_used= preconditioning(c, K, q, l, u, device = device)
            else:
                time_used = 0.0
                dt_precond = None
            
            #  --- Fishnet initialization --- 
            #  FISHNET starting point optimimzation
            if args.fishnet:
                fishnet_time = 0.0 #  For time tracking
                fishnet_start = time.time()
                x0, y0 = spectral_cast(
                K, c, q, l, u, m_ineq, k=32,  # your choice of hyperparameters
                device=device
            )
            #  Record fishnet speedup
                fishnet_time = time.time() - fishnet_start #  Get the time taken by fishnet
                time_used += fishnet_time  # Add to cumulative preprocessing time
                if verbose:
                    print(f"Fishnet completed in {fishnet_time:.4f}s")
            else:
                x0, y0 = None, None

            x, prim_obj, k, n, j, status, total_time = pdlp_algorithm(
                K, m_ineq, c, q, l, u, device, 
                max_kkt=max_kkt, tol=tol, verbose=verbose, restart_period=40, 
                precondition=precondition, primal_update=primal_weight_update, 
                adaptive=adaptive_stepsize, data_precond=dt_precond, 
                infeasibility_detect=infeasibility_detect,
                time_limit=time_limit, time_used=time_used
            )
            print(f"Solver uses {total_time:.4f} seconds.")
            print(f"Status: {status}")
            
            # Store results
            results.append({
                'File': mps_file,
                'Objective': f"{prim_obj:.6f}",
                'Iterations (k)': k,
                'Restarts (n)': n,
                'KKT Passes (j)': j,
                'Time (s)': f"{total_time:.4f}",
                'Status': status
            })
            
        except Exception as e:
            print(f"Solver failed for {mps_file}. Error: {e}")
            results.append({
                'File': mps_file,
                'Objective': 'N/A',
                'Iterations (k)': 'N/A',
                'Restarts (n)': 'N/A',
                'KKT Passes (j)': 'N/A',
                'Time (s)': 'N/A',
                'Status': f'Solver failed: {str(e)[:50]}...' if len(str(e)) > 50 else f'Solver failed: {str(e)}'
            })

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    #  Save results to csv file
    if results:
        df = pd.DataFrame(results)
        csv_filename = os.path.join(output_path, 'solver_results.csv')
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to CSV instead: {csv_filename}")
