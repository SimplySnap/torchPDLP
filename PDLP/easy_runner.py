"""
Simple PDLP Runner with Spectral Casting
========================================
Easy-to-use interface for running PDLP with optional fishnet (spectral casting) warm-start.
Place this file in your main directory alongside main.py.
"""

import os
import torch
import time
from pathlib import Path

# Import your modules
from util import mps_to_standard_form
from enhancements import ruiz_precondition
import spectral_casting as fishnet
from primal_dual_hybrid_gradient import pdlp_algorithm

def select_mps_directory():
    """
    Interactive directory selection for MPS files.
    Returns the path to the selected directory.
    """
    print("Available directories in current folder:")
    current_dir = Path(".")
    dirs = [d for d in current_dir.iterdir() if d.is_dir()]
    
    if not dirs:
        print("No directories found. Using current directory.")
        return "."
    
    # Show numbered list of directories
    for i, directory in enumerate(dirs, 1):
        mps_count = len(list(directory.glob("*.mps")))
        print(f"{i}. {directory.name} ({mps_count} MPS files)")
    
    while True:
        try:
            choice = input(f"\nSelect directory (1-{len(dirs)}) or press Enter for 'feasible': ")
            if choice == "":
                return "feasible" if Path("feasible").exists() else "."
            
            idx = int(choice) - 1
            if 0 <= idx < len(dirs):
                return str(dirs[idx])
            else:
                print(f"Please enter a number between 1 and {len(dirs)}")
        except ValueError:
            print("Please enter a valid number")

def get_device():
    """Auto-detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def run_single_instance(mps_file_path, device, config):
    """
    Run the complete pipeline on a single MPS instance.
    
    Args:
        mps_file_path (str): Path to the MPS file
        device: torch device
        config (dict): Configuration dictionary
    
    Returns:
        dict: Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Processing: {mps_file_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Step 1: Load MPS file
        print("Step 1: Loading MPS file...")
        c, K, q, m_ineq, l, u = mps_to_standard_form(
            mps_file_path, 
            device=device, 
            verbose=config['verbose']
        )
        print(f"  Problem size: {c.shape[0]} variables, {K.shape[0]} constraints")
        
        # Step 2: Preconditioning (optional)
        time_used = 0.0
        dt_precond = None
        if config['precondition']:
            print("Step 2: Applying Ruiz preconditioning...")
            K, c, q, l, u, dt_precond, time_used = ruiz_precondition(
                c, K, q, l, u, device=device
            )
            print(f"  Preconditioning completed in {time_used:.4f}s")
        else:
            print("Step 2: Skipping preconditioning")
        
        # Step 3: Spectral Casting (optional)
        x_init, y_init = None, None
        if config['use_fishnet']:
            print("Step 3: Generating spectral cast starting point...")
            x_init, y_init = fishnet.spectral_cast(
                K, c, q, l, u, m_ineq, 
                k=config['fishnet_iterations'],
                s=config['fishnet_reduction'],
                i=config['fishnet_points_exp'],
                device=device
            )
            print(f"  Generated starting point with {2**config['fishnet_points_exp']} initial samples")
        else:
            print("Step 3: Using zero starting point")
        
        # Step 4: Run PDLP Algorithm
        print("Step 4: Running PDLP algorithm...")
        x, prim_obj, iterations, restarts, kkt_passes, status, total_time = pdlp_algorithm(
            K, m_ineq, c, q, l, u, device,
            max_kkt=config['max_kkt'], 
            tol=config['tolerance'], 
            verbose=config['verbose'],
            restart_period=40,
            precondition=config['precondition'], 
            primal_update=config['primal_update'],
            adaptive=config['adaptive'], 
            data_precond=dt_precond,
            time_limit=config['time_limit'], 
            time_used=time_used,
            x_init=x_init, 
            y_init=y_init
        )
        
        # Results summary
        total_pipeline_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Status: {status}")
        print(f"Objective Value: {prim_obj:.6f}")
        print(f"Iterations: {iterations}")
        print(f"Restarts: {restarts}")
        print(f"KKT Passes: {kkt_passes}")
        print(f"Solver Time: {total_time:.4f}s")
        print(f"Total Pipeline Time: {total_pipeline_time:.4f}s")
        
        return {
            'file': os.path.basename(mps_file_path),
            'status': status,
            'objective': prim_obj,
            'iterations': iterations,
            'restarts': restarts,
            'kkt_passes': kkt_passes,
            'solver_time': total_time,
            'pipeline_time': total_pipeline_time,
            'success': True
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"\nERROR: {str(e)}")
        return {
            'file': os.path.basename(mps_file_path),
            'status': f'Error: {str(e)[:50]}...',
            'objective': 'N/A',
            'iterations': 'N/A',
            'restarts': 'N/A',
            'kkt_passes': 'N/A',
            'solver_time': 'N/A',
            'pipeline_time': error_time,
            'success': False
        }

def configure_solver():
    """
    Interactive configuration of solver parameters.
    Returns a configuration dictionary.
    """
    print("\n" + "="*60)
    print("SOLVER CONFIGURATION")
    print("="*60)
    
    config = {
        'tolerance': 1e-4,
        'max_kkt': 100_000,
        'time_limit': 3600,
        'precondition': True,
        'primal_update': False,
        'adaptive': True,
        'use_fishnet': True,
        'fishnet_iterations': 32,
        'fishnet_reduction': 2,
        'fishnet_points_exp': 5,
        'verbose': True
    }
    
    print("Default configuration loaded. Press Enter to use defaults, or 'c' to customize:")
    
    choice = input().strip().lower()
    if choice == 'c':
        print("\nCustomizing configuration:")
        
        # Basic solver settings
        config['tolerance'] = float(input(f"Tolerance [{config['tolerance']}]: ") or config['tolerance'])
        config['time_limit'] = int(input(f"Time limit (seconds) [{config['time_limit']}]: ") or config['time_limit'])
        config['max_kkt'] = int(input(f"Max KKT passes [{config['max_kkt']}]: ") or config['max_kkt'])
        
        # Enhancement options
        config['precondition'] = input(f"Use preconditioning? [Y/n]: ").lower() != 'n'
        config['adaptive'] = input(f"Use adaptive stepsize? [Y/n]: ").lower() != 'n'
        config['primal_update'] = input(f"Use primal weight update? [y/N]: ").lower() == 'y'
        
        # Fishnet options
        config['use_fishnet'] = input(f"Use fishnet (spectral casting)? [Y/n]: ").lower() != 'n'
        if config['use_fishnet']:
            config['fishnet_iterations'] = int(input(f"Fishnet PDHG iterations [{config['fishnet_iterations']}]: ") or config['fishnet_iterations'])
            config['fishnet_points_exp'] = int(input(f"Fishnet points exponent (2^i) [{config['fishnet_points_exp']}]: ") or config['fishnet_points_exp'])
        
        config['verbose'] = input(f"Verbose output? [Y/n]: ").lower() != 'n'
    
    # Display final configuration
    print(f"\nFinal Configuration:")
    print(f"  Tolerance: {config['tolerance']}")
    print(f"  Time Limit: {config['time_limit']}s")
    print(f"  Preconditioning: {config['precondition']}")
    print(f"  Adaptive Stepsize: {config['adaptive']}")
    print(f"  Fishnet: {config['use_fishnet']}")
    if config['use_fishnet']:
        print(f"    Initial Points: 2^{config['fishnet_points_exp']} = {2**config['fishnet_points_exp']}")
        print(f"    PDHG Iterations: {config['fishnet_iterations']}")
    
    return config

def main():
    """Main function - run the complete pipeline."""
    print("🚀 PDLP Solver with Spectral Casting")
    print("="*60)
    
    # Setup
    device = get_device()
    config = configure_solver()
    
    # Select MPS directory
    mps_dir = select_mps_directory()
    mps_path = Path(mps_dir)
    
    if not mps_path.exists():
        print(f"Error: Directory '{mps_dir}' does not exist!")
        return
    
    # Find MPS files
    mps_files = list(mps_path.glob("*.mps"))
    if not mps_files:
        print(f"No MPS files found in '{mps_dir}'!")
        return
    
    print(f"\nFound {len(mps_files)} MPS file(s) in '{mps_dir}'")
    
    # Process files
    results = []
    for i, mps_file in enumerate(sorted(mps_files), 1):
        print(f"\n[{i}/{len(mps_files)}] Processing {mps_file.name}")
        result = run_single_instance(str(mps_file), device, config)
        results.append(result)
        
        if not result['success']:
            continue_choice = input("\nContinue with remaining files? [Y/n]: ")
            if continue_choice.lower() == 'n':
                break
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_time = sum(r['pipeline_time'] for r in successful) / len(successful)
        print(f"Average pipeline time: {avg_time:.4f}s")
    
    if failed:
        print("\nFailed files:")
        for r in failed:
            print(f"  {r['file']}: {r['status']}")

if __name__ == "__main__":
    main()
