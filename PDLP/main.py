import argparse
import torch
import os

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

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # --- Call your solver/main logic here ---
    print(f"Instance path: {mps_folder_path}")
    print(f"Tolerance: {tol}")
    print(f"Output path: {output_path}")