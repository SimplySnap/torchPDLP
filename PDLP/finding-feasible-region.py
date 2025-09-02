import os
import pandas as pd
import torch
import numpy as np
from typing import List, Tuple, Optional, Union
from util import mps_to_standard_form

def count_bound_stats(bound: Optional[torch.Tensor]) -> Tuple[int, int, int]:
    """
    Count positive, negative, and infinite entries in PyTorch tensor bounds.
    
    Args:
        bound: PyTorch tensor containing bounds (can be None)
        
    Returns:
        (positive_count, negative_count, infinite_count)
    """
    if bound is None:
        return 0, 0, 0
    
    # Convert to numpy for easier analysis if needed
    bound_np = bound.detach().numpy() if bound.requires_grad else bound.numpy()
    
    # Count finite positive values
    pos = torch.sum((bound > 0) & torch.isfinite(bound)).item()
    
    # Count finite negative values  
    neg = torch.sum((bound < 0) & torch.isfinite(bound)).item()
    
    # Count infinite values (both positive and negative infinity)
    infs = torch.sum(torch.isinf(bound)).item()
    
    return pos, neg, infs

def analyze_mps_bounds():
    """Analyze bounds from MPS files and generate statistics DataFrame"""
    
    # Load MPS files
    mps_folder_path = 'datasets/netlib/feasible'
    mps_files = sorted([f for f in os.listdir(mps_folder_path) if f.endswith('.mps')])
    
    # Initialize dataframe columns
    columns = ['file', 'l_pos', 'l_neg', 'l_infs', 'u_pos', 'u_neg', 'u_infs', 
               'l_total', 'u_total', 'l_finite', 'u_finite']
    data = []
    
    print(f"Processing {len(mps_files)} MPS files...")
    
    # Process each MPS file
    for i, file in enumerate(mps_files):
        try:
            full_path = os.path.join(mps_folder_path, file)
            c, K, q, m_ineq, l, u = mps_to_standard_form(full_path, support_sparse=False, verbose=False)
            
            # Count statistics for lower bounds (l)
            l_pos, l_neg, l_infs = count_bound_stats(l)
            
            # Count statistics for upper bounds (u)
            u_pos, u_neg, u_infs = count_bound_stats(u)
            
            # Calculate totals
            l_total = l.numel() if l is not None else 0
            u_total = u.numel() if u is not None else 0
            l_finite = l_pos + l_neg
            u_finite = u_pos + u_neg
            
            # Append row for this file
            data.append([file, l_pos, l_neg, l_infs, u_pos, u_neg, u_infs, 
                        l_total, u_total, l_finite, u_finite])
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(mps_files)} files")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Create DataFrame
    stats_df = pd.DataFrame(data, columns=columns)
    
    # Calculate totals across all files
    totals = {'file': 'TOTAL_ALL_FILES'}
    for col in columns[1:]:  # Skip 'file' column
        totals[col] = stats_df[col].sum()
    
    # Add totals row
    totals_row = pd.DataFrame([totals])
    stats_df = pd.concat([stats_df, totals_row], ignore_index=True)
    
    return stats_df

def print_analysis_summary(stats_df: pd.DataFrame):
    """Print detailed analysis summary"""
    
    totals = stats_df.iloc[-1]  # Last row contains totals
    n_files = len(stats_df) - 1  # Exclude totals row
    
    print("\n" + "="*60)
    print("MPS BOUNDS ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total MPS files processed: {n_files}")
    print(f"Total variables with lower bounds: {totals['l_total']}")
    print(f"Total variables with upper bounds: {totals['u_total']}")
    
    print(f"\nLOWER BOUNDS (l):")
    print(f"  Positive: {totals['l_pos']} ({100*totals['l_pos']/max(totals['l_total'],1):.1f}%)")
    print(f"  Negative: {totals['l_neg']} ({100*totals['l_neg']/max(totals['l_total'],1):.1f}%)")
    print(f"  Infinite: {totals['l_infs']} ({100*totals['l_infs']/max(totals['l_total'],1):.1f}%)")
    
    print(f"\nUPPER BOUNDS (u):")
    print(f"  Positive: {totals['u_pos']} ({100*totals['u_pos']/max(totals['u_total'],1):.1f}%)")
    print(f"  Negative: {totals['u_neg']} ({100*totals['u_neg']/max(totals['u_total'],1):.1f}%)")
    print(f"  Infinite: {totals['u_infs']} ({100*totals['u_infs']/max(totals['u_total'],1):.1f}%)")
    
    total_bounds = totals['l_total'] + totals['u_total']
    total_infs = totals['l_infs'] + totals['u_infs']
    print(f"\nOVERALL:")
    print(f"  Total bound entries: {total_bounds}")
    print(f"  Percentage infinite: {100*total_infs/max(total_bounds,1):.1f}%")

# Main execution
if __name__ == "__main__":
    # Run analysis
    results_df = analyze_mps_bounds()
    
    # Display results
    print(results_df.to_string(index=False))
    
    # Print summary
    print_analysis_summary(results_df)
    
    # Save to CSV
    results_df.to_csv('experiments/mps_bounds_analysis.csv', index=False)
    print(f"\nResults saved to 'mps_bounds_analysis.csv'")
