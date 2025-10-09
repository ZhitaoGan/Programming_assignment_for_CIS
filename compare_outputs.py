#!/usr/bin/env python3
"""
Script to compare generated outputs with expected outputs for debug datasets.
"""

import numpy as np
import os
from pathlib import Path

def compare_output_files(generated_file, expected_file, tolerance=0.1):
    """Compare two output files, handling the header line properly."""
    try:
        # Read files
        with open(generated_file, 'r') as f:
            gen_lines = f.readlines()
        with open(expected_file, 'r') as f:
            exp_lines = f.readlines()
        
        # Skip header lines (first line)
        gen_data_lines = gen_lines[1:]
        exp_data_lines = exp_lines[1:]
        
        # Convert to numpy arrays
        gen_data = np.array([list(map(float, line.strip().split(','))) for line in gen_data_lines])
        exp_data = np.array([list(map(float, line.strip().split(','))) for line in exp_data_lines])
        
        if gen_data.shape != exp_data.shape:
            print(f"Shape mismatch: {gen_data.shape} vs {exp_data.shape}")
            return False
            
        diff = np.abs(gen_data - exp_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"File: {os.path.basename(generated_file)}")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Tolerance: {tolerance}")
        
        if max_diff <= tolerance:
            print(f" Files match within tolerance")
            return True
        else:
            print(f" Files differ beyond tolerance")
            return False
            
    except Exception as e:
        print(f"Error comparing files: {e}")
        return False

def compare_pivot_points_only(generated_file, expected_file, tolerance=0.05):
    """Compare only the pivot points (lines 2 and 3) from output files."""
    try:
        # Read files
        with open(generated_file, 'r') as f:
            gen_lines = f.readlines()
        with open(expected_file, 'r') as f:
            exp_lines = f.readlines()
        
        # Extract pivot points (lines 2 and 3, index 1 and 2)
        gen_em_pivot = np.array([float(x) for x in gen_lines[1].strip().split(',')])
        gen_opt_pivot = np.array([float(x) for x in gen_lines[2].strip().split(',')])
        
        exp_em_pivot = np.array([float(x) for x in exp_lines[1].strip().split(',')])
        exp_opt_pivot = np.array([float(x) for x in exp_lines[2].strip().split(',')])
        
        # Calculate differences
        em_diff = np.abs(exp_em_pivot - gen_em_pivot)
        opt_diff = np.abs(exp_opt_pivot - gen_opt_pivot)
        
        em_max_diff = np.max(em_diff)
        opt_max_diff = np.max(opt_diff)
        
        print(f"File: {os.path.basename(generated_file)}")
        print(f"  EM Pivot:")
        print(f"    Generated:  {gen_em_pivot}")
        print(f"    Expected:   {exp_em_pivot}")
        print(f"    Difference: {em_diff}")
        print(f"    Max diff:   {em_max_diff:.6f}")
        
        print(f"  Optical Pivot:")
        print(f"    Generated:  {gen_opt_pivot}")
        print(f"    Expected:   {exp_opt_pivot}")
        print(f"    Difference: {opt_diff}")
        print(f"    Max diff:   {opt_max_diff:.6f}")
        
        # Check if both pivot points are within tolerance
        em_match = em_max_diff <= tolerance
        opt_match = opt_max_diff <= tolerance
        
        if em_match and opt_match:
            print(f" Both pivot points match within tolerance ({tolerance})")
            return True
        else:
            print(f" Pivot points differ beyond tolerance ({tolerance})")
            return False
            
    except Exception as e:
        print(f"Error comparing pivot points: {e}")
        return False

def main():
    """Compare all debug dataset outputs."""
    print("Comparing Generated vs Expected Outputs")
    print("="*60)
    
    debug_datasets = ['pa1-debug-a', 'pa1-debug-b', 'pa1-debug-c', 'pa1-debug-d', 
                     'pa1-debug-e', 'pa1-debug-f', 'pa1-debug-g']
    
    data_dir = "PA 1 Student Data"
    output_dir = "output"
    
    # First, compare pivot points only (most important for calibration)
    print("\n" + "="*60)
    print("PIVOT POINTS COMPARISON (Most Important)")
    print("="*60)
    
    pivot_matches = 0
    pivot_total = 0
    
    for dataset in debug_datasets:
        print(f"\n{dataset.upper()}:")
        print("-" * 40)
        
        generated_file = f"{output_dir}/{dataset}-output1.txt"
        expected_file = f"{data_dir}/{dataset}-output1.txt"
        
        if os.path.exists(generated_file) and os.path.exists(expected_file):
            pivot_total += 1
            if compare_pivot_points_only(generated_file, expected_file, tolerance=0.05):
                pivot_matches += 1
        else:
            print(f"Missing files for {dataset}")
            if not os.path.exists(generated_file):
                print(f"  Generated file missing: {generated_file}")
            if not os.path.exists(expected_file):
                print(f"  Expected file missing: {expected_file}")
    
    print(f"\n{'='*60}")
    print(f"PIVOT POINTS SUMMARY:")
    print(f"  Matches: {pivot_matches}/{pivot_total}")
    print(f"  Success rate: {pivot_matches/pivot_total*100:.1f}%" if pivot_total > 0 else "  No files to compare")
    
    # Then, compare complete files (including C coordinates)
    print("\n" + "="*60)
    print("COMPLETE FILES COMPARISON (Including C Coordinates)")
    print("="*60)
    
    complete_matches = 0
    complete_total = 0
    
    for dataset in debug_datasets:
        print(f"\n{dataset.upper()}:")
        print("-" * 40)
        
        generated_file = f"{output_dir}/{dataset}-output1.txt"
        expected_file = f"{data_dir}/{dataset}-output1.txt"
        
        if os.path.exists(generated_file) and os.path.exists(expected_file):
            complete_total += 1
            if compare_output_files(generated_file, expected_file, tolerance=0.1):
                complete_matches += 1
        else:
            print(f"Missing files for {dataset}")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE FILES SUMMARY:")
    print(f"  Matches: {complete_matches}/{complete_total}")
    print(f"  Success rate: {complete_matches/complete_total*100:.1f}%" if complete_total > 0 else "  No files to compare")
    
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY:")
    print(f"  Pivot Points: {pivot_matches}/{pivot_total} ({pivot_matches/pivot_total*100:.1f}%)" if pivot_total > 0 else "  No pivot data")
    print(f"  Complete Files: {complete_matches}/{complete_total} ({complete_matches/complete_total*100:.1f}%)" if complete_total > 0 else "  No complete data")
    print(f"\nNote: Pivot points are the most important for calibration accuracy.")
    print(f"Large differences in C coordinates may indicate a separate calculation issue.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--pivot-only":
            # Only compare pivot points
            print("Comparing Pivot Points Only")
            print("="*60)
            
            debug_datasets = ['pa1-debug-a', 'pa1-debug-b', 'pa1-debug-c', 'pa1-debug-d', 
                             'pa1-debug-e', 'pa1-debug-f', 'pa1-debug-g']
            
            data_dir = "PA 1 Student Data"
            output_dir = "output"
            
            matches = 0
            total = 0
            
            for dataset in debug_datasets:
                print(f"\n{dataset.upper()}:")
                print("-" * 40)
                
                generated_file = f"{output_dir}/{dataset}-output1.txt"
                expected_file = f"{data_dir}/{dataset}-output1.txt"
                
                if os.path.exists(generated_file) and os.path.exists(expected_file):
                    total += 1
                    if compare_pivot_points_only(generated_file, expected_file, tolerance=0.05):
                        matches += 1
                else:
                    print(f"Missing files for {dataset}")
            
            print(f"\n{'='*60}")
            print(f"PIVOT POINTS SUMMARY:")
            print(f"  Matches: {matches}/{total}")
            print(f"  Success rate: {matches/total*100:.1f}%" if total > 0 else "  No files to compare")
            
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python compare_outputs.py              # Full comparison (pivot + complete)")
            print("  python compare_outputs.py --pivot-only # Only compare pivot points")
            print("  python compare_outputs.py --help       # Show this help")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        main()
