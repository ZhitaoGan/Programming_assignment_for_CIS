#!/usr/bin/env python3
"""
Comprehensive script to process all PA1 datasets.

This script processes all debug datasets (a-g) and unknown datasets (h-k) to generate:
1. Fa and Fd registrations for each dataset
2. EM pivot calibration results
3. Optical pivot calibration results  
4. Final output1 files for debug datasets (comparing with expected results)
5. Final output1 files for unknown datasets

Usage: python process_all_datasets.py
"""

import os
import subprocess
import sys
from pathlib import Path
import numpy as np

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/Users/zhitaogan/Desktop/JHU/Computer Integrated Surgery 1/CISPA")
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return False
        else:
            print(f"SUCCESS: {description}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return True
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False

def compare_files(file1, file2, tolerance=1e-3):
    """Compare two files with given tolerance."""
    try:
        data1 = np.loadtxt(file1, delimiter=',')
        data2 = np.loadtxt(file2, delimiter=',')
        
        if data1.shape != data2.shape:
            print(f"Shape mismatch: {data1.shape} vs {data2.shape}")
            return False
            
        diff = np.abs(data1 - data2)
        max_diff = np.max(diff)
        
        if max_diff <= tolerance:
            print(f"Files match within tolerance {tolerance} (max diff: {max_diff:.6f})")
            return True
        else:
            print(f"Files differ beyond tolerance {tolerance} (max diff: {max_diff:.6f})")
            return False
    except Exception as e:
        print(f"Error comparing files: {e}")
        return False

def process_dataset(dataset_name, is_debug=True):
    """Process a single dataset."""
    print(f"\n{'#'*80}")
    print(f"PROCESSING DATASET: {dataset_name}")
    print(f"{'#'*80}")
    
    data_dir = "PA 1 Student Data"
    output_dir = "output"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    total_operations = 0
    
    # 1. Generate Fa registration
    total_operations += 1
    cmd = f"python pa1.py --data_dir '{data_dir}' --output_dir '{output_dir}' --name {dataset_name}-calbody --name_2 {dataset_name}-calreadings --output_file Fa_{dataset_name}_registration"
    if run_command(cmd, f"Generate Fa registration for {dataset_name}"):
        success_count += 1
    
    # 2. Generate Fd registration  
    total_operations += 1
    cmd = f"python pa1.py --data_dir '{data_dir}' --output_dir '{output_dir}' --name {dataset_name}-calbody --name_2 {dataset_name}-calreadings --output_file Fd_{dataset_name}_registration"
    if run_command(cmd, f"Generate Fd registration for {dataset_name}"):
        success_count += 1
    
    # 3. Generate EM pivot calibration
    total_operations += 1
    cmd = f"python pa1.py --data_dir '{data_dir}' --output_dir '{output_dir}' --name_3 {dataset_name}-empivot --output_file1 {dataset_name.upper()}_EM_pivot"
    if run_command(cmd, f"Generate EM pivot calibration for {dataset_name}"):
        success_count += 1
    
    # 4. Generate Optical pivot calibration
    total_operations += 1
    cmd = f"python pa1.py --data_dir '{data_dir}' --output_dir '{output_dir}' --name {dataset_name}-calbody --name_4 {dataset_name}-optpivot --output_file2 {dataset_name.upper()}_Optpivot"
    if run_command(cmd, f"Generate Optical pivot calibration for {dataset_name}"):
        success_count += 1
    
    # 5. Generate final output1 file (only for debug datasets)
    if is_debug:
        total_operations += 1
        cmd = f"python pa1.py --data_dir '{data_dir}' --output_dir '{output_dir}' --name {dataset_name}-calbody --input_reg Fa_{dataset_name}_registration --input_reg2 Fd_{dataset_name}_registration --output_file {dataset_name}-output1"
        if run_command(cmd, f"Generate final output1 for {dataset_name}"):
            success_count += 1
            
            # Compare with expected output if it exists
            expected_file = f"{data_dir}/{dataset_name}-output1.txt"
            generated_file = f"{output_dir}/{dataset_name}-output1.txt"
            
            if os.path.exists(expected_file) and os.path.exists(generated_file):
                print(f"\nComparing generated output with expected output for {dataset_name}:")
                compare_files(generated_file, expected_file)
    
    print(f"\nDataset {dataset_name} completed: {success_count}/{total_operations} operations successful")
    return success_count, total_operations

def main():
    """Main function to process all datasets."""
    print("PA1 Dataset Processing Script")
    print("="*80)
    
    # Debug datasets (a-g)
    debug_datasets = ['pa1-debug-a', 'pa1-debug-b', 'pa1-debug-c', 'pa1-debug-d', 
                     'pa1-debug-e', 'pa1-debug-f', 'pa1-debug-g']
    
    # Unknown datasets (h-k)  
    unknown_datasets = ['pa1-unknown-h', 'pa1-unknown-i', 'pa1-unknown-j', 'pa1-unknown-k']
    
    total_success = 0
    total_operations = 0
    
    # Process debug datasets
    print("\n" + "="*80)
    print("PROCESSING DEBUG DATASETS (a-g)")
    print("="*80)
    
    for dataset in debug_datasets:
        success, ops = process_dataset(dataset, is_debug=True)
        total_success += success
        total_operations += ops
    
    # Process unknown datasets
    print("\n" + "="*80)
    print("PROCESSING UNKNOWN DATASETS (h-k)")
    print("="*80)
    
    for dataset in unknown_datasets:
        success, ops = process_dataset(dataset, is_debug=False)
        total_success += success
        total_operations += ops
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Total operations: {total_operations}")
    print(f"Successful operations: {total_success}")
    print(f"Success rate: {total_success/total_operations*100:.1f}%")
    
    if total_success == total_operations:
        print("🎉 All datasets processed successfully!")
    else:
        print(f"⚠️  {total_operations - total_success} operations failed. Check the output above for details.")
    
    # List generated files
    print(f"\nGenerated files in 'output' directory:")
    output_path = Path("output")
    if output_path.exists():
        for file in sorted(output_path.glob("*.txt")):
            print(f"  - {file.name}")

if __name__ == "__main__":
    main()
