#!/usr/bin/env python3
"""
Run PA2 pipeline for all debug and unknown datasets
"""

import numpy as np
from programs import utility_functions

# PA2 datasets
debug_datasets = ['pa2-debug-a', 'pa2-debug-b', 'pa2-debug-c',
                  'pa2-debug-d', 'pa2-debug-e', 'pa2-debug-f']
unknown_datasets = ['pa2-unknown-g', 'pa2-unknown-h',
                    'pa2-unknown-i', 'pa2-unknown-j']

data_dir = "PA 2 Student Data"
output_dir = "output"

print("=" * 80)
print("PA2 PIPELINE - GENERATING OUTPUT2 FILES FOR ALL DATASETS")
print("=" * 80)

# First, need to generate output1 files for unknown datasets
print("\n" + "=" * 80)
print("STEP 1: GENERATING OUTPUT1 FILES FOR UNKNOWN DATASETS")
print("=" * 80)

for dataset in unknown_datasets:
    print(f"\nProcessing {dataset}...")

    # Define file paths
    calbody_file = f"{data_dir}/{dataset}-calbody.txt"
    calreadings_file = f"{data_dir}/{dataset}-calreadings.txt"
    empivot_file = f"{data_dir}/{dataset}-empivot.txt"
    optpivot_file = f"{data_dir}/{dataset}-optpivot.txt"
    output1_file = f"{dataset}-output1"

    # Read calibration body
    cal_body = utility_functions.read_cal_data(calbody_file)

    # Read calibration readings
    cal_readings = utility_functions.read_cal_data(calreadings_file)

    # Read pivot data
    empivot = utility_functions.read_pivot_data(empivot_file)
    optpivot = utility_functions.read_optpivot(optpivot_file)

    # Generate Fa and Fd files
    print(f"  Generating frame transformations...")
    utility_functions.parse_files(f"{dataset}_Fa", cal_readings, cal_body, output_dir)
    utility_functions.parse_files(f"{dataset}_Fd", cal_readings, cal_body, output_dir)

    # Perform pivot calibrations
    print(f"  Performing pivot calibrations...")
    utility_functions.em_pivot(empivot, f"{dataset}_EM_pivot", output_dir)
    utility_functions.opt_pivot(optpivot, cal_body, f"{dataset}_Optpivot", output_dir)

    # Generate output1 file
    print(f"  Generating output1 file...")
    utility_functions.C_expected(
        cal_body,
        output1_file,
        f"{dataset}_Fa",
        f"{dataset}_Fd",
        output_dir
    )
    print(f"  ✓ {output1_file}.txt generated")

print("\n" + "=" * 80)
print("STEP 2: GENERATING OUTPUT2 FILES FOR ALL DATASETS")
print("=" * 80)

all_datasets = debug_datasets + unknown_datasets

for dataset in all_datasets:
    print(f"\n{'=' * 80}")
    print(f"Processing: {dataset}")
    print('=' * 80)

    # Define file paths
    empivot_file = f"{data_dir}/{dataset}-empivot.txt"
    em_fiducials_file = f"{data_dir}/{dataset}-em-fiducialss.txt"
    ct_fiducials_file = f"{data_dir}/{dataset}-ct-fiducials.txt"
    em_nav_file = f"{data_dir}/{dataset}-EM-nav.txt"
    calreadings_file = f"{data_dir}/{dataset}-calreadings.txt"

    # For unknown datasets, use generated output1 from output dir
    # For debug datasets, use provided output1 from data dir
    if 'unknown' in dataset:
        output1_file = f"{output_dir}/{dataset}-output1.txt"
    else:
        output1_file = f"{data_dir}/{dataset}-output1.txt"

    output_file = f"{dataset}-output2"

    # Run PA2 pipeline
    utility_functions.pa2_output2(
        empivot_file,
        em_fiducials_file,
        ct_fiducials_file,
        em_nav_file,
        calreadings_file,
        output1_file,
        output_file,
        output_dir
    )

    print(f"✓ Generated: {output_dir}/{output_file}.txt")

    # Compare with expected output if it exists (debug datasets only)
    if 'debug' in dataset:
        expected_file = f"{data_dir}/{dataset}-output2.txt"
        generated_file = f"{output_dir}/{output_file}.txt"

        try:
            generated = np.loadtxt(generated_file, delimiter=',', skiprows=1)
            expected = np.loadtxt(expected_file, delimiter=',', skiprows=1)

            if generated.shape == expected.shape:
                differences = np.abs(generated - expected)
                mean_diff = np.mean(np.linalg.norm(differences, axis=1))
                max_diff = np.max(np.linalg.norm(differences, axis=1))

                print(f"  Validation: Mean error = {mean_diff:.4f} mm, Max error = {max_diff:.4f} mm")
            else:
                print(f"  Warning: Shape mismatch {generated.shape} vs {expected.shape}")
        except Exception as e:
            print(f"  Could not validate: {e}")

print("\n" + "=" * 80)
print("ALL DATASETS PROCESSED SUCCESSFULLY!")
print("=" * 80)
print(f"\nGenerated files in '{output_dir}/' directory:")
print(f"  - Debug datasets (a-f): 6 output2 files with validation")
print(f"  - Unknown datasets (g-j): 4 output1 + 4 output2 files")
print("=" * 80)
