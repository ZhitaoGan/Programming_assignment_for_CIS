"""
PA3 - ICP Matching

Implements matching component of ICP algorithm for pointer-to-surface registration.
PA3: F_reg = Identity
PA4: Add iterative refinement

Usage: python pa3.py <dataset_letter>
Datasets: A-F (debug), G-J (unknown)
"""

import sys
import numpy as np
from pathlib import Path
from programs import utility_functions as uf
from programs import icp_matching


def run_pa3(dataset_letter, input_dir="2025 PA345 Student Data", output_dir="output"):
    """
    Run PA3 pipeline for a given dataset.

    Args:
        dataset_letter (str): Dataset letter (e.g., "A", "B", "G")
        input_dir (str): Input directory containing data files
        output_dir (str): Output directory for results
    """
    # Determine if this is a debug or unknown dataset
    debug_letters = ['A', 'B', 'C', 'D', 'E', 'F']
    unknown_letters = ['G', 'H', 'J']

    dataset_letter = dataset_letter.upper()

    if dataset_letter in debug_letters:
        dataset_type = "Debug"
    elif dataset_letter in unknown_letters:
        dataset_type = "Unknown"
    else:
        print(f"Error: Unknown dataset letter '{dataset_letter}'")
        print(f"Available: Debug={debug_letters}, Unknown={unknown_letters}")
        return

    dataset_name = f"PA3-{dataset_letter}-{dataset_type}"
    print(f"Processing dataset: {dataset_name}")

    print("Reading input files...")

    body_A_file = f"{input_dir}/Problem3-BodyA.txt"
    body_A = uf.read_body_markers_and_tip(body_A_file)
    if body_A is None:
        print(f"Error: Could not read {body_A_file}")
        return
    print(f"  Body A: {body_A['N_markers']} markers")

    body_B_file = f"{input_dir}/Problem3-BodyB.txt"
    body_B = uf.read_body_markers_and_tip(body_B_file)
    if body_B is None:
        print(f"Error: Could not read {body_B_file}")
        return
    print(f"  Body B: {body_B['N_markers']} markers")

    mesh_file = f"{input_dir}/Problem3Mesh.sur"
    mesh = uf.read_surface_mesh(mesh_file)
    if mesh is None:
        print(f"Error: Could not read {mesh_file}")
        return
    print(f"  Mesh: {mesh['N_vertices']} vertices, {mesh['N_triangles']} triangles")

    sample_file = f"{input_dir}/{dataset_name}-SampleReadingsTest.txt"
    sample_data = uf.read_sample_readings(sample_file)
    if sample_data is None:
        print(f"Error: Could not read {sample_file}")
        return

    samples = uf.parse_sample_readings(sample_data, body_A['N_markers'], body_B['N_markers'])
    print(f"  {samples['N_samps']} frames, {samples['N_D']} dummy markers")

    print("\nProcessing frames...")

    results = []
    for k, frame_data in enumerate(samples['frames']):
        result = icp_matching.process_pa3_frame(
            body_A,
            body_B,
            mesh,
            frame_data,
            F_reg=None
        )
        results.append(result)

        if (k + 1) % 10 == 0 or (k + 1) == samples['N_samps']:
            print(f"  {k + 1}/{samples['N_samps']} frames")

    print("\nWriting output...")

    output_file = f"{dataset_name}-Output"
    uf.write_pa3_output(output_file, results, output_dir=output_dir)
    print(f"  {output_dir}/{output_file}.txt")

    distances = [r['distance'] for r in results]
    print(f"\nStatistics:")
    print(f"  Mean: {np.mean(distances):.3f} mm")
    print(f"  Std:  {np.std(distances):.3f} mm")
    print(f"  Min:  {np.min(distances):.3f} mm")
    print(f"  Max:  {np.max(distances):.3f} mm")

    if dataset_type == "Debug":
        answer_file = f"{input_dir}/{dataset_name}-Output.txt"
        if Path(answer_file).exists():
            print(f"\nComparing with answer file...")
            compare_outputs(f"{output_dir}/{output_file}.txt", answer_file)

    print(f"\nDone: {dataset_name}")


def compare_outputs(output_file, answer_file):
    """Compare generated output with answer file."""
    try:
        with open(output_file, 'r') as f:
            output_lines = f.readlines()
        with open(answer_file, 'r') as f:
            answer_lines = f.readlines()

        max_diff_d = 0.0
        max_diff_c = 0.0
        max_diff_dist = 0.0

        for i in range(1, min(len(output_lines), len(answer_lines))):
            output_vals = [float(x.strip()) for x in output_lines[i].split()]
            answer_vals = [float(x.strip()) for x in answer_lines[i].split()]

            diff_d = np.linalg.norm(np.array(output_vals[:3]) - np.array(answer_vals[:3]))
            max_diff_d = max(max_diff_d, diff_d)

            diff_c = np.linalg.norm(np.array(output_vals[3:6]) - np.array(answer_vals[3:6]))
            max_diff_c = max(max_diff_c, diff_c)

            diff_dist = abs(output_vals[6] - answer_vals[6])
            max_diff_dist = max(max_diff_dist, diff_dist)

        print(f"  Max diff d_k: {max_diff_d:.6f} mm")
        print(f"  Max diff c_k: {max_diff_c:.6f} mm")
        print(f"  Max diff distance: {max_diff_dist:.6f} mm")

        if max_diff_d < 0.01 and max_diff_c < 0.01 and max_diff_dist < 0.001:
            print("  ✓ Match")
        else:
            print("  ⚠ Differs")

    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pa3.py <dataset_letter>")
        print("\nExamples:")
        print("  python pa3.py A    # Debug dataset A")
        print("  python pa3.py B    # Debug dataset B")
        print("  python pa3.py G    # Unknown dataset G")
        print("\nAvailable datasets:")
        print("  Debug (with answer files): A, B, C, D, E, F")
        print("  Unknown (no answer files): G, H, J")
        sys.exit(1)

    dataset_letter = sys.argv[1]

    # Create output directory if it doesn't exist
    output_dir = "output"
    Path(output_dir).mkdir(exist_ok=True)

    # Run PA3
    run_pa3(dataset_letter)
