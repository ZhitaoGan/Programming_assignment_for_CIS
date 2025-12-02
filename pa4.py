"""
PA4 - Complete ICP Algorithm

Implements complete Iterative Closest Point (ICP) algorithm with iterative refinement.
PA4: F_reg is iteratively refined until convergence

Usage: python pa4.py <dataset_letter>
Datasets: A-F (debug), G-K (unknown)
"""

import sys
import numpy as np
from pathlib import Path
from programs import utility_functions as uf
from programs import icp_matching


def run_pa4(dataset_letter, dataset_variant=None, input_dir="2025 PA345 Student Data", output_dir="output"):
    """
    Run PA4 pipeline for a given dataset.

    Args:
        dataset_letter (str): Dataset letter (e.g., "A", "B", "G")
        dataset_variant (str, optional): Variant type - "Debug", "Demo-Fast", "Demo-Slow", or None (auto-detect)
        input_dir (str): Input directory containing data files
        output_dir (str): Output directory for results
    """
    # Determine if this is a debug or unknown dataset
    debug_letters = ['A', 'B', 'C', 'D', 'E', 'F']
    unknown_letters = ['G', 'H', 'J', 'K']

    dataset_letter = dataset_letter.upper()

    # Auto-detect dataset type if not specified
    if dataset_variant is None:
        if dataset_letter in debug_letters:
            dataset_type = "Debug"
        elif dataset_letter in unknown_letters:
            dataset_type = "Unknown"
        else:
            print(f"Error: Unknown dataset letter '{dataset_letter}'")
            print(f"Available: Debug={debug_letters}, Unknown={unknown_letters}")
            return
    else:
        dataset_type = dataset_variant

    dataset_name = f"PA4-{dataset_letter}-{dataset_type}"
    print(f"Processing dataset: {dataset_name}")

    print("Reading input files...")

    # Read body definitions (shared across all PA4 datasets)
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

    # Read mesh (shared across all PA4 datasets)
    mesh_file = f"{input_dir}/Problem3Mesh.sur"
    mesh = uf.read_surface_mesh(mesh_file)
    if mesh is None:
        print(f"Error: Could not read {mesh_file}")
        return
    print(f"  Mesh: {mesh['N_vertices']} vertices, {mesh['N_triangles']} triangles")

    # Read sample readings for this specific dataset
    sample_file = f"{input_dir}/{dataset_name}-SampleReadingsTest.txt"
    sample_data = uf.read_sample_readings(sample_file)
    if sample_data is None:
        print(f"Error: Could not read {sample_file}")
        return

    samples = uf.parse_sample_readings(sample_data, body_A['N_markers'], body_B['N_markers'])
    print(f"  {samples['N_samps']} frames, {samples['N_D']} dummy markers")

    print("\nRunning ICP algorithm...")

    # Run ICP on all frames
    icp_result = icp_matching.run_icp_on_all_frames(
        body_A,
        body_B,
        mesh,
        samples['frames'],
        max_iterations=100,
        convergence_threshold=1e-6
    )

    results = icp_result['results']
    F_reg = icp_result['F_reg']
    iterations = icp_result['iterations']
    converged = icp_result['converged']

    print(f"  ICP iterations: {iterations}")
    print(f"  Converged: {converged}")
    print(f"  F_reg rotation:\n{F_reg.rotation_matrix}")
    print(f"  F_reg translation: {F_reg.translation_vector}")

    print("\nWriting output...")

    output_file = f"{dataset_name}-Output"
    uf.write_pa4_output(output_file, results, output_dir=output_dir)
    print(f"  {output_dir}/{output_file}.txt")

    # Statistics
    distances = [r['distance'] for r in results]
    print(f"\nStatistics:")
    print(f"  Mean distance: {np.mean(distances):.4f} mm")
    print(f"  Std distance:  {np.std(distances):.4f} mm")
    print(f"  Min distance:  {np.min(distances):.4f} mm")
    print(f"  Max distance:  {np.max(distances):.4f} mm")

    # Compare with answer file if debug dataset
    if dataset_type == "Debug":
        answer_file = f"{input_dir}/{dataset_name}-Answer.txt"
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

        max_diff_s = 0.0
        max_diff_c = 0.0
        max_diff_dist = 0.0

        for i in range(1, min(len(output_lines), len(answer_lines))):
            output_vals = [float(x.strip()) for x in output_lines[i].split()]
            answer_vals = [float(x.strip()) for x in answer_lines[i].split()]

            # For PA4, output is s_k (not d_k)
            diff_s = np.linalg.norm(np.array(output_vals[:3]) - np.array(answer_vals[:3]))
            max_diff_s = max(max_diff_s, diff_s)

            diff_c = np.linalg.norm(np.array(output_vals[3:6]) - np.array(answer_vals[3:6]))
            max_diff_c = max(max_diff_c, diff_c)

            diff_dist = abs(output_vals[6] - answer_vals[6])
            max_diff_dist = max(max_diff_dist, diff_dist)

        print(f"  Max diff s_k: {max_diff_s:.6f} mm")
        print(f"  Max diff c_k: {max_diff_c:.6f} mm")
        print(f"  Max diff distance: {max_diff_dist:.6f} mm")

        if max_diff_s < 0.01 and max_diff_c < 0.01 and max_diff_dist < 0.001:
            print("  ✓ Match")
        else:
            print("  ⚠ Differs")

    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pa4.py <dataset_letter> [variant]")
        print("\nExamples:")
        print("  python pa4.py A              # Debug dataset A")
        print("  python pa4.py A Demo-Fast    # Demo-Fast variant of dataset A")
        print("  python pa4.py A Demo-Slow    # Demo-Slow variant of dataset A")
        print("  python pa4.py G              # Unknown dataset G")
        print("\nAvailable datasets:")
        print("  Debug (with answer files): A, B, C, D, E, F")
        print("  Unknown (no answer files): G, H, J, K")
        print("\nDemo variants (A-B only):")
        print("  Demo-Fast: Scenarios that converge quickly")
        print("  Demo-Slow: Scenarios that require more iterations")
        sys.exit(1)

    dataset_letter = sys.argv[1]
    dataset_variant = sys.argv[2] if len(sys.argv) > 2 else None

    # Create output directory if it doesn't exist
    output_dir = "output"
    Path(output_dir).mkdir(exist_ok=True)

    # Run PA4
    run_pa4(dataset_letter, dataset_variant)
