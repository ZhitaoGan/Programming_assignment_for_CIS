# CISPA - Computer Integrated Surgery Programming Assignment 1

Implementation of calibration and tracking algorithms for optical and electromagnetic tracking systems.

## Overview

This project implements core algorithms for surgical navigation systems:
- **Point Set Registration**: SVD-based least squares algorithm for coordinate frame alignment
- **EM Pivot Calibration**: Calibrate electromagnetic tracking probe tip position
- **Optical Pivot Calibration**: Calibrate optical tracking probe using calibration body
- **Frame Transformations**: Compute expected marker positions across coordinate systems

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
conda env create -f environment.yml
conda activate cispa
```

Note: The `logging/` and `output/` directories are included in the repository.

### 2. Process All Datasets (Recommended)

```bash
# Process all 11 datasets (7 debug + 4 unknown)
python process_all_datasets.py
```

This generates:
- Frame registrations (Fa, Fd) for all datasets
- EM and optical pivot calibrations
- Final output files with expected C coordinates
- Comparison with expected results for debug datasets

**Expected Results**: 55/55 operations successful

### 3. Run Tests

```bash
# Run all tests (12 tests, ~0.07s)
pytest programs/tests/ -v
```

## Manual Processing (Individual Datasets)

Process a single dataset step-by-step:

```bash
DATASET="pa1-debug-a"

# 1. Generate frame registrations (4x4 transformation matrices)
python pa1.py --name ${DATASET}-calbody --name_2 ${DATASET}-calreadings --output_file Fa_${DATASET}_registration
python pa1.py --name ${DATASET}-calbody --name_2 ${DATASET}-calreadings --output_file Fd_${DATASET}_registration

# 2. Perform pivot calibrations
python pa1.py --name_3 ${DATASET}-empivot --output_file1 ${DATASET}_EM_pivot
python pa1.py --name ${DATASET}-calbody --name_4 ${DATASET}-optpivot --output_file2 ${DATASET}_Optpivot

# 3. Generate final output with expected C coordinates
python pa1.py --name ${DATASET}-calbody --input_reg Fa_${DATASET}_registration --input_reg2 Fd_${DATASET}_registration --output_file ${DATASET}-output1
```

## Project Structure

```
.
├── programs/
│   ├── frame_transform.py        # Point set registration & transformations
│   ├── pivot_calibration.py      # EM and optical pivot calibration
│   ├── utility_functions.py      # File I/O and data processing
│   └── tests/                    # Unit tests (12 tests)
├── pa1.py                        # Main CLI interface
├── process_all_datasets.py       # Batch processing script
├── PA 1 Student Data/            # Input datasets (7 debug + 4 unknown)
├── output/                       # Generated results (created automatically)
└── logging/                      # Log files (created automatically)
```

## Input Data

Input files are in `PA 1 Student Data/` directory:

| File | Format | Description |
|------|--------|-------------|
| `*-calbody.txt` | `N_D, N_A, N_C` + coordinates | Calibration body marker positions |
| `*-calreadings.txt` | `N_D, N_A, N_C, N_frames` + data | Multi-frame calibration readings |
| `*-empivot.txt` | `N_G, N_frames` + data | EM probe marker data |
| `*-optpivot.txt` | `N_D, N_H, N_frames` + data | Optical tracking data |

All coordinates are in millimeters (mm).

## Output Files

Generated in `output/` directory:

| File Type | Example | Description |
|-----------|---------|-------------|
| Frame registrations | `Fa_*_registration.txt`, `Fd_*_registration.txt` | 4x4 transformation matrices (flattened) |
| Pivot calibrations | `*_EM_pivot.txt`, `*_Optpivot.txt` | Tip position, pivot point, residual error |
| Final outputs | `*-output1.txt` | Complete results with C coordinates |

**Pivot calibration output format:**
```
tip_x, tip_y, tip_z
pivot_x, pivot_y, pivot_z
residual_error, 0, 0
```

## Testing

The test suite validates core algorithms using synthetic data with known ground truth:

**Test Coverage (12 tests):**
- Frame transformations (3 tests): transform, inverse, compose
- Point set registration (5 tests): identity, translation, rotation, general, noise robustness
- Pivot calibration (4 tests): least squares solver + EM/optical calibration with noise

**Run tests:**
```bash
# All tests
pytest programs/tests/ -v

# Specific module
pytest programs/tests/test_pivot_calibration.py -v
pytest programs/tests/test_frame_transform.py -v

# With detailed output
pytest programs/tests/ -v -s
```

## Algorithm Details

### Point Set Registration
- **Method**: SVD-based closed-form least squares
- **Input**: Two corresponding point sets (source, target)
- **Output**: Rotation matrix R and translation vector t
- **Equation**: `target = R @ source + t`

### EM Pivot Calibration
- **Method**: Least squares optimization
- **Input**: Multiple frames of probe marker positions
- **Output**: Tip position (probe frame), pivot point (EM frame), residual error
- **Constraint**: Pivot point remains fixed across all poses

### Optical Pivot Calibration
- **Method**: Integrated calibration using EM geometry
- **Input**: Optical tracker data + calibration body geometry
- **Steps**: Register optical→EM, transform probe markers, solve pivot equation
- **Output**: Tip position (probe frame), pivot point (EM frame), residual error

## Citation

Template for CIS I programming assignments at Johns Hopkins. [Course page](https://ciis.lcsr.jhu.edu/doku.php?id=courses:455-655:455-655).

If you use this template or any of the code within for your project, please cite:

```bibtex
@misc{benjamindkilleen2022Sep,
 author = {Killeen, Benjamin D.},
 title = {{cispa: Template for CIS I programming assignments at Johns Hopkins}},
 journal = {GitHub},
 year = {2022},
 month = {Sep},
 url = {https://github.com/benjamindkilleen/cispa}
}

@misc{taylor_cispa1_2025,
  author       = {Russell H. Taylor and Computer Integrated Surgery I},
  title        = {{Computer Integrated Surgery I (600.445/645) Lecture Slides: Calibration and Point cloud point cloud rigid transformations}},
  howpublished = {Course materials, Johns Hopkins University},
  year         = {2025},
  month        = {Oct},
  note         = {Algorithms for rigid point set registration, frame transformations, and pivot calibration used as references for implementation.},
  url          = {https://ciis.lcsr.jhu.edu/doku.php?id=courses:455-655:2025:fall-2025-schedule}
}
```
