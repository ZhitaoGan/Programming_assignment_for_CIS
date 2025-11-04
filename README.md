# CISPA - Computer Integrated Surgery Programming Assignments

Implementation of calibration, tracking, and distortion correction algorithms for optical and electromagnetic tracking systems.

## Overview

This project implements core algorithms for surgical navigation systems:

### Programming Assignment 1 (PA1)
- **Point Set Registration**: SVD-based least squares algorithm for coordinate frame alignment
- **EM Pivot Calibration**: Calibrate electromagnetic tracking probe tip position
- **Optical Pivot Calibration**: Calibrate optical tracking probe using calibration body
- **Frame Transformations**: Compute expected marker positions across coordinate systems

### Programming Assignment 2 (PA2)
- **Distortion Correction**: 3D Bernstein polynomial-based correction for EM tracker distortion
- **Corrected Pivot Calibration**: EM pivot calibration with distortion correction applied
- **Fiducial Registration**: Compute EM→CT coordinate transformation using fiducials
- **Surgical Navigation**: Track probe tip positions in CT image coordinates

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
conda env create -f environment.yml
conda activate cispa
```

Note: The `logging/` and `output/` directories are included in the repository.

### 2. Run Programming Assignment 1

```bash
# Process all PA1 datasets (7 debug + 4 unknown)
python run_all_pa1.py
```

This generates:
1. **Frame registrations** (Fa, Fd) for all datasets
2. **EM and optical pivot calibrations**
3. **Final output files** with expected C coordinates
4. **Comparison with expected results** for debug datasets

**Output Files Generated:**
- `output/Fa_*_registration.txt`, `output/Fd_*_registration.txt` - Frame transformation matrices
- `output/*_EM_pivot.txt`, `output/*_Optpivot.txt` - Pivot calibration results
- `output/pa1-*-output1.txt` - Final results with expected C coordinates (11 datasets)


### 3. Run Programming Assignment 2

```bash
# Process all PA2 datasets (6 debug + 4 unknown)
python run_all_pa2.py
```

This generates:
1. **Output1 files** for unknown datasets (debug datasets already have them)
2. **Distortion correction** using 3D Bernstein polynomials (degree 5)
3. **EM probe calibration** with distortion correction applied
4. **Fiducial positions** in EM tracker coordinates
5. **EM→CT registration** using point set registration
6. **Navigation tracking** with probe tip positions in CT coordinates

**Output Files Generated:**
- `output/pa2-unknown-*-output1.txt` - Distortion calibration data (4 unknown datasets)
- `output/pa2-*-output2.txt` - Final probe tip positions in CT coordinates (10 datasets)
- `output/pa2-unknown-*-Fa.txt`, `output/pa2-unknown-*-Fd.txt` - Frame transformation matrices (unknown datasets only)
- `output/pa2-unknown-*-EM_pivot.txt`, `output/pa2-unknown-*-Optpivot.txt` - Pivot calibration results (unknown datasets only)

### 4. Run Tests

```bash
# Run all tests
python test_runner.py all
```


## Project Structure

```
.
├── programs/
│   ├── frame_transform.py
│   ├── pivot_calibration.py
│   ├── distortion_correction.py
│   ├── utility_functions.py
│   └── tests/
│       ├── test_frame_transform.py
│       ├── test_pivot_calibration.py
│       └── test_distortion_correction.py
├── pa1.py
├── run_all_pa1.py
├── run_all_pa2.py
├── PA 1 Student Data/
│   ├── pa1-debug-*.txt
│   └── pa1-unknown-*.txt
├── PA 2 Student Data/
│   ├── pa2-debug-*.txt
│   └── pa2-unknown-*.txt
├── output/
│   ├── pa1-*.txt
│   └── pa2-*.txt
└── logging/
```

## Input Data

### PA1 Input Data

**Location:** `PA 1 Student Data/` directory

**Datasets:**
- **Debug datasets:** `pa1-debug-a` through `pa1-debug-g` (7 datasets)
- **Unknown datasets:** `pa1-unknown-h` through `pa1-unknown-k` (4 datasets)

**Input Files:**

| File | Format | Description |
|------|--------|-------------|
| `*-calbody.txt` | `N_D, N_A, N_C` + coordinates | Calibration body marker positions |
| `*-calreadings.txt` | `N_D, N_A, N_C, N_frames` + data | Multi-frame calibration readings |
| `*-empivot.txt` | `N_G, N_frames` + data | EM probe marker data |
| `*-optpivot.txt` | `N_D, N_H, N_frames` + data | Optical tracking data |
| `*-auxilliary1.txt` | (optional) | Auxiliary data files |

### PA2 Input Data

**Location:** `PA 2 Student Data/` directory

**Datasets:**
- **Debug datasets:** `pa2-debug-a` through `pa2-debug-f` (6 datasets)
- **Unknown datasets:** `pa2-unknown-g` through `pa2-unknown-j` (4 datasets)

**Input Files:**

| File | Format | Description |
|------|--------|-------------|
| `*-calbody.txt` | `N_D, N_A, N_C` + coordinates | Calibration body marker positions |
| `*-calreadings.txt` | `N_D, N_A, N_C, N_frames` + data | Multi-frame calibration readings (for distortion correction) |
| `*-empivot.txt` | `N_G, N_frames` + data | EM probe marker data (will be corrected for distortion) |
| `*-optpivot.txt` | `N_D, N_H, N_frames` + data | Optical tracking data |
| `*-em-fiducialss.txt` | `N_G, N_frames` + data | EM probe touching each fiducial |
| `*-ct-fiducials.txt` | `N_B` + coordinates | Fiducial positions in CT image coordinates |
| `*-EM-nav.txt` | `N_G, N_frames` + data | EM navigation frames to track |
| `*-output1.txt` | `N_C, N_frames` + data | Expected C positions (debug datasets only) |
| `*-auxilliary1.txt`, `*-auxilliary2.txt` | (optional) | Auxiliary data files |

**Note:** All coordinates are in millimeters (mm).

## Output Data

### PA1 Output Data

**Location:** `output/` directory

**Output Files:**

| File Type | Example | Description |
|-----------|---------|-------------|
| Frame registrations | `Fa_*_registration.txt`, `Fd_*_registration.txt` | 4x4 transformation matrices (flattened) |
| Pivot calibrations | `*_EM_pivot.txt`, `*_Optpivot.txt` | Tip position, pivot point, residual error |
| Final outputs | `pa1-*-output1.txt` | Complete results with C coordinates |


**Generated for:**
- All debug datasets (a-g)
- All unknown datasets (h-k)

### PA2 Output Data

**Location:** `output/` directory

**Output Files:**

| File Type | Example | Description |
|-----------|---------|-------------|
| Frame registrations(unknown only) | `pa2-*-Fa.txt`, `pa2-*-Fd.txt` | 4x4 transformation matrices (flattened) |
| Pivot calibrations(unknown only) | `pa2-*-EM_pivot.txt`, `pa2-*-Optpivot.txt` | Tip position, pivot point, residual error |
| Output1 (unknown only) | `pa2-unknown-*-output1.txt` | Expected C positions (auto-generated for unknown datasets) |
| Output2 (all datasets) | `pa2-*-output2.txt` | Probe tip positions in CT coordinates |


**Generated for:**
- All debug datasets (a-f): Output2 files
- All unknown datasets (g-j): Output1 and Output2 files, plus intermediate files (Fa, Fd, EM_pivot, Optpivot)

## Testing

The test suite validates core algorithms using synthetic data with known ground truth:

### PA1 Tests

**Test Coverage (12 tests):**
- **Frame transformations** (`test_frame_transform.py` - 8 tests):
  - Transform operations (transform, inverse, compose)
  - Point set registration (identity, translation, rotation, general transformations)
  - Noise robustness testing
- **Pivot calibration** (`test_pivot_calibration.py` - 4 tests):
  - Least squares solver validation
  - EM pivot calibration with synthetic data
  - Optical pivot calibration with synthetic data
  - Noise robustness testing

**Run PA1 tests:**
```bash
# All PA1 tests
python test_runner.py frame
python test_runner.py pivot

# Or run both together
python test_runner.py all
```

### PA2 Tests

**Test Coverage (6 tests):**
- **Distortion correction** (`test_distortion_correction.py` - 6 tests):
  - Zero distortion (identity correction)
  - Linear distortion fitting and correction
  - Quadratic distortion fitting and correction
  - Single point correction
  - Frame marker correction
  - Correction consistency validation

**Run PA2 tests:**
```bash
# All PA2 tests
python test_runner.py distortion
```

### Run All Tests

```bash
# All tests (PA1 + PA2)
python test_runner.py all
```

**Available test commands:**
- `python test_runner.py all` - Run all tests
- `python test_runner.py frame` - Run frame transform tests (PA1)
- `python test_runner.py pivot` - Run pivot calibration tests (PA1)
- `python test_runner.py distortion` - Run distortion correction tests (PA2)

## Algorithm Details

### PA1 Algorithms

#### Point Set Registration
- **Method**: SVD-based closed-form least squares
- **Input**: Two corresponding point sets (source, target)
- **Output**: Rotation matrix R and translation vector t
- **Equation**: `target = R @ source + t`
- **Application**: Register calibration body markers (A→D, C→D), optical tracker frames

#### EM Pivot Calibration
- **Method**: Least squares optimization
- **Input**: Multiple frames of probe marker positions (EM tracker)
- **Output**: Tip position (probe frame), pivot point (EM frame), residual error
- **Constraint**: Pivot point remains fixed across all poses
- **Equation**: `p_pivot = R_i @ p_tip + t_i` for all frames i

#### Optical Pivot Calibration
- **Method**: Integrated calibration using EM geometry
- **Input**: Optical tracker data + calibration body geometry
- **Steps**:
  1. Register optical→EM using calibration body
  2. Transform probe markers to EM frame
  3. Solve pivot equation using EM pivot calibration method
- **Output**: Tip position (probe frame), pivot point (EM frame), residual error

### PA2 Algorithms

#### Distortion Correction
- **Method**: 3D Bernstein (Bezier) polynomial fitting (degree 5)
- **Input**: Distorted EM measurements and corresponding expected coordinates
- **Model**: Tensor-product Bernstein polynomial that maps distorted coordinates to corrected coordinates
  - The correction function is a 3D tensor product of Bernstein basis polynomials:
    
    $$
    \hat{x}(\mathbf{u}) = \sum_{i=0}^{n} \sum_{j=0}^{n} \sum_{k=0}^{n} c_{ijk}^{(x)} \cdot B_i^{(n)}(u_x) \cdot B_j^{(n)}(u_y) \cdot B_k^{(n)}(u_z)
    $$
    
    **Notation:**
    - $\mathbf{u} = (u_x, u_y, u_z)$: scaled distorted coordinates (normalized to [0,1]³)
    - $\hat{x}(\mathbf{u})$: corrected x-coordinate
    - $n$: polynomial degree (typically 5)
    - $B_i^{(n)}(t) = \binom{n}{i} t^i (1-t)^{n-i}$: i-th Bernstein basis polynomial of degree n
    - $c_{ijk}^{(x)}$: fitted coefficients for the x-coordinate (total of $(n+1)^3$ coefficients)
    - Similar equations apply for $\hat{y}(\mathbf{u})$ and $\hat{z}(\mathbf{u})$ with separate coefficient sets
- **Steps**:
  1. Scale distorted coordinates to [0,1]³ box
  2. Fit coefficients using least squares (one per coordinate)
  3. Apply correction by evaluating Bernstein polynomial
- **Output**: Corrected coordinates with reduced EM tracker distortion

#### Corrected EM Pivot Calibration
- **Method**: EM pivot calibration with distortion correction applied
- **Input**: EM probe data (distorted) + distortion correction model
- **Steps**:
  1. Apply distortion correction to all EM probe marker positions
  2. Perform standard EM pivot calibration on corrected data
- **Output**: Tip position (probe frame), pivot point (EM frame), residual error

#### Fiducial Registration
- **Method**: Point set registration (SVD-based) using fiducial pairs
- **Input**: 
  - Fiducial positions in EM tracker coordinates (from corrected probe measurements)
  - Fiducial positions in CT image coordinates
- **Output**: Transformation matrix EM→CT
- **Application**: Register EM tracker space to CT image space for surgical navigation

#### Surgical Navigation
- **Method**: Transform probe tip positions to CT coordinates
- **Input**: 
  - EM navigation frames (distorted)
  - Distortion correction model
  - EM pivot calibration (tip position)
  - EM→CT transformation (from fiducial registration)
- **Steps**:
  1. Apply distortion correction to navigation frames
  2. Compute probe tip positions in EM tracker space
  3. Transform to CT image coordinates using EM→CT registration
- **Output**: Probe tip positions in CT image coordinates

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
