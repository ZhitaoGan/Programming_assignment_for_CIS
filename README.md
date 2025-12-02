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

### Programming Assignment 3 (PA3)
- **ICP Matching**: Implement matching component of Iterative Closest Point algorithm
- **Rigid Body Tracking**: Track two rigid bodies (pointer and bone) using optical markers
- **Surface Mesh Registration**: Find closest points on triangular bone surface mesh
- **Pointer-to-Surface Matching**: Compute pointer tip position relative to bone surface

### Programming Assignment 4 (PA4)
- **Complete ICP Algorithm**: Full Iterative Closest Point with iterative refinement
- **Registration Refinement**: Iteratively update F_reg transformation until convergence
- **Point-to-Mesh Registration**: Register pointer tip positions to bone surface mesh
- **Convergence Optimization**: Automatic convergence detection based on transformation change

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

### 4. Run Programming Assignment 3

```bash
# Process individual PA3 dataset
python pa3.py A    # Debug dataset A
python pa3.py G    # Unknown dataset G

# Or process all debug datasets
python pa3.py A && python pa3.py B && python pa3.py C && python pa3.py D && python pa3.py E && python pa3.py F
```

This generates:
1. **Rigid body transformations** (F_A, F_B) for pointer and bone bodies
2. **Pointer tip position** in bone coordinate frame (d_k)
3. **Closest point on mesh** for each sample (c_k)
4. **Distance measurements** between pointer tip and bone surface

**Output Files Generated:**
- `output/PA3-A-Debug-Output.txt` through `output/PA3-F-Debug-Output.txt` - Results for debug datasets (6 datasets)
- `output/PA3-G-Unknown-Output.txt` through `output/PA3-J-Unknown-Output.txt` - Results for unknown datasets (3 datasets)

**Available Datasets:**
- Debug (with answer files): A, B, C, D, E, F
- Unknown (no answer files): G, H, J

### 5. Run Programming Assignment 4

```bash
# Process individual PA4 dataset
python pa4.py A              # Debug dataset A
python pa4.py G              # Unknown dataset G

# Process Demo variants (only A and B have Demo variants)
python pa4.py A Demo-Fast    # Fast convergence scenario
python pa4.py A Demo-Slow    # Slow convergence scenario
python pa4.py B Demo-Fast
python pa4.py B Demo-Slow

# Process all datasets (debug, demo, and unknown)
python pa4.py A && python pa4.py B && python pa4.py C && python pa4.py D && python pa4.py E && python pa4.py F && python pa4.py A Demo-Fast && python pa4.py A Demo-Slow && python pa4.py B Demo-Fast && python pa4.py B Demo-Slow && python pa4.py G && python pa4.py H && python pa4.py J && python pa4.py K
```

This generates:
1. **Complete ICP algorithm** with iterative refinement of F_reg
2. **Registered pointer tip positions** in bone coordinate frame (s_k = F_reg · d_k)
3. **Closest point on mesh** for each sample (c_k)
4. **Distance measurements** between registered tip and bone surface

**Output Files Generated:**
- `output/PA4-A-Debug-Output.txt` through `output/PA4-F-Debug-Output.txt` - Results for debug datasets (6 files)
- `output/PA4-A-Demo-Fast-Output.txt`, `output/PA4-A-Demo-Slow-Output.txt`, etc. - Results for demo variants (4 files)
- `output/PA4-G-Unknown-Output.txt` through `output/PA4-K-Unknown-Output.txt` - Results for unknown datasets (4 files)

**Available Datasets:**
- Debug (with answer files): A, B, C, D, E, F
- Demo variants (A-B only): Demo-Fast, Demo-Slow - demonstrate different ICP convergence speeds
- Unknown (no answer files): G, H, J, K

### 6. Run Tests

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
│   ├── icp_matching.py
│   ├── utility_functions.py
│   └── tests/
│       ├── test_frame_transform.py
│       ├── test_pivot_calibration.py
│       ├── test_distortion_correction.py
│       └── test_icp_matching.py
├── pa1.py
├── pa3.py
├── pa4.py
├── run_all_pa1.py
├── run_all_pa2.py
├── PA 1 Student Data/
│   ├── pa1-debug-*.txt
│   └── pa1-unknown-*.txt
├── PA 2 Student Data/
│   ├── pa2-debug-*.txt
│   └── pa2-unknown-*.txt
├── 2025 PA345 Student Data/
│   ├── Problem3-BodyA.txt
│   ├── Problem3-BodyB.txt
│   ├── Problem3Mesh.sur
│   ├── PA3-*-Debug-SampleReadingsTest.txt
│   └── PA3-*-Unknown-SampleReadingsTest.txt
├── output/
│   ├── pa1-*.txt
│   ├── pa2-*.txt
│   └── PA3-*-Output.txt
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

### PA3 and PA4 Input Data

**Location:** `2025 PA345 Student Data/` directory

**PA3 Datasets:**
- **Debug datasets:** `PA3-A-Debug` through `PA3-F-Debug` (6 datasets)
- **Unknown datasets:** `PA3-G-Unknown`, `PA3-H-Unknown`, `PA3-J-Unknown` (3 datasets)

**PA4 Datasets:**
- **Debug datasets:** `PA4-A-Debug` through `PA4-F-Debug` (6 datasets)
- **Demo variants:** `PA4-A-Demo-Fast`, `PA4-A-Demo-Slow`, `PA4-B-Demo-Fast`, `PA4-B-Demo-Slow` (4 datasets)
  - Demo-Fast: Scenarios with good initial alignment (converge quickly)
  - Demo-Slow: Scenarios with poor initial alignment (require more iterations)
- **Unknown datasets:** `PA4-G-Unknown`, `PA4-H-Unknown`, `PA4-J-Unknown`, `PA4-K-Unknown` (4 datasets)

**Input Files:**

| File | Format | Description |
|------|--------|-------------|
| `Problem3-BodyA.txt` | `N_markers` + coordinates + tip | Rigid body A (pointer) definition in body coordinates |
| `Problem3-BodyB.txt` | `N_markers` + coordinates + tip | Rigid body B (bone-attached) definition in body coordinates |
| `Problem3Mesh.sur` | `N_vertices`, vertices, `N_triangles`, triangles | Bone surface mesh (vertices + triangle indices) |
| `PA3-*-SampleReadingsTest.txt` | `N_s, N_samps` + marker data | LED marker positions in tracker coordinates (PA3) |
| `PA4-*-SampleReadingsTest.txt` | `N_s, N_samps` + marker data | LED marker positions in tracker coordinates (PA4) |
| `PA3-*-Output.txt` | (debug only) | Expected output for validation (PA3) |
| `PA4-*-Answer.txt` | (debug only) | Expected output for validation (PA4) |

**Note:** Problem3-BodyA.txt, Problem3-BodyB.txt, and Problem3Mesh.sur are shared across all PA3 and PA4 datasets. All coordinates are in millimeters (mm).

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

### PA3 and PA4 Output Data

**Location:** `output/` directory

**PA3 Output Files:**

| File Type | Example | Description |
|-----------|---------|-------------|
| ICP matching results | `PA3-*-Output.txt` | Pointer tip position (d_k), closest point on mesh (c_k), distance |

**PA3 Output Format:**
- Header: `N_samps filename 0`
- Each line: `d_x d_y d_z c_x c_y c_z distance`
  - `d_k = (d_x, d_y, d_z)`: Pointer tip position in bone coordinate frame
  - `c_k = (c_x, c_y, c_z)`: Closest point on bone surface mesh
  - `distance = ||d_k - c_k||`: Distance from pointer tip to mesh surface (mm)

**PA4 Output Files:**

| File Type | Example | Description |
|-----------|---------|-------------|
| Complete ICP results | `PA4-*-Output.txt` | Registered tip position (s_k), closest point on mesh (c_k), distance |

**PA4 Output Format:**
- Header: `N_samps filename 0`
- Each line: `s_x s_y s_z c_x c_y c_z distance`
  - `s_k = (s_x, s_y, s_z)`: Registered pointer tip position (s_k = F_reg · d_k) in bone coordinate frame
  - `c_k = (c_x, c_y, c_z)`: Closest point on bone surface mesh
  - `distance = ||s_k - c_k||`: Distance from registered tip to mesh surface (mm)

**Generated for:**
- **PA3**: Debug datasets (A-F): 6 output files with answer file comparison; Unknown datasets (G, H, J): 3 output files
- **PA4**:
  - Debug datasets (A-F): 6 output files with answer file comparison
  - Demo variants (A-B only): 4 output files (Demo-Fast and Demo-Slow for datasets A and B)
  - Unknown datasets (G, H, J, K): 4 output files

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

### PA3 Tests

**Test Coverage (11 tests):**
- **Closest point on triangle** (`test_icp_matching.py` - 5 tests):
  - Point inside triangle
  - Point on triangle vertex
  - Point on triangle edge
  - Point outside triangle near vertex
  - Point in triangle plane
- **Project on segment** (`test_icp_matching.py` - 3 tests):
  - Project onto segment midpoint
  - Project clamped to segment endpoint
  - Degenerate segment (point)
- **Closest point on mesh** (`test_icp_matching.py` - 3 tests):
  - Simple two-triangle mesh
  - Closest to mesh vertex
  - Single triangle mesh

**Run PA3 tests:**
```bash
# All PA3 tests
python test_runner.py icp
```

### Run All Tests

```bash
# All tests (PA1 + PA2 + PA3 = 29 tests total)
python test_runner.py all
```

**Available test commands:**
- `python test_runner.py all` - Run all tests
- `python test_runner.py frame` - Run frame transform tests (PA1)
- `python test_runner.py pivot` - Run pivot calibration tests (PA1)
- `python test_runner.py distortion` - Run distortion correction tests (PA2)
- `python test_runner.py icp` - Run ICP matching tests (PA3)

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

### PA3 Algorithms

#### Rigid Body Tracking
- **Method**: Point set registration for two rigid bodies with LED markers
- **Input**:
  - Rigid body definitions (A and B) with marker positions in body coordinates
  - Optical tracker readings of LED markers
- **Steps**:
  1. Register body A markers to tracker coordinates → F_A
  2. Register body B markers to tracker coordinates → F_B
  3. Compute pointer tip in bone frame: d_k = F_B⁻¹ · F_A · A_tip
- **Output**: Pointer tip position in bone coordinate frame

#### Find Closest Point on Triangle
- **Method**: Least squares projection with barycentric coordinates (from lecture slides 11-13)
- **Input**: Query point `a` and triangle vertices `[p, q, r]`
- **Steps**:
  1. Solve least squares: `a - p ≈ λ(q-p) + μ(r-p)` for λ, μ
  2. Compute projection: `c = p + λ(q-p) + μ(r-p)`
  3. Check if inside triangle: `λ ≥ 0 AND μ ≥ 0 AND λ+μ ≤ 1`
     - If inside: return `c`
     - If outside: project onto appropriate edge
  4. Edge projection cases:
     - If `λ < 0`: project onto edge `[r, p]`
     - If `μ < 0`: project onto edge `[p, q]`
     - If `λ+μ > 1`: project onto edge `[q, r]`
- **Output**: Closest point on triangle and distance

#### Find Closest Point on Mesh
- **Method**: Linear search through all triangles (brute force)
- **Input**: Query point and triangular mesh (vertices + triangle indices)
- **Steps**:
  1. For each triangle in mesh:
     - Find closest point on that triangle
     - Track minimum distance
  2. Return closest point across all triangles
- **Output**: Closest point on mesh surface (c_k) and distance
- **Note**: Linear search is acceptable for PA3. Can be optimized with spatial data structures (bounding sphere tree, k-d tree) for larger meshes.

#### ICP Matching (PA3)
- **Method**: Single iteration of ICP matching algorithm (no refinement loop)
- **Input**:
  - Rigid body definitions (A and B)
  - Bone surface mesh
  - Sample readings (LED marker positions)
  - F_reg = Identity (for PA3)
- **Steps**:
  1. Compute rigid body transformations (F_A, F_B)
  2. Compute pointer tip in bone frame: d_k = F_B⁻¹ · F_A · A_tip
  3. Apply F_reg: s_k = F_reg · d_k (for PA3: s_k = d_k since F_reg = I)
  4. Find closest point on mesh: c_k
  5. Compute distance: ||s_k - c_k||
- **Output**: d_k, c_k, and distance for each sample frame
- **Note**: PA4 will add iterative refinement loop to update F_reg

### PA4 Algorithms

#### Complete ICP Algorithm (PA4)
- **Method**: Full Iterative Closest Point algorithm with iterative refinement
- **Input**:
  - Rigid body definitions (A and B)
  - Bone surface mesh
  - Sample readings (LED marker positions)
  - Max iterations and convergence threshold
- **Algorithm**:
  1. Initialize F_reg = Identity
  2. For each iteration (until convergence or max iterations):
     - For each frame k:
       - Compute d_k = F_B_k⁻¹ · F_A_k · A_tip
       - Apply current F_reg: s_k = F_reg · d_k
       - Find closest point on mesh: c_k
     - Update F_reg using point set registration: F_reg_new = Register({d_k}, {c_k})
     - Check convergence: ||F_reg_new - F_reg|| < threshold
     - Update F_reg = F_reg_new
  3. Compute final results with converged F_reg
- **Output**: s_k, c_k, distance, final F_reg, and iteration count
- **Convergence**: Based on change in rotation matrix and translation vector between iterations
- **Key Difference from PA3**: F_reg is iteratively refined instead of remaining identity

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
