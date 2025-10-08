# CISPA Testing Suite

This directory contains unit tests for the core algorithms in the CISPA (Computer Integrated Surgery 1) project.

## Test Structure

```
programs/tests/
├── test_frame_transform.py        # Tests for FrameTransform class and Point Set Registration
└── test_pivot_calibration.py     # Tests for pivot calibration algorithms
```

## Core Algorithms Tested

### FrameTransform Class (`test_frame_transform.py`)
- **Point Set Registration Algorithm**: Tests the core registration algorithm that aligns two point sets
- **Basic Transform Operations**: Tests point transformation, inverse operations, and composition
- **Mathematical Properties**: Validates rotation matrices, determinant properties, and orthogonality

### Pivot Calibration (`test_pivot_calibration.py`)
- **solve_for_pivot**: Tests the core least squares algorithm for pivot point calculation
- **em_pivot_calibration**: Tests EM (Electromagnetic) pivot calibration algorithm
- **opt_pivot_calibration**: Tests optical pivot calibration algorithm
- **Algorithm Consistency**: Validates that EM and optical methods produce consistent results

## Running Tests

### Using pytest Directly

```bash
# Run all tests
pytest programs/tests/

# Run specific test file
pytest programs/tests/test_frame_transform.py
pytest programs/tests/test_pivot_calibration.py

# Run with verbose output
pytest -v programs/tests/

# Run specific test
pytest programs/tests/test_frame_transform.py::TestPointSetRegistration::test_point_set_registration_identity
```

## Test Coverage

The test suite focuses on validating the core algorithms:

### Point Set Registration Algorithm
- ✅ Identity transformation (identical point sets)
- ✅ Pure translation recovery
- ✅ Pure rotation recovery
- ✅ General transformation (rotation + translation)
- ✅ Weighted point registration
- ✅ Reflection case handling (ensures proper rotation matrices)

### Pivot Calibration Algorithms
- ✅ EM pivot calibration with multiple frames
- ✅ Optical pivot calibration with different input formats
- ✅ Least squares solving for tip and pivot points
- ✅ Residual error calculation
- ✅ Input validation and error handling
- ✅ Consistency between EM and optical methods

### Frame Transform Operations
- ✅ Point transformation (single and multiple points)
- ✅ Inverse transformation calculation
- ✅ Transformation composition
- ✅ Mathematical properties validation

## Dependencies

The test suite requires:
- `pytest` - Testing framework
- `numpy` - Numerical computations

Install with:
```bash
pip install pytest numpy
```

## Test Philosophy

These tests focus on validating the mathematical correctness of the core algorithms rather than testing file I/O or utility functions. The goal is to ensure that:

1. **Point Set Registration** correctly aligns point sets using the weighted least squares approach
2. **Pivot Calibration** accurately determines tool tip and pivot points using least squares
3. **Frame Transforms** maintain mathematical properties (orthogonality, proper composition, etc.)

Each test uses synthetic data with known ground truth to validate algorithm correctness.