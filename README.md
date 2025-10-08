# cispa

Template for CIS I programming assignments at Johns Hopkins. [Here](https://ciis.lcsr.jhu.edu/doku.php?id=courses:455-655:455-655) is the course page.

You may need to modify this README to contain the proper urls for your repository.

If you use this template or any of the code within for your project, please cite

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

### Setup

1. **Install Dependencies**: Create and activate the conda environment:
   ```bash
   cd cispa
   conda env create -f environment.yml
   conda activate cispa
   
   # Create necessary directories
   mkdir -p logging
   mkdir -p output
   ```

2. **Verify Installation**: Run the tests to ensure everything is working:
   ```bash
   pytest -s
   ```

### Running the Program

The main program `pa1.py` provides several functionalities for different problems:

#### Complete Workflow (Problems 4a, 4b, 4c, 5, 6)

**Step 1: Generate Frame Registrations (Problems 4a & 4b)**
```bash
# Generate Fa frame registration (A to a points)
python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fa_a_registration --output_dir output

# Generate Fd frame registration (D to d points)
python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fd_a_registration --output_dir output
```

**Note**: You may see error messages about missing optional files (`pa1-debug-b-empivot.txt`, `pa1-debug-g-optpivot.txt`). These are harmless and don't affect the main functionality.

**Step 2: Perform Pivot Calibrations (Problems 5 & 6)**
```bash
# EM pivot calibration
python pa1.py --name_3 pa1-debug-a-empivot --output_file1 A_EM_pivot --output_dir output

# Optical pivot calibration
python pa1.py --name pa1-debug-a-calbody --name_4 pa1-debug-a-optpivot --output_file2 A_Optpivot --output_dir output
```

**Step 3: Generate Expected C Coordinates (Problem 4c)**
```bash
# Generate NAME-OUTPUT1.TXT format file with expected C coordinates
python pa1.py --name pa1-debug-a-calbody --input_reg Fa_a_registration --input_reg2 Fd_a_registration --output_file pa1-debug-a-output1 --output_dir output
```

#### Individual Problem Commands

**Problem 4a: Fa Frame Registration**
```bash
python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fa_a_registration --output_dir output
```

**Problem 4b: Fd Frame Registration**
```bash
python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fd_a_registration --output_dir output
```

**Problem 5: EM Pivot Calibration**
```bash
python pa1.py --name_3 pa1-debug-a-empivot --output_file1 A_EM_pivot --output_dir output
```

**Problem 6: Optical Pivot Calibration**
```bash
python pa1.py --name pa1-debug-a-calbody --name_4 pa1-debug-a-optpivot --output_file2 A_Optpivot --output_dir output
```

**Problem 4c: Expected C Coordinates**
```bash
python pa1.py --name pa1-debug-a-calbody --input_reg Fa_a_registration --input_reg2 Fd_a_registration --output_file pa1-debug-a-output1 --output_dir output
```

### Input Data Format

All input files use comma-separated values (CSV) format with the following structure:

- **CALBODY.TXT**: `N_D, N_A, N_C, filename` (header) followed by coordinate data
- **CALREADINGS.TXT**: `N_D, N_A, N_C, N_frames, filename` (header) followed by multi-frame data
- **EMPIVOT.TXT**: `N_G, N_frames, filename` (header) followed by EM probe marker data
- **OPTPIVOT.TXT**: `N_D, N_H, N_frames, filename` (header) followed by optical tracking data

All coordinates are in millimeters (mm).

### Sample Data

Sample input data files are provided in the `PA1 Student Data/` directory:
- `pa1-debug-a-CALBODY.TXT`
- `pa1-debug-a-CALREADINGS.TXT` 
- `pa1-debug-a-EMPIVOT.TXT`
- `pa1-debug-a-OPTPIVOT.TXT`

**Note**: The sample files use `.TXT` extension (uppercase), but the program accepts both `.txt` and `.TXT` extensions.

### Output Files

The program generates output files in the `output/` directory (created automatically if it doesn't exist):

#### Intermediate Output Files
- `output/Fa_a_registration.txt` - Fa frame registration matrices (4x4 transformation matrices flattened to 16 values per frame)
- `output/Fd_a_registration.txt` - Fd frame registration matrices (4x4 transformation matrices flattened to 16 values per frame)
- `output/A_EM_pivot.txt` - EM pivot calibration results (tip position, pivot point, residual error)
- `output/A_Optpivot.txt` - Optical pivot calibration results (tip position, pivot point, residual error)

#### Final Output File (NAME-OUTPUT1.TXT Format)
- `output/pa1-debug-a-output1.txt` - Complete output file following NAME-OUTPUT1.TXT specification:
  - **Line 1**: `N_C, N_frames, filename` (header with marker count, frame count, filename)
  - **Line 2**: `P_x, P_y, P_z` (EM probe pivot calibration tip position)
  - **Line 3**: `P_x, P_y, P_z` (Optical probe pivot calibration tip position)
  - **Lines 4+**: Expected C coordinates for each frame (N_C markers × N_frames)

#### Output File Format Details
The final output file follows the exact NAME-OUTPUT1.TXT specification:
```
3, 5, pa1-debug-a-output1.txt
-8.000, -2.500, 0.000
-6.000, -2.250, 0.250
-15.050, -15.197, 14.909
14.950, -15.287, 14.939
0.040, 14.758, 14.924
...
```
Where each frame contains N_C lines of C_x, C_y, C_z coordinates for the expected EM marker positions.

### Unit Tests

Use [pytest](https://docs.pytest.org/en/6.2.x/). In the `tests/` directory, place `.py` files that
start with `test_`, and contain functions that start with `test_`. Then use `assert` statements to
evaluate parts of your code.

Run all tests with

```sh
pytest -s
```

Or focus on a particular test, for example a function called `test_registration()` in `test_frame.py`:

```sh
pytest -s tests/test_frame.py::test_registration
```

The `-s` option tells pytest to allow `print` statements and other logging (use
[logging](https://docs.python.org/3/library/logging.html)!) to be passed through.
