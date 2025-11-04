'''
Created on Oct 20, 2022
@author: Trevor Schwehr and Idris Sunmola
, 0\n"  Main file (Executable program) serving as interface for the project. Access to the solutions for
        problems 4a, 4b, 4c, 4d, 5, and 6 are provided below.

@summary: Main file that provides access to the utility functions for the project. 
'''
import pdb
from codecs import utf_16_encode
from pathlib import Path
import readline
import programs
import click
import logging
import itertools
import ast
import json

from rich.logging import RichHandler
import numpy as np
from programs import FrameTransform, pivot_calibration, utility_functions

logging.basicConfig(
    filename='./logging/Fd_reg.log',
    level="ERROR",
    format="%(message)s",
    datefmt="[%X]",
    #handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)


@click.command()
@click.option("--data_dir", "-d", default="PA1 Student Data", help="Input data directory")
@click.option("--output_dir", "-o", default="output", help="Output directory")
@click.option("--name", "-n", default="pa1-debug-a-calbody", help="Name of the input file")
@click.option("--name_2", "-n2", default="pa1-debug-a-calreadings", help="Name of the second input file")
@click.option("--name_3", "-n3", default="pa1-debug-b-empivot", help="Name of EM Pivot file")
@click.option("--name_4", "-n4", default="pa1-debug-g-optpivot", help="Name of OPT Pivot file")
@click.option("--output_file", "-of", help="Name of the output file")
@click.option("--output_file1", "-of1", help="Name of the output file")
@click.option("--output_file2", "-of2", help="Name of the output file")
@click.option("--input_reg", "-ir", help="Name of the registration file")
@click.option("--input_reg2", "-ir2", default="", help="Name of the second registration file")

def main(data_dir, output_dir, name, name_2, name_3, name_4, output_file, output_file1, output_file2, input_reg, input_reg2):
    #log.info(f"data_dir, output_dir")
    

    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    cal_path = data_dir / f"{name}"
    calreadings = data_dir / f"{name_2}"
    em_path = data_dir / f"{name_3}"
    opt_path = data_dir / f"{name_4}"
    cal_body = utility_functions.read_cal_data(
        f"{cal_path}.txt"
    )

    cal_read = utility_functions.read_cal_data(
        f"{calreadings}.txt"
    )

    empivot = utility_functions.read_pivot_data(
        f"{em_path}.txt"
    )

    optpivot = utility_functions.read_optpivot(
        f"{opt_path}.txt"
    )
    
    
    if not output_dir.exists():
        output_dir.mkdir()

    """ Solutions to 4a, 4b, 4c, 4d, 5, and 6

        To generated the Fa frame for points a, run:
        python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fa_a_registration

        To generate the Fd frame for points a, run:
        python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fd_a_registration

        To generate the EM pivot for points a, run:
        python pa1.py --name_3 pa1-debug-a-empivot --output_file1 A_EM_pivot

        To generate the Opt pivot for points a, run:
        python pa1.py --name pa1-debug-a-calbody --name_4 pa1-debug-a-optpivot --output_file2 A_Optpivot 
    """ 
    if not output_file1 and not output_file2 and not \
        input_reg and name and name_2 and (not input_reg or not input_reg2):
        try:
            utility_functions.parse_files(output_file, cal_read, cal_body)
        except BaseException as err:
            log.error(f"Unexpected {err=}, {type(err)=}")
            raise
    elif input_reg and input_reg2:
    #python pa1.py --name pa1-debug-b-calbody --input_reg Fa_b_registration --input_reg2 Fd_b_registration --output_file pa1-debug-b-output1
        try:
            utility_functions.C_expected(cal_body, output_file, input_reg, input_reg2)
        except BaseException as err:
            log.error(f"Unexpected {err=}, {type(err)=}",exc_info=True)
            raise
    #Question 5
    #To run:
    #python pa1.py --name_3 pa1-debug-a-empivot --output_file1 A_EM_pivot
    elif em_path and output_file1:
        try:
            utility_functions.em_pivot(empivot, output_file1)
        except BaseException as err:
            log.error(f"Unexpected {err=}, {type(err)=}")
            raise
    #Question 6
    #To run:
    #python pa1.py --name pa1-debug-a-calbody --name_4 pa1-debug-a-optpivot --output_file2 A_Optpivot 
    elif opt_path and output_file2:
        try:
            utility_functions.opt_pivot(optpivot, cal_body, output_file2)
        except BaseException as err:
            log.error(f"Unexpected {err=}, {type(err)=}")
            raise
    else:
        print("Input correct terminal arguments and try again!")


if __name__ == "__main__":
    main()
