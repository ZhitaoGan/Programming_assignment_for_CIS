"""
Test runner and configuration for CISPA project

Provides easy ways to run different types of tests and generate reports.
"""

import pytest
import sys
import os
from pathlib import Path


def run_unit_tests():
    """Run only unit tests"""
    print("Running unit tests...")
    return pytest.main([
        "programs/tests/",
        "-m", "unit",
        "-v",
        "--tb=short"
    ])


def run_integration_tests():
    """Run only integration tests"""
    print("Running integration tests...")
    return pytest.main([
        "programs/tests/",
        "-m", "integration",
        "-v",
        "--tb=short"
    ])


def run_all_tests():
    """Run all tests"""
    print("Running all tests...")
    return pytest.main([
        "programs/tests/",
        "-v",
        "--tb=short"
    ])


def run_fast_tests():
    """Run only fast tests (exclude slow tests)"""
    print("Running fast tests...")
    return pytest.main([
        "programs/tests/",
        "-m", "not slow",
        "-v",
        "--tb=short"
    ])


def run_with_coverage():
    """Run tests with coverage report"""
    print("Running tests with coverage...")
    return pytest.main([
        "programs/tests/",
        "--cov=programs",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ])


def run_specific_module(module_name):
    """Run tests for a specific module"""
    print(f"Running tests for {module_name}...")
    return pytest.main([
        f"programs/tests/test_{module_name}.py",
        "-v",
        "--tb=short"
    ])


def main():
    """Main test runner"""
    if len(sys.argv) < 2:
        print("Usage: python test_runner.py <command>")
        print("Commands:")
        print("  unit        - Run unit tests only")
        print("  integration - Run integration tests only")
        print("  all         - Run all tests")
        print("  fast        - Run fast tests only")
        print("  coverage    - Run tests with coverage")
        print("  frame       - Run frame transform tests")
        print("  pivot       - Run pivot calibration tests")
        print("  utility     - Run utility function tests")
        return
    
    command = sys.argv[1].lower()
    
    if command == "unit":
        exit_code = run_unit_tests()
    elif command == "integration":
        exit_code = run_integration_tests()
    elif command == "all":
        exit_code = run_all_tests()
    elif command == "fast":
        exit_code = run_fast_tests()
    elif command == "coverage":
        exit_code = run_with_coverage()
    elif command == "frame":
        exit_code = run_specific_module("frame_transform")
    elif command == "pivot":
        exit_code = run_specific_module("pivot_calibration")
    elif command == "utility":
        exit_code = run_specific_module("utility_functions")
    else:
        print(f"Unknown command: {command}")
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
