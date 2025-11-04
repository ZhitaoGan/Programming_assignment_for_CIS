"""
Test runner and configuration for CISPA project

Provides easy ways to run different types of tests.
"""

import pytest
import sys


def run_all_tests():
    """Run all tests"""
    print("Running all tests...")
    return pytest.main([
        "programs/tests/",
        "-v",
        "--tb=short"
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
        print("  all         - Run all tests")
        print("  frame       - Run frame transform tests")
        print("  pivot       - Run pivot calibration tests")
        return

    command = sys.argv[1].lower()

    if command == "all":
        exit_code = run_all_tests()
    elif command == "frame":
        exit_code = run_specific_module("frame_transform")
    elif command == "pivot":
        exit_code = run_specific_module("pivot_calibration")
    else:
        print(f"Unknown command: {command}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
