"""
Test runner script for GateID Vision.
Executes pytest if available, or seamlessly delegates to unittest discover.
"""
import sys

def main():
    try:
        import pytest
        print("Running tests via pytest...")
        sys.exit(pytest.main(["tests", "-v"]))
    except ImportError:
        import unittest
        print("pytest not installed; running tests via Python standard library unittest...")
        loader = unittest.TestLoader()
        suite = loader.discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
