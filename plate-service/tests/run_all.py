"""
run_all.py
--------------------------------------------------------------------
Runs every test module in this directory and exits non-zero on any
failure — for CI or a quick local sanity check.

    python3 tests/run_all.py
--------------------------------------------------------------------
"""

import os
import sys
import unittest

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=here, pattern="test_*.py", top_level_dir=os.path.dirname(here))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
