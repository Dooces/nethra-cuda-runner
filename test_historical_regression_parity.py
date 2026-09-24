import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

BASE_COMMIT = "8abeaf358b34aed6b18e10cc0f7fc17de6e7db93"
LEGACY_TESTS = (
    "test_interval_boundary_freeze.py",
    "test_subtraction_before_construction.py",
)
STATUS_RE = re.compile(r"^(test_[^( ]+ .*?) \.\.\. (ok|FAIL|ERROR)$")


def run_suite(nethra_source):
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "nethra.py").write_text(nethra_source)
        here = pathlib.Path(__file__).resolve().parent
        for name in LEGACY_TESTS:
            shutil.copy2(here / name, root / name)
        proc = subprocess.run(
            ["python3", "-m", "unittest", "-v", *LEGACY_TESTS],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        statuses = {}
        for line in proc.stdout.splitlines():
            match = STATUS_RE.match(line.strip())
            if match:
                statuses[match.group(1)] = match.group(2)
        return proc.returncode, statuses, proc.stdout


class HistoricalRegressionParityTests(unittest.TestCase):
    def test_historical_suite_has_identical_status_map_to_frozen_commit(self):
        baseline_source = subprocess.check_output(
            ["git", "show", f"{BASE_COMMIT}:nethra.py"],
            text=True,
        )
        optimized_source = pathlib.Path("nethra.py").read_text()

        base_rc, base_status, base_out = run_suite(baseline_source)
        opt_rc, opt_status, opt_out = run_suite(optimized_source)

        self.assertTrue(base_status, base_out)
        self.assertEqual(base_status, opt_status, opt_out)
        self.assertEqual(base_rc == 0, opt_rc == 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
