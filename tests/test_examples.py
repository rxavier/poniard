"""Examples must run end-to-end so they can never silently go stale."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted(Path("examples").glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.stem)
def test_examples_run(script):
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"{script} failed:\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
