import os
import subprocess
import sys


def test_importing_bergson_turns_off_numpy_huge_pages():
    env = {k: v for k, v in os.environ.items() if k != "NUMPY_MADVISE_HUGEPAGE"}
    code = (
        "import bergson, numpy; "
        "print(numpy._core.multiarray._set_madvise_hugepage(True))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    # Dependencies may log warnings to stdout on import; the print is the last line.
    assert out.stdout.strip().splitlines()[-1] == "False"
