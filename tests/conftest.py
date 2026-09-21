"""/ Test-path bootstrap: makes the two importable roots available to the suite.

- src/ — the evalcompare package (standard src layout: putting src/ on sys.path
  exposes exactly one top-level name, the qualified package, so nothing can
  shadow stdlib modules).
- scripts/ — the standalone tool modules (run_eval, gpqa_zip) that tests import
  directly; their names don't collide with stdlib or site-packages either."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
