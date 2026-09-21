"""Makes the flat src/ modules importable as the `evalcompare` package.
src/ itself never goes on sys.path — its io.py would shadow the stdlib io."""
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if "evalcompare" not in sys.modules:
    _pkg = types.ModuleType("evalcompare")
    _pkg.__path__ = [str(SRC)]
    sys.modules["evalcompare"] = _pkg
