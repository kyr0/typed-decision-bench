"""/ Tests for scripts/gpqa_zip.py: ZipCrypto interop with independent readers
(stdlib zipfile + Info-ZIP unzip when present) plus the full lock/unlock/cleanup
lifecycle including the relock-if-changed path and the wrong-password guard.

Run via `make test` (conftest.py puts scripts/ on sys.path for the import).
"""
import shutil
import subprocess
import zipfile
from pathlib import Path

import gpqa_zip as gz

DATA = ('{"questions": {"q1": "Why? ↔ äöü"}, "answers": {"q1": {"type": "noul", "noul": 0.5}}}\n' * 500).encode('utf-8')


def test_stdlib_interop(tmp_path: Path) -> None:
    """/ Our hand-built ZipCrypto zip must decrypt byte-identically with the stdlib,
    and a wrong password must be rejected (stdlib raises RuntimeError)."""
    zp = tmp_path / 'interop.zip'
    zp.write_bytes(gz.encrypted_zip_bytes('x.jsonl', DATA, 'pw'))
    with zipfile.ZipFile(zp) as zf:
        assert zf.namelist() == ['x.jsonl'], zf.namelist()
        assert zf.read('x.jsonl', pwd=b'pw') == DATA
        try:
            zf.read('x.jsonl', pwd=b'wrong')
            raise AssertionError('wrong password was accepted')
        except RuntimeError:
            pass
    # cross-check against Info-ZIP unzip when available (fully independent implementation)
    if shutil.which('unzip'):
        out = subprocess.run(['unzip', '-P', 'pw', '-p', str(zp)], capture_output=True, check=True)
        assert out.stdout == DATA, 'Info-ZIP unzip round-trip differs'


def test_lifecycle(tmp_path: Path) -> None:
    """/ lock -> unlock -> cleanup removes the plaintext; a changed file is
    re-encrypted into the zip before removal (no data loss)."""
    target = tmp_path / 'gpqa_diamond.jsonl'
    target.write_bytes(DATA)
    assert gz.lock_file(target) and not target.exists() and Path(str(target) + '.zip').exists()

    # untouched copy: cleanup just deletes it, zip keeps the original content
    unlocked = gz.unlock_paths([target])
    assert len(unlocked) == 1 and target.read_bytes() == DATA
    gz.cleanup_unlocked(unlocked)
    assert not target.exists()

    # modified copy (e.g. eval overwrote it): cleanup relocks the new content
    unlocked = gz.unlock_paths([target])
    changed = DATA + b'{"extra": true}\n'
    target.write_bytes(changed)
    gz.cleanup_unlocked(unlocked)
    assert not target.exists()
    unlocked2 = gz.unlock_paths([target])
    assert target.read_bytes() == changed
    gz.cleanup_unlocked(unlocked2)

    # idempotence: unlocking a path without a zip does nothing
    plain = tmp_path / 'other.jsonl'
    plain.write_bytes(b'x\n')
    assert gz.unlock_paths([plain]) == []


def test_wrong_password_guard(tmp_path: Path) -> None:
    """/ A locked zip that does not match the configured password must fail loudly
    rather than silently producing garbage."""
    zp = tmp_path / 'gpqa_diamond.jsonl.zip'
    zp.write_bytes(gz.encrypted_zip_bytes('gpqa_diamond.jsonl', DATA, 'other-pw'))
    try:
        gz.unlock_file(tmp_path / 'gpqa_diamond.jsonl', password='pw')
        raise AssertionError('mismatched password was not detected')
    except RuntimeError:
        pass
