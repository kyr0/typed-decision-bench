#!/usr/bin/env python3
"""Password-protected storage for gated capability data files (gpqa_diamond).

At rest, each protected file exists only as <name>.jsonl.zip next to its
original path - never as plaintext on disk or in git. Any tool that needs the
jsonl (scripts/run_eval.py, scripts/validate.py, scripts/score.py,
scripts/sync-metadata.py) calls unlock_paths() at start and registers
cleanup_unlocked() with atexit, so the plaintext exists exactly for the
lifetime of the process and is removed (changed copies are re-encrypted
first) when the process ends.

The password is deliberately a public constant: the lock keeps plaintext out
of checkouts/context windows, it is not a secrecy boundary. Set the
GPQA_ZIP_PASSWORD environment variable to override it, e.g. for privately
re-encrypted copies.

Archives use classic PKZIP ZipCrypto because the stdlib can decrypt such
entries but not create them - so no third-party dependency is needed for
reading (zipfile) or writing (this module).
"""
import argparse
import os
import struct
import sys
import time
import zipfile
import zlib
from pathlib import Path

# public default password (see module docstring); override via GPQA_ZIP_PASSWORD
PASSWORD = 'typed-decision-bench-gpqa'
ENV_OVERRIDE = 'GPQA_ZIP_PASSWORD'
# the slugs under protection today; extend here when more suites get gated
PROTECTED_SLUGS = ('gpqa_diamond',)
KINDS = ('requests', 'responses', 'metadata')

# ZipCrypto initial key state (PKWARE APPNOTE, section 6.1)
_K0, _K1, _K2 = 305419896, 591751049, 878082192
# PKZIP CRC-32 table (poly 0xEDB88320). Raw registers, NOT zlib.crc32(): zlib's
# incremental API wraps each update in XOR 0xFFFFFFFF, which corrupts the keys.
_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ (0xEDB88320 if _c & 1 else 0)
    _CRC_TABLE.append(_c)


def zip_password() -> str:
    """/ Single source for the archive password; the env var beats the public default."""
    return os.environ.get(ENV_OVERRIDE, PASSWORD)


def _crc32_byte(value: int, byte: int) -> int:
    """/ Feeds one byte into a running raw CRC-32 register (APPNOTE crc32() macro)."""
    return (value >> 8) ^ _CRC_TABLE[(value ^ byte) & 0xFF]


class _ZipCrypto:
    """/ Classic PKZIP keystream cipher (APPNOTE 6.1), write-side only: the stdlib
    decrypts ZipCrypto entries but cannot create them, so this fills that gap."""

    def __init__(self, password: str):
        self.keys = (_K0, _K1, _K2)
        for b in password.encode('utf-8'):
            self._update(b)

    def _update(self, plain: int) -> None:
        k0, k1, k2 = self.keys
        k0 = _crc32_byte(k0, plain)
        k1 = ((k1 + (k0 & 0xFF)) * 134775813 + 1) & 0xFFFFFFFF
        self.keys = (k0, k1, _crc32_byte(k2, k1 >> 24))

    def encrypt(self, data: bytes) -> bytes:
        """/ XORs each byte with the keystream, then advances the keys with that
        plaintext byte - the exact order readers expect when decrypting."""
        out = bytearray(len(data))
        for i, p in enumerate(data):
            temp = (self.keys[2] | 2) & 0xFFFF
            kbyte = ((temp * (temp ^ 1)) >> 8) & 0xFF
            out[i] = p ^ kbyte
            self._update(p)
        return bytes(out)


def encrypted_zip_bytes(name: str, data: bytes, password: str) -> bytes:
    """/ Builds a complete single-member zip with a ZipCrypto-encrypted deflate entry.

    Everything is built in memory (capability files are a few hundred KB max),
    so CRC and sizes are known upfront: the general-purpose flag is 0x1
    (encrypted) only, and byte 12 of the encryption header is crc>>24 - the
    exact check value stdlib zipfile and Info-ZIP unzip verify on decrypt.
    """
    deflater = zlib.compressobj(9, zlib.DEFLATED, -15)
    compressed = deflater.compress(data) + deflater.flush()
    crc = zlib.crc32(data) & 0xFFFFFFFF
    t = time.localtime()
    dos_time = (t.tm_hour << 11) | (t.tm_min << 5) | (t.tm_sec // 2)
    dos_date = ((t.tm_year - 1980) << 9) | (t.tm_mon << 5) | t.tm_mday
    cipher = _ZipCrypto(password)
    # 11 random bytes + the CRC check byte, then the compressed data, one keystream
    payload = cipher.encrypt(os.urandom(11) + bytes(((crc >> 24) & 0xFF,)) + compressed)
    nb = name.encode('utf-8')
    local = struct.pack('<IHHHHHIIIHH', 0x04034B50, 20, 0x1, 8, dos_time, dos_date,
                        crc, len(payload), len(data), len(nb), 0) + nb
    central = struct.pack('<IHHHHHHIIIHHHHHII', 0x02014B50, 0x031E, 20, 0x1, 8, dos_time,
                          dos_date, crc, len(payload), len(data), len(nb), 0, 0, 0, 0,
                          0o600 << 16, 0) + nb
    # the central directory sits after the local header + encrypted payload
    eocd = struct.pack('<IHHHHIIH', 0x06054B50, 0, 0, 1, 1, len(central), len(local) + len(payload), 0)
    return local + payload + central + eocd


def _log(msg: str) -> None:
    """/ Logs to stderr on purpose: `make validate` redirects stdout into a JSON file."""
    print(f'[gpqa] {msg}', file=sys.stderr, flush=True)


class UnlockedFile:
    """/ Bookkeeping for one extracted file: origin zip plus the fingerprint
    (mtime_ns, size) the freshly written file had, so cleanup can tell an
    untouched copy (just delete) from a modified one (re-encrypt, then delete)."""

    def __init__(self, target: Path, zip_path: Path, fingerprint: tuple):
        self.target = target
        self.zip_path = zip_path
        self.fingerprint = fingerprint


def _write_zip(zip_path: Path, name: str, data: bytes, password: str) -> None:
    """/ Atomic archive write (tmp + rename) so an interrupted relock can never
    leave a half-written zip behind."""
    tmp = zip_path.with_name(zip_path.name + '.tmp')
    tmp.write_bytes(encrypted_zip_bytes(name, data, password))
    os.chmod(tmp, 0o600)
    os.replace(tmp, zip_path)


def lock_file(target: Path, password: str = None) -> bool:
    """/ Encrypts target into <target>.zip (same folder) and removes the plaintext
    file. Returns True when a lock happened, False when there was nothing to lock."""
    if not target.exists():
        return False
    _write_zip(Path(str(target) + '.zip'), target.name, target.read_bytes(),
               password or zip_password())
    target.unlink()
    return True


def unlock_file(target: Path, password: str = None):
    """/ Materialises target from its .zip sibling when needed (target missing,
    zip present). Returns an UnlockedFile for cleanup_unlocked(), or None when
    nothing was extracted. A plaintext file that already exists is left alone
    and deliberately NOT tracked, so cleanup never deletes files it did not create."""
    zip_path = Path(str(target) + '.zip')
    if target.exists():
        if zip_path.exists():
            _log(f'WARNING {target}: plaintext next to {zip_path.name} - leaving both, not tracked for cleanup')
        return None
    if not zip_path.exists():
        return None
    pwd = (password or zip_password()).encode('utf-8')
    with zipfile.ZipFile(zip_path) as zf:
        member = next((n for n in zf.namelist() if n == target.name), zf.namelist()[0])
        data = zf.read(member, pwd=pwd)
    target.write_bytes(data)
    os.chmod(target, 0o600)
    # ponytail: mtime_ns+size change detection misses a rewrite that lands within
    # coarse mtime granularity (2s FAT); upgrade path: hash the content instead
    st = target.stat()
    return UnlockedFile(target, zip_path, (st.st_mtime_ns, st.st_size))


def unlock_paths(paths, password: str = None) -> list:
    """/ unlock_file() over many paths (idempotent no-op where no zip exists);
    returns only the handles of files that were actually extracted."""
    unlocked = [u for p in paths if (u := unlock_file(Path(p), password)) is not None]
    for u in unlocked:
        _log(f'unlocked {u.target} (removing at exit)')
    return unlocked


def cleanup_file(f: UnlockedFile, password: str = None) -> None:
    """/ End-of-run handler for one extracted file: removes the plaintext copy,
    refreshing the zip first when the file changed since extraction (e.g.
    run_eval.py overwrote responses/gpqa_diamond.jsonl) so no data is lost."""
    if not f.target.exists():
        return
    st = f.target.stat()
    if (st.st_mtime_ns, st.st_size) != f.fingerprint:  # changed since extraction
        _write_zip(f.zip_path, f.target.name, f.target.read_bytes(), password or zip_password())
        _log(f'relocked changed {f.zip_path}')
    f.target.unlink()


def cleanup_unlocked(unlocked: list, password: str = None) -> None:
    """/ Removes/re-encrypts everything unlock_paths() extracted; safe to register
    with atexit (runs on normal exit, exceptions and KeyboardInterrupt)."""
    for f in unlocked:
        cleanup_file(f, password)


def default_paths(root: str = '.') -> list:
    """/ The files under protection today: <kind>/<slug>.jsonl per protected slug."""
    r = Path(root)
    return [r / kind / f'{slug}.jsonl' for slug in PROTECTED_SLUGS for kind in KINDS]


def main() -> None:
    """/ CLI: `lock` / `unlock` / `status`, optionally with repeatable --file paths
    (default: the PROTECTED_SLUGS x KINDS trio). Status reports locked / unlocked /
    plaintext / missing per file without changing anything."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for cmd in ('lock', 'unlock', 'status'):
        p = sub.add_parser(cmd, help=f'{cmd} capability files')
        p.add_argument('--file', action='append', default=None, metavar='JSONL',
                       help='path to operate on (repeatable); default: ' +
                            ', '.join(PROTECTED_SLUGS) + ' requests/responses/metadata trio')
    a = ap.parse_args()
    paths = [Path(f) for f in (a.file or [])] or default_paths()
    if a.cmd == 'lock':
        for p in paths:
            _log(f'locked {p}' if lock_file(p) else f'skipped {p} (no plaintext file)')
    elif a.cmd == 'unlock':
        unlock_paths(paths)
    else:
        for p in paths:
            z = Path(str(p) + '.zip')
            state = ('locked' if not p.exists() else 'unlocked (both present)') if z.exists() \
                else ('plaintext' if p.exists() else 'missing')
            _log(f'{p}: {state}')


if __name__ == '__main__':
    main()
