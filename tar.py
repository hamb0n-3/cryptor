import io
import os
import tarfile
from pathlib import Path

import core


def _is_tarball(path: str) -> bool:
    try:
        with tarfile.open(path, 'r:*'):
            return True
    except (tarfile.TarError, OSError, EOFError):
        return False


def encrypt_path(src: str, password: str) -> bytes:
    if os.path.isdir(src):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as tf:
            tf.add(src, arcname='.')
        tar_bytes = buf.getvalue()
    elif os.path.isfile(src):
        if not _is_tarball(src):
            raise ValueError(f"{src!r} is a regular file but not a recognizable tarball")
        tar_bytes = Path(src).read_bytes()
    else:
        raise ValueError(f"{src!r} is neither a directory nor a regular file")

    return core.encrypt(tar_bytes, password)


def decrypt_to_path(blob: bytes, password: str, dest: str) -> None:
    if os.path.exists(dest):
        raise ValueError(f"destination {dest!r} already exists; refusing to clobber")

    tar_bytes = core.decrypt(blob, password)

    buf = io.BytesIO(tar_bytes)
    try:
        tf = tarfile.open(fileobj=buf, mode='r:*')
    except tarfile.TarError as e:
        raise ValueError(f"decrypted data is not a valid tar archive: {e}") from e

    try:
        os.makedirs(dest)
        # filter='data' (Python 3.12+) rejects unsafe entries. Calling
        # extractall without a filter is deprecated and becomes an error
        # in 3.14, so this also future-proofs the call.
        tf.extractall(dest, filter='data')
    except tarfile.TarError as e:
        raise ValueError(f"tarball extraction failed: {e}") from e
    finally:
        tf.close()