from secrets import token_bytes
from hashlib import scrypt

from nacl.bindings import (
    crypto_aead_xchacha20poly1305_ietf_encrypt,
    crypto_aead_xchacha20poly1305_ietf_decrypt,
    crypto_aead_xchacha20poly1305_ietf_KEYBYTES,
    crypto_aead_xchacha20poly1305_ietf_NPUBBYTES,
    crypto_aead_xchacha20poly1305_ietf_ABYTES,
)
from nacl.exceptions import CryptoError


VERSION = 2                   # bumped: incompatible with v1 (HMAC-SHA256) blobs
SCRYPT_LOG2_N = 17            # n = 2**17 = 131072 (OWASP minimum)
SCRYPT_R = 8
SCRYPT_P = 1
NONCE_LEN = crypto_aead_xchacha20poly1305_ietf_NPUBBYTES    # 24
KEY_LEN = crypto_aead_xchacha20poly1305_ietf_KEYBYTES       # 32
TAG_LEN = crypto_aead_xchacha20poly1305_ietf_ABYTES         # 16
HEADER_LEN = 4 + NONCE_LEN + 4                              # 4 params + nonce + length = 32
BLOB_OVERHEAD = HEADER_LEN + TAG_LEN                        # bytes added by encrypt()


def _derive_key(password: str, nonce: bytes, log2_n: int, r: int, p: int) -> bytes:
    n = 1 << log2_n
    # scrypt memory cost is ~128 * N * r bytes; give 2x headroom, clamped to
    # hashlib.scrypt's signed-int32 cap on maxmem.
    maxmem = min(256 * n * r, (1 << 31) - 2)
    return scrypt(password.encode(), salt=nonce, n=n, r=r, p=p,
                  dklen=KEY_LEN, maxmem=maxmem)


def encrypt(plaintext: bytes, password: str) -> bytes:
    nonce = token_bytes(NONCE_LEN)  # 192-bit nonce, also used as scrypt salt
    key = _derive_key(password, nonce, SCRYPT_LOG2_N, SCRYPT_R, SCRYPT_P)

    hdr = (bytes([VERSION, SCRYPT_LOG2_N, SCRYPT_R, SCRYPT_P])
           + nonce + len(plaintext).to_bytes(4, 'big'))
    # XChaCha20-Poly1305 returns ciphertext || tag; the header is AAD so
    # any tampering with version, scrypt params, or length is detected.
    ct_and_tag = crypto_aead_xchacha20poly1305_ietf_encrypt(
        plaintext, hdr, nonce, key,
    )
    return hdr + ct_and_tag


def expected_blob_len(prefix: bytes) -> int:
    if len(prefix) < HEADER_LEN:
        raise ValueError(f"need at least {HEADER_LEN} bytes to read header, got {len(prefix)}")
    L = int.from_bytes(prefix[4 + NONCE_LEN:HEADER_LEN], 'big')
    return HEADER_LEN + L + TAG_LEN


def decrypt(blob: bytes, password: str) -> bytes:
    if len(blob) < HEADER_LEN + TAG_LEN:
        raise ValueError("ciphertext too short")

    version, log2_n, r, p = blob[0], blob[1], blob[2], blob[3]
    if version != VERSION:
        raise ValueError(f"unsupported format version: {version}")
    # Bound scrypt params so a malicious blob can't request absurd memory.
    if not (10 <= log2_n <= 20):
        raise ValueError(f"unreasonable scrypt log2_n: {log2_n}")
    if not (1 <= r <= 32) or not (1 <= p <= 16):
        raise ValueError("unreasonable scrypt r/p parameters")

    nonce = blob[4:4 + NONCE_LEN]
    L = int.from_bytes(blob[4 + NONCE_LEN:HEADER_LEN], 'big')

    if len(blob) != HEADER_LEN + L + TAG_LEN:
        raise ValueError("blob length doesn't match length field")

    hdr = blob[:HEADER_LEN]
    ct_and_tag = blob[HEADER_LEN:]

    key = _derive_key(password, nonce, log2_n, r, p)
    try:
        return crypto_aead_xchacha20poly1305_ietf_decrypt(
            ct_and_tag, hdr, nonce, key,
        )
    except CryptoError as e:
        raise ValueError("authentication failed") from e