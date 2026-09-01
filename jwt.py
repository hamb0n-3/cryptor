import base64
import json

import core


JWT_ALG = "XChaCha20-Poly1305"
JWT_TYP = "JWT"


def to_jwt(text: str, password: str) -> str:
    return render(core.encrypt(text.encode('utf-8'), password))


def from_jwt(token: str, password: str) -> str:
    return core.decrypt(parse(token), password).decode('utf-8')


def render(blob: bytes) -> str:
    if len(blob) < core.HEADER_LEN + core.TAG_LEN:
        raise ValueError("blob too short to render as JWT")

    bin_hdr = blob[:core.HEADER_LEN]
    ct = blob[core.HEADER_LEN:-core.TAG_LEN]
    tag = blob[-core.TAG_LEN:]

    version = bin_hdr[0]
    log2_n = bin_hdr[1]
    r = bin_hdr[2]
    p = bin_hdr[3]
    nonce = bin_hdr[4:4 + core.NONCE_LEN]

    json_hdr = {
        "alg": JWT_ALG,
        "typ": JWT_TYP,
        "v": version,
        "kdf": "scrypt",
        "scrypt": {"log2_n": log2_n, "r": r, "p": p},
        "iv": _b64url_encode(nonce),
    }

    return ".".join([
        _b64url_encode(json.dumps(json_hdr, separators=(',', ':')).encode('utf-8')),
        _b64url_encode(ct),
        _b64url_encode(tag),
    ])


def parse(token: str) -> bytes:
    parts = token.split('.')
    if len(parts) != 3:
        raise ValueError(f"not a valid JWT (need 3 dot-separated parts, got {len(parts)})")

    try:
        json_hdr = json.loads(_b64url_decode(parts[0]))
        ct = _b64url_decode(parts[1])
        tag = _b64url_decode(parts[2])
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"malformed JWT structure: {e}") from e

    if not isinstance(json_hdr, dict):
        raise ValueError("JWT header must be a JSON object")
    if json_hdr.get("alg") != JWT_ALG:
        raise ValueError(f"unsupported alg: {json_hdr.get('alg')!r}")

    try:
        version = int(json_hdr["v"])
        scrypt_params = json_hdr["scrypt"]
        log2_n = int(scrypt_params["log2_n"])
        r = int(scrypt_params["r"])
        p = int(scrypt_params["p"])
        nonce = _b64url_decode(json_hdr["iv"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed JWT header fields: {e}") from e

    # First-pass byte-range check so bytes([...]) below can't blow up; core's
    # decrypt() applies the stricter scrypt-param bounds and the AEAD check.
    if not all(0 <= v <= 255 for v in (version, log2_n, r, p)):
        raise ValueError("JWT header byte-sized fields out of range")
    if len(nonce) != core.NONCE_LEN:
        raise ValueError(f"nonce length {len(nonce)} != expected {core.NONCE_LEN}")
    if len(tag) != core.TAG_LEN:
        raise ValueError(f"tag length {len(tag)} != expected {core.TAG_LEN}")

    # Reconstruct the binary header that was used as AEAD AAD when encrypting.
    # Tampering with version / scrypt params / nonce in the JSON header
    # changes these bytes (or the derived key) so the tag check downstream
    # rejects it.
    bin_hdr = bytes([version, log2_n, r, p]) + nonce + len(ct).to_bytes(4, 'big')
    return bin_hdr + ct + tag


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def _b64url_decode(s) -> bytes:
    if isinstance(s, bytes):
        s = s.decode('ascii')
    if not isinstance(s, str):
        raise TypeError(f"expected str/bytes for base64url, got {type(s).__name__}")
    # JWT spec drops base64 padding; re-add it for the stdlib decoder.
    pad = '=' * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)
