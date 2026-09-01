from __future__ import annotations

import hashlib

import numpy as np

import jpeglib
import stc
import core


VAR_WINDOW = 3            # half-width of local-variance neighborhood (→ 7×7 window)
VAR_FLOOR = 1.0           # σ² lower bound; bounds ρ in nearly-flat regions
PERMUTATION_INFO = b'stego:jmipod:permutation-v1'
DIRECTION_INFO = b'stego:jmipod:direction-v1'


def _idct_basis() -> np.ndarray:
    x = np.arange(8)
    basis = np.zeros((8, 8, 8, 8), dtype=np.float64)
    for u in range(8):
        cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
        cos_u = np.cos((2 * x + 1) * u * np.pi / 16)
        for v in range(8):
            cv = 1.0 / np.sqrt(2) if v == 0 else 1.0
            cos_v = np.cos((2 * x + 1) * v * np.pi / 16)
            basis[u, v] = 0.25 * cu * cv * np.outer(cos_u, cos_v)
    return basis


def _spatial_from_dct(Y_dct: np.ndarray, qt: np.ndarray) -> np.ndarray:
    basis = _idct_basis()
    deq = Y_dct.astype(np.float64) * qt[None, None, :, :]
    blocks = np.einsum('bhuv,uvxy->bhxy', deq, basis)
    bv, bh = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(bv * 8, bh * 8) + 128.0


def _uniform_filter(arr: np.ndarray, size) -> np.ndarray:
    sizes = [int(size)] * arr.ndim if np.isscalar(size) else [int(s) for s in size]
    out = arr.astype(np.float64, copy=True)
    for axis, n in enumerate(sizes):
        if n <= 1:
            continue
        k = n // 2
        pad = [(0, 0)] * out.ndim
        pad[axis] = (k, k)
        csum = np.cumsum(np.pad(out, pad, mode='symmetric'), axis=axis)
        csum = np.concatenate([np.zeros_like(csum.take([0], axis=axis)), csum], axis=axis)
        hi = [slice(None)] * out.ndim
        lo = [slice(None)] * out.ndim
        hi[axis] = slice(n, None)
        lo[axis] = slice(0, -n)
        out = (csum[tuple(hi)] - csum[tuple(lo)]) / n
    return out


def _wiener(im: np.ndarray, mysize) -> np.ndarray:
    local_mean = _uniform_filter(im, mysize)
    local_var = _uniform_filter(im ** 2, mysize) - local_mean ** 2
    noise = local_var.mean()
    # res blows up where local_var≈0, but those positions satisfy
    # local_var < noise and are overwritten by local_mean below.
    with np.errstate(divide='ignore', invalid='ignore'):
        res = local_mean + (1.0 - noise / local_var) * (im - local_mean)
    return np.where(local_var < noise, local_mean, res)


def _estimate_pixel_variance(spatial: np.ndarray) -> np.ndarray:
    denoised = _wiener(spatial, (3, 3))
    # wiener can yield NaN in totally-constant regions; replace with 0 so
    # the residual ends up at 0 and the local variance snaps to VAR_FLOOR.
    denoised = np.nan_to_num(denoised, nan=0.0)
    residual = spatial - denoised

    win = 2 * VAR_WINDOW + 1
    mu = _uniform_filter(residual, win)
    var_local = _uniform_filter(residual ** 2, win) - mu ** 2
    return np.maximum(var_local, VAR_FLOOR)


def compute_costs(Y_dct: np.ndarray, qt: np.ndarray) -> np.ndarray:
    blocks_v, blocks_h = Y_dct.shape[:2]
    spatial = _spatial_from_dct(Y_dct, qt)
    pixel_var = _estimate_pixel_variance(spatial)

    # Project σ²_pixel into σ²_DCT[u,v] = Σ basis[u,v,x,y]² · σ²_pixel[x,y].
    basis_sq = _idct_basis() ** 2                                  # (8,8,8,8) [u,v,x,y]
    pv = pixel_var.reshape(blocks_v, 8, blocks_h, 8).transpose(0, 2, 1, 3)
    var_dct = np.einsum('uvxy,bhxy->bhuv', basis_sq, pv)           # (bv,bh,8,8)

    qt_sq = qt.astype(np.float64) ** 2
    return qt_sq[None, None, :, :] / (var_dct + 1e-10)


def _seeded_rng(info: bytes, password: str) -> np.random.Generator:
    digest = hashlib.sha256(info + password.encode('utf-8')).digest()
    seed_seq = np.random.SeedSequence(list(np.frombuffer(digest, dtype=np.uint32)))
    return np.random.default_rng(seed_seq)


def _position_permutation(blocks_v: int, blocks_h: int, password: str) -> np.ndarray:
    total = blocks_v * blocks_h * 64
    all_idx = np.arange(total, dtype=np.int64)
    non_dc = all_idx[(all_idx % 64) != 0]
    rng = _seeded_rng(PERMUTATION_INFO, password)
    rng.shuffle(non_dc)
    return non_dc


def _bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bytes_from_bits(bits: np.ndarray) -> bytes:
    return np.packbits(bits).tobytes()


def capacity_bytes(cover_path: str) -> int:
    jpg = jpeglib.read_dct(cover_path)
    bv, bh = jpg.Y.shape[:2]
    cover_bits = bv * bh * 63                       # non-DC coefficients in Y plane
    message_bits = cover_bits // stc.W              # STC rate is 1/W
    return message_bits // 8


def embed(cover_path: str, payload: bytes, password: str, out_path: str) -> None:
    jpg = jpeglib.read_dct(cover_path)
    Y = jpg.Y.astype(np.int32).copy()
    qt_y = jpg.qt[0]
    bv, bh = Y.shape[:2]

    n_message_bits = len(payload) * 8
    n_cover_bits = n_message_bits * stc.W
    max_cover_bits = bv * bh * 63
    if n_cover_bits > max_cover_bits:
        max_bytes = max_cover_bits // (8 * stc.W)
        raise ValueError(
            f"payload too large for cover: needs {n_cover_bits} cover bits at "
            f"STC rate 1/{stc.W}, image has {max_cover_bits} non-DC "
            f"coefficients ({len(payload)} bytes vs {max_bytes} max)"
        )

    cost = compute_costs(Y, qt_y)
    perm = _position_permutation(bv, bh, password)
    positions = perm[:n_cover_bits]                # coefficient indices used

    flat_Y = Y.reshape(-1)
    flat_cost = cost.reshape(-1)

    cover_bits = (flat_Y[positions] & 1).astype(np.uint8)
    embed_costs = flat_cost[positions]
    message_bits = _bits_from_bytes(payload)

    # STC chooses which LSBs to flip to minimize total ρ.
    stego_bits = stc.encode(cover_bits, message_bits, embed_costs)

    # Per-position direction (+1 vs -1) for actually flipping coefficient LSBs.
    # The cost ρ is symmetric in ±1, so direction doesn't affect what STC just
    # optimized. Password-derived means the receiver doesn't need it — wrong
    # direction guesses don't change LSB parity, which is all extraction reads.
    dir_rng = _seeded_rng(DIRECTION_INFO, password)
    dir_bits = dir_rng.integers(0, 2, size=n_cover_bits, dtype=np.int8)

    flip_mask = stego_bits != cover_bits
    flip_positions = positions[flip_mask]
    flip_dirs = dir_bits[flip_mask]
    coefs = flat_Y[flip_positions]
    # Sign-clip near JPEG's int16 limits: whichever direction would overflow,
    # go the other way. ±32760 leaves a small headroom inside int16.
    saturated_high = coefs >= 32760
    saturated_low = coefs <= -32760
    delta = np.where(saturated_high, -1,
             np.where(saturated_low, 1,
              np.where(flip_dirs.astype(bool), 1, -1)))
    flat_Y[flip_positions] = coefs + delta

    jpg.Y = flat_Y.reshape(Y.shape).astype(jpg.Y.dtype)
    jpg.write_dct(out_path)


def extract(stego_path: str, password: str, n_bytes: int) -> bytes:
    jpg = jpeglib.read_dct(stego_path)
    Y = jpg.Y.astype(np.int32)
    bv, bh = Y.shape[:2]

    n_message_bits = n_bytes * 8
    n_cover_bits = n_message_bits * stc.W
    perm = _position_permutation(bv, bh, password)
    if n_cover_bits > perm.size:
        raise ValueError("requested more bits than the image can hold")

    flat_Y = Y.reshape(-1)
    stego_bits = (flat_Y[perm[:n_cover_bits]] & 1).astype(np.uint8)
    message_bits = stc.decode(stego_bits, n_message_bits)
    return _bytes_from_bits(message_bits)


def embed_text(cover_path: str, text: str, password: str, out_path: str) -> None:
    blob = core.encrypt(text.encode('utf-8'), password)
    embed(cover_path, blob, password, out_path)


def extract_text(stego_path: str, password: str) -> str:
    jpg = jpeglib.read_dct(stego_path)
    Y = jpg.Y.astype(np.int32)
    bv, bh = Y.shape[:2]
    perm = _position_permutation(bv, bh, password)
    flat_Y = Y.reshape(-1)

    def _read(n_bytes: int) -> bytes:
        n_message_bits = n_bytes * 8
        n_cover_bits = n_message_bits * stc.W
        if n_cover_bits > perm.size:
            raise ValueError("stego too short for requested read")
        stego_bits = (flat_Y[perm[:n_cover_bits]] & 1).astype(np.uint8)
        return _bytes_from_bits(stc.decode(stego_bits, n_message_bits))

    header = _read(core.HEADER_LEN)
    blob_len = core.expected_blob_len(header)
    blob = _read(blob_len)
    return core.decrypt(blob, password).decode('utf-8')
