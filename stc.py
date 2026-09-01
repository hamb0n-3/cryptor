import numpy as np


H = 10                          # constraint height; state space is 2**H = 1024
W = 2                           # cover bits per message bit; α = 1/W = 0.5
NSTATE = 1 << H                 # 1024
NHALF = NSTATE >> 1             # 512 — half of states are reachable post-shift
NV = 1 << W                     # 4 — block values {0..3}

# Submatrix Ĥ (H × W) given as two H-bit integers. Any pair where row 0 has
# at least one '1' guarantees the trellis can always satisfy the constraint;
# specific values affect the rate-distortion gap (typical reference codes
# achieve ~0.5 dB from the optimal bound at h=10).
_HAT_COLS = (0x32D, 0x2A7)      # = (813, 679); both have bit 0 set


def _block_xor_values() -> np.ndarray:
    out = np.zeros(NV, dtype=np.int64)
    for v in range(NV):
        for j in range(W):
            if (v >> j) & 1:
                out[v] ^= _HAT_COLS[j]
    return out


_BLOCK_XOR = _block_xor_values()


def encode(cover_bits, message_bits, costs) -> np.ndarray:
    cover_bits = np.asarray(cover_bits, dtype=np.uint8)
    message_bits = np.asarray(message_bits, dtype=np.uint8)
    costs = np.asarray(costs, dtype=np.float64)

    n = len(cover_bits)
    m = len(message_bits)
    if n != m * W:
        raise ValueError(f"cover len {n} != message len {m} * W={W}")
    if len(costs) != n:
        raise ValueError(f"costs len {len(costs)} != cover len {n}")

    # state_cost[s] = minimum cost to reach state s at the start of a block.
    # Only state 0 reachable initially (no rows finalized yet, all parities 0).
    state_cost = np.full(NSTATE, np.inf, dtype=np.float64)
    state_cost[0] = 0.0

    # Backpointer: at block b, given the post-shift final state f ∈ [0, NHALF),
    # which block-value v was chosen. The previous state can be reconstructed
    # via: prev = (2f + msg_bit[b]) XOR block_xor[v].
    back_v = np.zeros((m, NHALF), dtype=np.uint8)

    final_s_arr = np.arange(NHALF, dtype=np.int64)

    # Precompute per-block costs of all 2^W block-value choices.
    # block_costs[b, v] = sum_j (bit_j(v) XOR x[b*W+j]) * costs[b*W+j]
    cov = cover_bits.reshape(m, W).astype(np.int8)
    c = costs.reshape(m, W)
    v_bits = np.array([[v & 1, (v >> 1) & 1] for v in range(NV)], dtype=np.int8)
    diff = v_bits[None, :, :] ^ cov[:, None, :]      # (m, NV, W)
    block_costs = (diff * c[:, None, :]).sum(axis=2)  # (m, NV)

    # Trellis forward pass.
    for b in range(m):
        msg_bit = int(message_bits[b])
        # Pre-shift state pattern: ps = 2*f + msg_bit for f ∈ [0, NHALF).
        ps_arr = 2 * final_s_arr + msg_bit            # (NHALF,)

        # For all v simultaneously: src[v, f] = ps[f] XOR block_xor[v].
        src_all = ps_arr[None, :] ^ _BLOCK_XOR[:, None]  # (NV, NHALF)
        cand_all = state_cost[src_all] + block_costs[b, :, None]   # (NV, NHALF)

        # Pick best v per final state.
        best_v = np.argmin(cand_all, axis=0)                  # (NHALF,)
        new_cost = cand_all[best_v, final_s_arr]              # (NHALF,)

        state_cost[:NHALF] = new_cost
        state_cost[NHALF:] = np.inf
        back_v[b] = best_v.astype(np.uint8)

    # Best terminal state (any final state is acceptable; the matrix is
    # truncated to m rows so rows beyond m have no syndrome constraint).
    final_s = int(np.argmin(state_cost[:NHALF]))
    if state_cost[final_s] == np.inf:
        raise ValueError("STC encoding failed: no path through the trellis")

    # Backtrack to recover the chosen block-values, hence the stego bits.
    stego_bits = np.empty(n, dtype=np.uint8)
    s = final_s
    for b in range(m - 1, -1, -1):
        v = int(back_v[b, s])
        stego_bits[b * W] = v & 1
        stego_bits[b * W + 1] = (v >> 1) & 1
        s = int((2 * s + int(message_bits[b])) ^ _BLOCK_XOR[v])

    return stego_bits


def decode(stego_bits, n_message_bits: int) -> np.ndarray:
    stego_bits = np.asarray(stego_bits, dtype=np.uint8)
    needed = n_message_bits * W
    if needed > len(stego_bits):
        raise ValueError(f"need {needed} stego bits, have {len(stego_bits)}")

    # Convert each W-bit block to its v ∈ {0..NV-1} index.
    blocks = stego_bits[:needed].reshape(n_message_bits, W).astype(np.int64)
    v_arr = blocks[:, 0] | (blocks[:, 1] << 1)
    xor_contrib = _BLOCK_XOR[v_arr]                  # (n_message_bits,)

    # The state recurrence is sequential (each step shifts), so a Python
    # loop over integers is faster than vectorizing across blocks.
    state = 0
    message = np.zeros(n_message_bits, dtype=np.uint8)
    for b in range(n_message_bits):
        state ^= int(xor_contrib[b])
        message[b] = state & 1
        state >>= 1

    return message
