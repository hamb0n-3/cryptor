import core


def to_ipv4(text: str, password: str) -> list[str]:
    return render(core.encrypt(text.encode('utf-8'), password))


def from_ipv4(ip_list: list[str], password: str) -> str:
    raw = parse(ip_list)
    if len(raw) < core.HEADER_LEN:
        raise ValueError("input too short to contain blob header")
    blob_len = core.expected_blob_len(raw)
    if len(raw) < blob_len:
        raise ValueError("ciphertext truncated")
    trailing = raw[blob_len:]
    if len(trailing) >= 4 or any(t != 0 for t in trailing):
        raise ValueError("invalid trailing padding")
    return core.decrypt(raw[:blob_len], password).decode('utf-8')


def render(data: bytes) -> list[str]:
    if len(data) % 4:
        data = data + b'\x00' * (4 - (len(data) % 4))
    return ['.'.join(str(b) for b in data[i:i+4]) for i in range(0, len(data), 4)]


def parse(ip_list: list[str]) -> bytes:
    out = bytearray()
    for quad in ip_list:
        parts = quad.split('.')
        if len(parts) != 4:
            raise ValueError(f"not a valid IPv4 literal: {quad!r}")
        for part in parts:
            if not part.isdigit():       # rejects '-1', '0x10', '  42 ', '+1', etc.
                raise ValueError(f"non-numeric octet in {quad!r}")
            v = int(part)
            if not 0 <= v <= 255:
                raise ValueError(f"octet out of range in {quad!r}")
            out.append(v)
    return bytes(out)
