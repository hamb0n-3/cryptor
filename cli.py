import argparse
import getpass
import os
import sys
from pathlib import Path

import ipv4
import ipynb
import jmipod
import jwt
import tar


TEXT_PLUGINS = {
    'ipv4': {
        'encode': lambda text, pw: '\n'.join(ipv4.to_ipv4(text, pw)),
        'decode': lambda payload, pw: ipv4.from_ipv4(payload.split(), pw),
    },
    'jwt': {
        'encode': lambda text, pw: jwt.to_jwt(text, pw),
        # JWTs are single tokens; tolerate surrounding whitespace from files.
        'decode': lambda payload, pw: jwt.from_jwt(payload.strip(), pw),
    },
    'ipynb': {
        'encode': lambda text, pw: ipynb.to_ipynb(text, pw),
        # Notebook JSON tolerates trailing whitespace fine; pass as-is.
        'decode': lambda payload, pw: ipynb.from_ipynb(payload, pw),
    },
}
ALL_ENCODINGS = sorted(list(TEXT_PLUGINS) + ['jpg', 'tar'])


def _can_prompt_securely() -> bool:
    if sys.platform == 'win32':
        return True  # Windows getpass uses msvcrt, independent of stdin
    try:
        with open('/dev/tty', 'r+b', buffering=0) as tty:
            import termios
            termios.tcgetattr(tty.fileno())
        return True
    except (OSError, ImportError):
        return False


def _read_payload(args) -> str:
    if args.file is not None:
        try:
            return Path(args.file).read_text(encoding='utf-8')
        except OSError as e:
            sys.exit(f"error reading {args.file}: {e}")
        except UnicodeDecodeError as e:
            sys.exit(f"{args.file} is not valid UTF-8: {e}")
    if args.input is not None:
        return args.input
    # stdin fallback; hint to TTY users so they don't think it hung waiting
    # for Ctrl-D after the password prompt.
    if sys.stdin.isatty():
        print("Reading payload from stdin; press Ctrl-D when done.",
              file=sys.stderr, flush=True)
    try:
        return sys.stdin.buffer.read().decode('utf-8')
    except UnicodeDecodeError as e:
        sys.exit(f"stdin is not valid UTF-8: {e}")


def _write_output(text: str, out_path, ensure_newline: bool) -> None:
    if ensure_newline and not text.endswith('\n'):
        text = text + '\n'
    if out_path is None:
        sys.stdout.write(text)
        return
    try:
        Path(out_path).write_text(text, encoding='utf-8')
    except OSError as e:
        sys.exit(f"error writing {out_path}: {e}")


def _read_password(confirm: bool) -> str:
    # PASSWORD env var bypasses interactive prompting (for scripted use).
    if 'PASSWORD' in os.environ:
        pw = os.environ['PASSWORD']
        if not pw:
            sys.exit("PASSWORD env var is set but empty")
        return pw
    # Prompt for password BEFORE reading stdin so the prompt isn't buried
    # under piped input, and so getpass doesn't fall back to stdin and
    # consume the payload.
    pw = getpass.getpass("Password: ")
    if not pw:
        sys.exit("empty password")
    if confirm:
        if getpass.getpass("Confirm password: ") != pw:
            sys.exit("passwords don't match")
    return pw


def _check_password_safety(args) -> None:
    payload_from_stdin = args.input is None and args.file is None
    if (payload_from_stdin
            and 'PASSWORD' not in os.environ
            and not _can_prompt_securely()):
        sys.exit(
            "no terminal available for password input AND payload would come "
            "from stdin (getpass would eat the first line). Use -i / -f, set "
            "the PASSWORD environment variable, or run from a terminal."
        )


def _run_text_plugin(args) -> None:
    _check_password_safety(args)
    plugin = TEXT_PLUGINS[args.enc]
    password = _read_password(confirm=args.encrypt)
    payload = _read_payload(args)
    try:
        if args.encrypt:
            output = plugin['encode'](payload, password)
        else:
            output = plugin['decode'](payload, password)
    except ValueError as e:
        sys.exit(f"error: {e}")
    # Trailing newline on encrypt output (it's a structured artifact), but
    # preserve the plaintext byte-for-byte on decrypt.
    _write_output(output, args.out, ensure_newline=args.encrypt)


def _run_jpg(args) -> None:
    if args.encrypt:
        if args.cover is None:
            sys.exit("--enc jpg encrypt requires --cover (cover JPEG path)")
        if args.out is None:
            sys.exit("--enc jpg encrypt requires -o (stego JPEG output path)")
        _check_password_safety(args)         # plaintext may come from stdin
        password = _read_password(confirm=True)
        text = _read_payload(args)
        try:
            jmipod.embed_text(args.cover, text, password, args.out)
        except (ValueError, OSError) as e:
            sys.exit(f"error: {e}")
        print(f"{len(text.encode('utf-8'))} bytes",
              file=sys.stderr)
    else:
        if args.file is None:
            sys.exit("--enc jpg decrypt requires -f (stego JPEG path)")
        if args.input is not None:
            sys.exit("--enc jpg decrypt doesn't accept -i; use -f for the "
                     "stego JPEG path")
        # No payload-from-stdin path here, so no _check_password_safety needed.
        password = _read_password(confirm=False)
        try:
            text = jmipod.extract_text(args.file, password)
        except (ValueError, OSError) as e:
            sys.exit(f"error: {e}")
        _write_output(text, args.out, ensure_newline=False)


def _run_tar(args) -> None:
    if args.encrypt:
        if args.file is None:
            sys.exit("--enc tar encrypt requires -f (source directory or tarball)")
        if args.out is None:
            sys.exit("--enc tar encrypt requires -o (encrypted output file path)")
        if args.input is not None:
            sys.exit("--enc tar encrypt doesn't accept -i; use -f for the source path")
        # No payload-from-stdin path here, so no _check_password_safety needed.
        password = _read_password(confirm=True)
        try:
            blob = tar.encrypt_path(args.file, password)
            Path(args.out).write_bytes(blob)
        except (ValueError, OSError) as e:
            sys.exit(f"error: {e}")
        print(f"encrypted {args.file} → {args.out} ({len(blob)} bytes)",
              file=sys.stderr)
    else:
        if args.file is None:
            sys.exit("--enc tar decrypt requires -f (encrypted file path)")
        if args.out is None:
            sys.exit("--enc tar decrypt requires -o (destination directory)")
        if args.input is not None:
            sys.exit("--enc tar decrypt doesn't accept -i; use -f for the encrypted file")
        password = _read_password(confirm=False)
        try:
            blob = Path(args.file).read_bytes()
            tar.decrypt_to_path(blob, password, args.out)
        except (ValueError, OSError) as e:
            sys.exit(f"error: {e}")
        print(f"decrypted {args.file} → {args.out}/", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encrypt or decrypt text via a chosen encoding plugin.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('-e', '--encrypt', action='store_true', help='encrypt mode')
    mode.add_argument('-d', '--decrypt', action='store_true', help='decrypt mode')

    parser.add_argument('--enc', choices=ALL_ENCODINGS, default='ipv4',
                        help='encoding plugin (default: ipv4)')

    src = parser.add_mutually_exclusive_group()
    src.add_argument('-f', '--file', metavar='FILE',
                     help='read payload from FILE; for --enc jpg decrypt this '
                          'is the stego JPEG path (default for text encodings: '
                          'stdin)')
    src.add_argument('-i', '--input', metavar='TEXT',
                     help='read payload from this argument '
                          '(NOTE: visible in process listings and shell history)')

    parser.add_argument('-o', '--out', metavar='FILE',
                        help='write output to FILE (default: stdout; '
                             'required for --enc jpg encrypt)')
    parser.add_argument('--cover', metavar='FILE',
                        help='cover JPEG path (required for --enc jpg encrypt)')

    args = parser.parse_args()

    if args.enc == 'jpg':
        _run_jpg(args)
    elif args.enc == 'tar':
        _run_tar(args)
    else:
        _run_text_plugin(args)


if __name__ == "__main__":
    main()