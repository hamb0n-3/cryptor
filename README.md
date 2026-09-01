# cryptor

Encrypts text and hides the ciphertext inside something that looks like an
ordinary file: a list of IPv4 addresses, a JWT, a Jupyter notebook, a JPEG, or
a tar. Decrypt reverses it.

Encryption is XChaCha20-Poly1305 with a key derived from your password via
scrypt (N=2^17). The password is asked for interactively (`getpass`), never
taken on the command line.

## Usage

```
# encrypt to stdout as IPv4 addresses
python cli.py -e --enc ipv4 -i "some secret"

# decrypt from a file
python cli.py -d --enc ipv4 -f payload.txt

# hide inside a cover JPEG
python cli.py -e --enc jpg --cover cover.jpg -i "some secret" -o out.jpg
python cli.py -d --enc jpg -f out.jpg
```

`--enc` is one of `ipv4`, `jwt`, `ipynb`, `jpg`, `tar` (default `ipv4`).
Input comes from `-i TEXT` or `-f FILE`; output goes to stdout or `-o FILE`.

## Requires

`pynacl`. The `jpg` carrier also needs `jpeglib` and `numpy`.
