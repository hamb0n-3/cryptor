import ast
import base64
import binascii
import json

import core


METADATA_KEY = "x_encrypted_payload"   # custom metadata marker on the payload cell
PAYLOAD_VARIABLE = "ENCRYPTED_PAYLOAD"  # Python var name holding the base64 string
PAYLOAD_VERSION = "v1"
JOSE_ALG = "XChaCha20-Poly1305"        # for visibility in the metadata
WRAP_WIDTH = 72                         # base64 chars per line when wrapping


def to_ipynb(text: str, password: str) -> str:
    return render(core.encrypt(text.encode('utf-8'), password))


def from_ipynb(notebook: str, password: str) -> str:
    return core.decrypt(parse(notebook), password).decode('utf-8')


def render(blob: bytes) -> str:
    b64 = base64.b64encode(blob).decode('ascii')

    # Short blobs on one line, longer ones wrapped via implicit string
    # concatenation in parens. Both round-trip cleanly through
    # ast.literal_eval because CPython merges adjacent string literals at
    # parse time into a single Constant node.
    if len(b64) <= WRAP_WIDTH:
        payload_source = [f'{PAYLOAD_VARIABLE} = {json.dumps(b64)}\n']
    else:
        payload_source = [f'{PAYLOAD_VARIABLE} = (\n']
        for i in range(0, len(b64), WRAP_WIDTH):
            payload_source.append(f'    {json.dumps(b64[i:i + WRAP_WIDTH])}\n')
        payload_source.append(')\n')

    notebook = {
        "cells": [
            {
                "id": "title",
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Encrypted Message\n",
                    "\n",
                    f"Payload encrypted with `{JOSE_ALG}`. To decrypt:\n",
                    "\n",
                    "```\n",
                    "python cli.py -d --enc ipynb -f this_notebook.ipynb\n",
                    "```\n",
                ],
            },
            {
                "id": "payload",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    METADATA_KEY: {"v": PAYLOAD_VERSION, "alg": JOSE_ALG},
                },
                "outputs": [],
                "source": payload_source,
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1) + "\n"


def parse(notebook: str) -> bytes:
    try:
        nb = json.loads(notebook)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid notebook JSON: {e}") from e

    if not isinstance(nb, dict):
        raise ValueError("notebook must be a JSON object")
    cells = nb.get("cells")
    if not isinstance(cells, list):
        raise ValueError("notebook missing 'cells' array")

    # Find the cell tagged with our metadata key.
    payload_cell = None
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        md = cell.get("metadata", {})
        if isinstance(md, dict) and METADATA_KEY in md:
            payload_cell = cell
            break
    if payload_cell is None:
        raise ValueError(f"no cell with {METADATA_KEY!r} metadata found")

    # Per nbformat spec, cell.source is either a string or a list of strings.
    source = payload_cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    if not isinstance(source, str):
        raise ValueError("payload cell 'source' has unexpected type")

    b64 = _extract_payload_string(source)
    try:
        return base64.b64decode(b64, validate=True)
    except binascii.Error as e:
        raise ValueError(f"payload is not valid base64: {e}") from e


def _extract_payload_string(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"payload cell source is not valid Python: {e}") from e

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == PAYLOAD_VARIABLE:
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"can't evaluate {PAYLOAD_VARIABLE}: {e}") from e
                if not isinstance(value, str):
                    raise ValueError(f"{PAYLOAD_VARIABLE} is not a string literal")
                return value

    raise ValueError(f"no `{PAYLOAD_VARIABLE} = ...` assignment in payload cell")