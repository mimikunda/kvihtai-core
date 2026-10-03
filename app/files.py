"""Writing files so that pulling the plug never leaves half of one.

The Pi is often switched off at the wall. A file written in place and cut off
half way cannot be read back: camera.json then falls back to the defaults and
the station forgets which way is up, where the near plate is and where it was
focused. Written to a temporary file, flushed to the card and renamed, a file
is either the old one or the new one.
"""

import json
import os


def write_json(path, data, indent=None):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=indent)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default
