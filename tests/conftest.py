"""Make the project importable from the tests without installing it."""

import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The database and everything else the app keeps go to a directory of their
# own, not the checkout's data/. Set before app.config is first imported.
os.environ.setdefault("KVIHTAI_DATA_DIR", tempfile.mkdtemp(prefix="kvihtai-test-"))
os.environ.setdefault("KVIHTAI_CAMERA", "none")
