"""The station's own log, kept on the card across restarts and power cuts.

The journal holds the same and more, but Raspberry Pi OS keeps it in memory
unless told otherwise, and then what led up to a power cut goes with it. This
file is small, rotates, and is what the app shows as the station's log: the
station's own lines, the camera library's, and every warning and error from
anywhere in the process, including a thread that died.
"""

import logging
import logging.handlers
import os
import re
import sys
import threading

from app.system import PROBLEM

FILE = "station.log"
KEEP_BYTES = 1_000_000
BACKUPS = 4
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# a record starts with its time; the lines after it, as in a traceback, belong to it
RECORD = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ")
LEVEL = re.compile(r"^\S+ \S+ (WARNING|ERROR|CRITICAL) ")

_handler = None
_hooked = False


def _worth_keeping(record):
    return record.levelno >= logging.WARNING or record.name.startswith(("kvihtai", "picamera2"))


def _thread_died(args):
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else "?"
    logging.getLogger("kvihtai.crash").error(f"thread {name} died",
                                             exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


def _process_died(exc_type, exc_value, tb):
    logging.getLogger("kvihtai.crash").critical("the station died",
                                                exc_info=(exc_type, exc_value, tb))
    sys.__excepthook__(exc_type, exc_value, tb)


def install(directory):
    """Start writing the log to directory/station.log."""
    global _handler, _hooked
    uninstall()
    os.makedirs(directory, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(os.path.join(directory, FILE), maxBytes=KEEP_BYTES,
                                                   backupCount=BACKUPS, encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S"))
    handler.addFilter(_worth_keeping)
    # uvicorn's loggers do not pass their records on to the root logger
    for name in ("", "uvicorn"):
        logging.getLogger(name).addHandler(handler)
    # the station's own lines are kept whatever the root logger lets through
    if logging.getLogger("kvihtai").level == logging.NOTSET:
        logging.getLogger("kvihtai").setLevel(logging.INFO)
    _handler = handler
    if not _hooked:
        threading.excepthook = _thread_died
        sys.excepthook = _process_died
        _hooked = True


def uninstall():
    global _handler
    if _handler is None:
        return
    for name in ("", "uvicorn"):
        logging.getLogger(name).removeHandler(_handler)
    _handler.close()
    _handler = None


def _records(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        if RECORD.match(line) or not out:
            out.append(line)
        else:
            out[-1] += "\n" + line
    return out


def is_problem(record):
    return bool(LEVEL.match(record) or PROBLEM.search(record))


def read(directory, lines=400, problems=False):
    """The newest records, oldest first; only warnings, errors and failures if problems."""
    base = os.path.join(directory, FILE)
    got = []
    for path in [base] + [f"{base}.{i}" for i in range(1, BACKUPS + 1)]:
        records = _records(path)
        if problems:
            records = [r for r in records if is_problem(r)]
        got = records + got
        if len(got) >= lines:
            break
    return got[-lines:]


def files(directory):
    """The log's files on disk, newest first, for the diagnostics bundle."""
    base = os.path.join(directory, FILE)
    return [p for p in [base] + [f"{base}.{i}" for i in range(1, BACKUPS + 1)] if os.path.exists(p)]
