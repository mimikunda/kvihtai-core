"""The analysis of a set, in a process of its own.

The analysis is Python and numpy for tens of seconds per set on a Pi 4. In a
thread it holds the interpreter's lock often enough that the camera and the
watcher, which must never wait, lose frames: on the Pi, a set played while
the one before it was being analysed lost more than half of its frames. A
process of its own shares no lock with them, and runs at a lower priority.

The crops go over one at a time and are dropped on this side as they go, so
that a set is never held in memory twice. A worker that dies, for instance
killed for memory, is started again for the next set.
"""

import dataclasses
import multiprocessing
import os
import threading


class ProcessAnalyser:
    """Callable like app.capture.session.analyse, but runs it in a worker process."""

    def __init__(self, nice=10):
        self.nice = nice
        self._ctx = multiprocessing.get_context("spawn")
        self._proc = None
        self._conn = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._proc is not None and self._proc.is_alive():
            return
        parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_serve, args=(child, self.nice), name="kvihtai-analysis", daemon=True)
        self._proc.start()
        child.close()
        self._conn = parent

    def __call__(self, rec, quarter_turns=0):
        with self._lock:
            self._ensure()
            crops = rec.crops
            try:
                self._conn.send((dataclasses.replace(rec, crops=[]), len(crops), quarter_turns))
                for i in range(len(crops)):
                    self._conn.send(crops[i])
                    crops[i] = None
                ok, value = self._conn.recv()
            except (EOFError, OSError) as e:
                self._proc = None
                raise RuntimeError(f"the analysis process died: {e!r}") from e
        if not ok:
            raise RuntimeError(value)
        return value

    def close(self):
        with self._lock:
            if self._proc is None:
                return
            try:
                self._conn.send(None)
            except OSError:
                pass
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None


def _serve(conn, nice):
    os.nice(nice)
    from app.capture.session import analyse

    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        head, count, quarter_turns = message
        head.crops = [conn.recv() for _ in range(count)]
        try:
            result = analyse(head, quarter_turns)
        except Exception as e:
            conn.send((False, repr(e)))
        else:
            conn.send((True, result))
        head = None
