"""A fixed block of memory holding the last few seconds of frames.

The camera writes into it continuously and never waits. Everything else reads
from it, behind the camera, and can fall behind by up to the buffer's length
before a frame is lost. That slack is what lets the start of a lift be kept:
by the time the watcher notices the plate moving, the frames before it are
still here.

All memory is allocated once. A frame is copied into its slot on arrival and
never allocated again, so memory use is flat however long the camera runs.
"""

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity, shape, dtype=np.uint8):
        self.capacity = int(capacity)
        self._images = np.empty((self.capacity, *shape), dtype)
        self._times = np.zeros(self.capacity)
        self._seq = np.full(self.capacity, -1, np.int64)   # which frame each slot holds
        self._newest = -1
        self._lock = threading.Lock()
        self._arrived = threading.Condition(self._lock)
        self.closed = False

    @property
    def nbytes(self):
        return self._images.nbytes

    @property
    def newest(self):
        return self._newest

    @property
    def oldest(self):
        return max(0, self._newest - self.capacity + 1) if self._newest >= 0 else 0

    def push(self, image, t):
        """Store a frame; returns its sequence number."""
        with self._lock:
            seq = self._newest + 1
            slot = seq % self.capacity
            self._seq[slot] = -1              # mark the slot as being written
        self._images[slot] = image
        with self._lock:
            self._times[slot] = t
            self._seq[slot] = seq
            self._newest = seq
            self._arrived.notify_all()
        return seq

    def wait(self, seq, timeout=None):
        """Block until frame seq has arrived or the buffer is closed; True if it has."""
        with self._arrived:
            return self._arrived.wait_for(lambda: self._newest >= seq or self.closed, timeout) \
                and self._newest >= seq

    def close(self):
        with self._lock:
            self.closed = True
            self._arrived.notify_all()

    def read(self, seq, crop=None):
        """A copy of frame seq, or of a crop (x0, y0, x1, y1) of it, with its time.

        Returns None if the frame is not there: not yet written, or already
        overwritten. The copy is checked after it is made, so a frame that was
        overwritten while it was being copied is never returned.
        """
        if seq < 0:
            return None
        slot = seq % self.capacity
        with self._lock:
            if self._seq[slot] != seq:
                return None
            t = self._times[slot]
        if crop is None:
            out = self._images[slot].copy()
        else:
            x0, y0, x1, y1 = crop
            out = self._images[slot, y0:y1, x0:x1].copy()
        with self._lock:
            if self._seq[slot] != seq:
                return None
        return out, float(t)
