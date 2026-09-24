"""Frames for analysis, whole or cropped.

Offline the tracker has whole frames from a video file. On the Pi a set is
kept as a crop round the plate per frame, since whole frames at 80 fps would
fill the memory in seconds. Both look the same to the tracker: an image, the
position of its top-left corner in the full frame, and a timestamp.
"""

import numpy as np


class Frames:
    """Whole frames held in memory."""

    def __init__(self, images, times_s, frame_size=None):
        self.images = images
        self.times = np.asarray(times_s, float)
        h, w = images[0].shape[:2]
        self.frame_size = frame_size or (w, h)

    def __len__(self):
        return len(self.images)

    def image(self, i):
        return self.images[i], np.zeros(2)


class Crops(Frames):
    """A crop per frame, with where each crop sat in the full frame."""

    def __init__(self, images, origins, times_s, frame_size):
        super().__init__(images, times_s, frame_size)
        self.origins = [np.asarray(o, float) for o in origins]

    def image(self, i):
        return self.images[i], self.origins[i]


def read_video(path):
    """Every frame of a video file and its timestamp in seconds.

    Timestamps come from the container, not from the nominal frame rate: phone
    footage has a variable rate, and the second test clip runs at 29.6 fps
    while claiming 30.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    images, times = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        images.append(frame)
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()
    if not images:
        raise OSError(f"no frames in {path}")
    times = np.asarray(times)
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        # Some containers report no usable timestamps; fall back to the rate.
        fps = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS) or 30.0
        times = np.arange(len(images)) / fps
    return Frames(images, times)
