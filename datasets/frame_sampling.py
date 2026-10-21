import numpy as np


def uniform_frame_indices(frame_count, num_frames):
    if frame_count <= 0:
        return np.zeros(num_frames, dtype=np.int64)
    if frame_count < num_frames:
        # Repeat samples evenly, including both ends, for short clips.
        return np.linspace(0, frame_count - 1, num_frames).round().astype(np.int64)
    return np.linspace(0, frame_count - 1, num_frames).round().astype(np.int64)
