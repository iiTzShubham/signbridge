"""Loads sign-language video clips for PyTorch.

For each clip listed in the manifest it opens the video with OpenCV, picks evenly spaced frames,
resizes them to 224x224, converts colours BGR->RGB and normalises them the way ImageNet models expect.
The result is a tensor shaped [frames, 3, 224, 224] plus the word's class number.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .frame_sampling import uniform_frame_indices
from .preprocessing import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class SignVideoDataset(Dataset):
    def __init__(self, manifest, dataset_root, split, label_to_index,
                 num_frames=8, image_size=IMAGE_SIZE):
        rows = pd.read_csv(manifest).fillna("")
        self.rows = rows[rows["split"] == split].reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.label_to_index = label_to_index
        self.num_frames = num_frames
        self.image_size = image_size
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def _video_path(self, value):
        path = Path(str(value))
        return path if path.is_absolute() else self.dataset_root / path

    def _normalize(self, frames):
        tensor = torch.from_numpy(np.stack(frames).copy()).permute(0, 3, 1, 2).float().div_(255.0)
        return (tensor - self.mean) / self.std

    def augmented_views(self, index, views, seed=0):
        """Return [views, time, C, H, W]: view 0 is the clean clip; others use temporal/crop/flip jitter."""
        row = self.rows.iloc[index]
        path = self._video_path(row["video_path"])
        capture = cv2.VideoCapture(str(path))
        decoded = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if not decoded:
            raise RuntimeError(f"Cannot decode video: {path}")
        frame_start = min(max(0, int(row.get("frame_start", 0) or 0)), len(decoded) - 1)
        frame_end = int(row.get("frame_end", -1) or -1)
        if frame_end < frame_start or frame_end >= len(decoded):
            frame_end = len(decoded) - 1
        decoded = decoded[frame_start:frame_end + 1]
        height, width = decoded[0].shape[:2]
        rng = np.random.default_rng(seed * 1_000_003 + index)
        last = len(decoded) - 1
        output = []
        for view in range(views):
            if view == 0:
                start, end, scale, cx, cy, flip = 0, last, 1.0, 0.5, 0.5, False
            else:
                start = int(rng.uniform(0, 0.15) * last)
                end = last - int(rng.uniform(0, 0.15) * last)
                scale, cx, cy = rng.uniform(0.8, 1.0), rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.6)
                flip = rng.random() < 0.5
            crop_w, crop_h = int(width * scale), int(height * scale)
            x0 = int(np.clip(cx * width - crop_w / 2, 0, width - crop_w))
            y0 = int(np.clip(cy * height - crop_h / 2, 0, height - crop_h))
            frames = []
            for frame_index in uniform_frame_indices(max(1, end - start + 1), self.num_frames) + start:
                image = decoded[int(frame_index)][y0:y0 + crop_h, x0:x0 + crop_w]
                image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                frames.append(image[:, ::-1] if flip else image)
            output.append(self._normalize(frames))
        return torch.stack(output)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        path = self._video_path(row["video_path"])
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_start = max(0, int(row.get("frame_start", 0) or 0))
        frame_end = int(row.get("frame_end", -1) or -1)
        frame_start = min(frame_start, max(0, frame_count - 1))
        if frame_end < frame_start or frame_end >= frame_count:
            frame_end = max(0, frame_count - 1)
        clip_length = max(1, frame_end - frame_start + 1)
        indices = uniform_frame_indices(clip_length, self.num_frames) + frame_start
        frames = []
        last_frame = None
        try:
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok:
                    if last_frame is None:
                        raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
                    frame = last_frame.copy()
                else:
                    last_frame = frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                frames.append(frame)
        finally:
            capture.release()
        array = np.stack(frames).copy()
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float().div_(255.0)
        tensor = (tensor - self.mean) / self.std
        label = self.label_to_index[str(row["label"])]
        return tensor, torch.tensor(label, dtype=torch.long), str(row["video_id"])
