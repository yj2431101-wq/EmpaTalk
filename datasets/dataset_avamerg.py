"""AvaMERG dataset for ListenerBank training.

Expects the AvaMERG directory layout::

    <root>/
        video_v5_0/
            <video_id>.mp4   (or subdirs)
        audio_v5_0/
            <video_id>.wav
        train.json

train.json format (three accepted variants):
    Variant A (ID-based):
        [{"speaker": "video_id_spk", "listener": "video_id_lis"}, ...]
    Variant B (path-based):
        [{"speaker_video": "rel/path/spk.mp4", "listener_video": "rel/path/lis.mp4"}, ...]
    Variant C (AvaMERG conversation format):
        [{"conversation_id": "00001",
          "speaker_profile": {"ID": 16, ...},
          "listener_profile": {"ID": 42, ...},
          "turns": [{"turn_id": "0",
                     "dialogue_history": [{"index": 0, "role": "speaker", ...}],
                     "response": "...", ...}, ...]}, ...]
        Video filenames are inferred as dia{conv_id}utt{utt_idx}_{speaker_id}.mp4

Each __getitem__ returns a dict with keys:
    speaker_frame   – (C, H, W) float32 in [-1, 1]  speaker at time t
    listener_source – (C, H, W) float32 in [-1, 1]  listener identity frame (t - gap)
    listener_target – (C, H, W) float32 in [-1, 1]  ground-truth listener at time t
    pair_id         – str  "<speaker_id>#<listener_id>@<frame_idx>"
"""

import json
import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


def _find_video(video_dir: str, video_id: str) -> str:
    """Resolve a video_id to a .mp4 path under video_dir."""
    # Direct match
    direct = os.path.join(video_dir, video_id)
    if os.path.isfile(direct):
        return direct
    direct_mp4 = direct if direct.endswith(".mp4") else direct + ".mp4"
    if os.path.isfile(direct_mp4):
        return direct_mp4
    # Recursive search (one level of subdirs)
    for entry in os.scandir(video_dir):
        if entry.is_dir():
            candidate = os.path.join(entry.path, video_id)
            if os.path.isfile(candidate):
                return candidate
            candidate_mp4 = candidate if candidate.endswith(".mp4") else candidate + ".mp4"
            if os.path.isfile(candidate_mp4):
                return candidate_mp4
    raise FileNotFoundError(
        f"Cannot find video '{video_id}' under '{video_dir}'"
    )


def _read_frame(cap: cv2.VideoCapture, frame_idx: int, size: int) -> torch.Tensor: # 전처리 미리하면 확 빨라질까?
    """Read a single frame from an open VideoCapture and return (C, H, W) in [-1, 1]."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    if not ok:
        # Fall back to last frame
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
        _, bgr = cap.read()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [0,1]
    tensor = (tensor - 0.5) / 0.5  # [-1, 1]
    return tensor


class AvaMERGDataset(Dataset):
    """Dyadic speaker-listener frame pair dataset built from AvaMERG.

    Args:
        root:        Path to the AvaMERG root directory.
        split_json:  Path to the JSON split file (default: <root>/train.json).
        size:        Spatial resolution to resize frames to (default 256).
        identity_gap_min / identity_gap_max:
                     The listener identity frame is sampled from a frame that
                     is ``gap`` frames *before* the target frame, where gap is
                     drawn uniformly from [identity_gap_min, identity_gap_max].
                     This prevents the identity frame from leaking pose/motion.
        frames_per_video:
                     How many (speaker, listener) frame pairs to sample from
                     each video pair per epoch (default 8).
    """

    def __init__(
        self,
        root: str,
        split_json: Optional[str] = None,
        size: int = 256,
        identity_gap_min: int = 10,
        identity_gap_max: int = 30,
        frames_per_video: int = 8,
    ):
        self.root = root
        self.video_dir = os.path.join(root, "video_v5_0")
        self.size = size
        self.identity_gap_min = identity_gap_min
        self.identity_gap_max = identity_gap_max
        self.frames_per_video = frames_per_video

        if split_json is None:
            split_json = os.path.join(root, "train.json")
        with open(split_json, "r") as f:
            raw = json.load(f)

        self.pairs = self._parse_pairs(raw)
        # Build flat index: (pair_idx, frame_seed)
        self._index = [
            (pi, seed)
            for pi in range(len(self.pairs))
            for seed in range(frames_per_video)
        ]

    # ------------------------------------------------------------------
    def _parse_pairs(self, raw: list) -> list:
        """Normalise JSON entries into {'spk_id', 'lis_id', 'spk_path', 'lis_path'}."""
        pairs = []
        for item in raw:
            if isinstance(item, str):
                # "spk_id#lis_id" shorthand
                parts = item.split("#")
                spk_id, lis_id = parts[0], parts[-1]
                spk_path = _find_video(self.video_dir, spk_id)
                lis_path = _find_video(self.video_dir, lis_id)
                pairs.append({"spk_id": spk_id, "lis_id": lis_id,
                               "spk_path": spk_path, "lis_path": lis_path})
            elif "conversation_id" in item:
                # Variant C: AvaMERG conversation format
                pairs.extend(self._parse_avamerg_conversation(item))
            elif "speaker_video" in item:
                # Variant B: explicit relative paths
                spk_path = os.path.join(self.root, item["speaker_video"])
                lis_path = os.path.join(self.root, item["listener_video"])
                spk_id = os.path.splitext(os.path.basename(spk_path))[0]
                lis_id = os.path.splitext(os.path.basename(lis_path))[0]
                pairs.append({"spk_id": spk_id, "lis_id": lis_id,
                               "spk_path": spk_path, "lis_path": lis_path})
            else:
                # Variant A: ID-based
                spk_id = item.get("speaker", item.get("speaker_id", ""))
                lis_id = item.get("listener", item.get("listener_id", ""))
                spk_path = _find_video(self.video_dir, spk_id)
                lis_path = _find_video(self.video_dir, lis_id)
                pairs.append({"spk_id": spk_id, "lis_id": lis_id,
                               "spk_path": spk_path, "lis_path": lis_path})
        return pairs

    def _parse_avamerg_conversation(self, item: dict) -> list:
        """Parse one AvaMERG conversation entry into speaker-listener video pairs.

        Video naming convention: dia{conv_id}utt{utt_idx}_{speaker_id}.mp4

        For each turn, the speaker's utterance index is taken from
        dialogue_history[-1]["index"], and the listener's response video is
        at index + 1.  Pairs where either video is missing are silently skipped.
        """
        conv_id = item["conversation_id"]
        spk_num = item.get("speaker_profile", {}).get("ID", "")
        lis_num = item.get("listener_profile", {}).get("ID", "")

        pairs = []
        for turn in item.get("turns", []):
            history = turn.get("dialogue_history", [])
            if not history:
                continue
            # The last entry in dialogue_history is the current speaker utterance
            last_utt = history[-1]
            spk_utt_idx = last_utt.get("index", 0)
            lis_utt_idx = spk_utt_idx + 1

            spk_stem = f"dia{conv_id}utt{spk_utt_idx}_{spk_num}"
            lis_stem = f"dia{conv_id}utt{lis_utt_idx}_{lis_num}"

            try:
                spk_path = _find_video(self.video_dir, spk_stem)
                lis_path = _find_video(self.video_dir, lis_stem)
            except FileNotFoundError:
                continue

            pairs.append({
                "spk_id":   spk_stem,
                "lis_id":   lis_stem,
                "spk_path": spk_path,
                "lis_path": lis_path,
            })
        return pairs

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict:
        pair_idx, _ = self._index[index]
        pair = self.pairs[pair_idx]

        cap_spk = cv2.VideoCapture(pair["spk_path"])
        cap_lis = cv2.VideoCapture(pair["lis_path"])

        n_spk = int(cap_spk.get(cv2.CAP_PROP_FRAME_COUNT))
        n_lis = int(cap_lis.get(cv2.CAP_PROP_FRAME_COUNT))

        # Synchronised target frame (clipped to both video lengths)
        max_frame = min(n_spk, n_lis) - 1
        gap = random.randint(self.identity_gap_min, self.identity_gap_max)
        t = random.randint(gap, max(gap, max_frame))

        speaker_frame   = _read_frame(cap_spk, t,          self.size)
        listener_target = _read_frame(cap_lis, t,          self.size)
        listener_source = _read_frame(cap_lis, max(0, t - gap), self.size)

        cap_spk.release()
        cap_lis.release()

        return {
            "speaker_frame":   speaker_frame,
            "listener_source": listener_source,
            "listener_target": listener_target,
            "pair_id": f"{pair['spk_id']}#{pair['lis_id']}@{t}",
        }
