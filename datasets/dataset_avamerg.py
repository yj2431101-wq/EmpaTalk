"""AvaMERG dataset for ListenerBank training (utterance-level).

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
    speaker_video   – (T, C, H, W) float32 in [-1, 1]  speaker video frames
    listener_source – (C, H, W) float32 in [-1, 1]      listener identity frame
    listener_target – (T, C, H, W) float32 in [-1, 1]   ground-truth listener frames
    speaker_mel     – (T, N_MELS) float32                speaker mel sequence
    listener_mel    – (T, N_MELS) float32                listener mel sequence
    pair_id         – str  "<speaker_id>#<listener_id>@<frame_idx>"
"""

import glob
import json
import os
import sys
import random
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

# Audio utilities (audio.py lives one level up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audio as _audio_utils

# Mel parameters (must match hparams.py)
_SR       = 16000   # sample rate
_FPS      = 25      # video fps
_HOP      = 200     # hop_size in samples
_N_MELS   = 80      # num_mels
_MEL_WIN  = 16      # mel time-frames averaged per video frame (matches Audio2Lip window)


def _find_audio(audio_dir: str, video_id: str) -> Optional[str]:
    """Try to find a .wav file matching video_id under audio_dir. Returns None if not found."""
    stem = os.path.splitext(video_id)[0]
    for ext in (".wav", ".mp3"):
        candidate = os.path.join(audio_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    # One level of subdirs
    for entry in os.scandir(audio_dir):
        if entry.is_dir():
            for ext in (".wav", ".mp3"):
                candidate = os.path.join(entry.path, stem + ext)
                if os.path.isfile(candidate):
                    return candidate
    return None


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


def _read_frame(cap: cv2.VideoCapture, frame_idx: int, size: int) -> torch.Tensor:
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


def _read_frames_uniform(cap: cv2.VideoCapture, size: int, total: int, T: int):
    """Read the full video and uniformly sample T frames to cover the entire utterance.

    This preserves the full temporal context (beginning, middle, end) of the
    speaker's utterance while fitting within the T-frame memory budget.

    Returns:
        (T, C, H, W) tensor.
    """
    # Compute which frames to sample (evenly spaced)
    if total <= T:
        indices = list(range(total))
    else:
        indices = [int(i * total / T) for i in range(T)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        frames.append(tensor)
    return torch.stack(frames) if frames else torch.zeros(0, 3, size, size)


def _read_all_frames(cap: cv2.VideoCapture, size: int, n_frames: int):
    """Read up to n_frames sequentially from the beginning of the video.

    Returns:
        (T, C, H, W) tensor where T <= n_frames.
    """
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(n_frames):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        frames.append(tensor)
    return torch.stack(frames)  # (T, C, H, W)


class AvaMERGDataset(Dataset):
    """Dyadic speaker-listener utterance-level dataset built from AvaMERG.

    Args:
        root:        Path to the AvaMERG root directory.
        split_json:  Path to the JSON split file (default: <root>/train.json).
        size:        Spatial resolution to resize frames to (default 256).
        identity_gap_min / identity_gap_max:
                     The listener identity frame is sampled from a frame that
                     is ``gap`` frames *before* the target frame, where gap is
                     drawn uniformly from [identity_gap_min, identity_gap_max].
        frames_per_video:
                     How many times each video pair appears per epoch (default 8).
        max_frames:  Maximum number of video frames to load per utterance.
                     Limits GPU memory usage. Set to 0 for no limit (default 64).
    """

    def __init__(
        self,
        root: str,
        split_json: Optional[str] = None,
        size: int = 256,
        identity_gap_min: int = 10,
        identity_gap_max: int = 30,
        frames_per_video: int = 8,
        max_frames: int = 64,
    ):
        self.root = root
        self.video_dir = os.path.join(root, "video_v5_0")
        self.audio_dir = os.path.join(root, "audio_v5_0")
        self.size = size
        self.identity_gap_min = identity_gap_min
        self.identity_gap_max = identity_gap_max
        self.frames_per_video = frames_per_video
        self.max_frames = max_frames
        # mel cache: audio_path -> (N_MELS, T_mel) ndarray
        self._mel_cache: dict = {}

        if split_json is None:
            split_json = os.path.join(root, "train.json")
        with open(split_json, "r") as f:
            raw = json.load(f)

        self.pairs = self._parse_pairs(raw)
        # Build flat index: (pair_idx, seed)
        self._index = [
            (pi, seed)
            for pi in range(len(self.pairs))
            for seed in range(frames_per_video)
        ]

    # ------------------------------------------------------------------
    def _parse_pairs(self, raw: list) -> list:
        """Normalise JSON entries into {'spk_id', 'lis_id', 'spk_path', 'lis_path', 'spk_audio_path', 'lis_audio_path'}."""
        pairs = []
        for item in raw:
            if isinstance(item, str):
                # "spk_id#lis_id" shorthand
                parts = item.split("#")
                spk_id, lis_id = parts[0], parts[-1]
                spk_path = _find_video(self.video_dir, spk_id)
                lis_path = _find_video(self.video_dir, lis_id)
                pairs.append({"spk_id": spk_id, "lis_id": lis_id,
                               "spk_path": spk_path, "lis_path": lis_path,
                               "spk_audio_path": _find_audio(self.audio_dir, spk_id),
                               "lis_audio_path": _find_audio(self.audio_dir, lis_id)})
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
                               "spk_path": spk_path, "lis_path": lis_path,
                               "spk_audio_path": _find_audio(self.audio_dir, spk_id),
                               "lis_audio_path": _find_audio(self.audio_dir, lis_id)})
            else:
                # Variant A: ID-based
                spk_id = item.get("speaker", item.get("speaker_id", ""))
                lis_id = item.get("listener", item.get("listener_id", ""))
                spk_path = _find_video(self.video_dir, spk_id)
                lis_path = _find_video(self.video_dir, lis_id)
                pairs.append({"spk_id": spk_id, "lis_id": lis_id,
                               "spk_path": spk_path, "lis_path": lis_path,
                               "spk_audio_path": _find_audio(self.audio_dir, spk_id),
                               "lis_audio_path": _find_audio(self.audio_dir, lis_id)})
        return pairs

    def _find_video_glob(self, conv_id: str, utt_idx: int):
        """dia{conv_id}utt{utt_idx}_*.mp4 pattern search. Returns None if not found."""
        pattern = os.path.join(self.video_dir, f"dia{conv_id}utt{utt_idx}_*.mp4")
        matches = glob.glob(pattern)
        return matches[0] if matches else None

    def _parse_avamerg_conversation(self, item: dict) -> list:
        """Parse one AvaMERG conversation entry into speaker-listener video pairs."""
        conv_id = item["conversation_id"]
        spk_num = item.get("speaker_profile", {}).get("ID", "")
        lis_num = item.get("listener_profile", {}).get("ID", "")

        pairs = []
        for turn in item.get("turns", []):
            history = turn.get("dialogue_history", [])
            if not history:
                continue
            last_utt = history[-1]
            spk_utt_idx = last_utt.get("index", 0)
            lis_utt_idx = spk_utt_idx + 1

            if spk_num and lis_num:
                spk_stem = f"dia{conv_id}utt{spk_utt_idx}_{spk_num}"
                lis_stem = f"dia{conv_id}utt{lis_utt_idx}_{lis_num}"
                try:
                    spk_path = _find_video(self.video_dir, spk_stem)
                    lis_path = _find_video(self.video_dir, lis_stem)
                except FileNotFoundError:
                    continue
            else:
                spk_path = self._find_video_glob(conv_id, spk_utt_idx)
                lis_path = self._find_video_glob(conv_id, lis_utt_idx)
                if not spk_path or not lis_path:
                    continue
                spk_stem = os.path.splitext(os.path.basename(spk_path))[0]
                lis_stem = os.path.splitext(os.path.basename(lis_path))[0]

            pairs.append({
                "spk_id":        spk_stem,
                "lis_id":        lis_stem,
                "spk_path":      spk_path,
                "lis_path":      lis_path,
                "spk_audio_path": _find_audio(self.audio_dir, spk_stem),
                "lis_audio_path": _find_audio(self.audio_dir, lis_stem),
            })
        return pairs

    # ------------------------------------------------------------------
    def _load_mel_at_frame(self, audio_path: Optional[str], frame_idx: int) -> torch.Tensor:
        """Return averaged mel feature (N_MELS,) for the video frame at frame_idx."""
        if audio_path is None or not os.path.isfile(audio_path):
            return torch.zeros(_N_MELS, dtype=torch.float32)

        if audio_path not in self._mel_cache:
            wav = _audio_utils.load_wav(audio_path, sr=_SR)
            mel = _audio_utils.melspectrogram(wav)       # (N_MELS, T_mel)
            self._mel_cache[audio_path] = mel

        mel = self._mel_cache[audio_path]                # (N_MELS, T_mel)
        mel_start = int(frame_idx * _SR / _FPS / _HOP)
        mel_end   = mel_start + _MEL_WIN

        if mel_start >= mel.shape[1]:
            return torch.zeros(_N_MELS, dtype=torch.float32)

        mel_slice = mel[:, mel_start:mel_end]            # (N_MELS, <=_MEL_WIN)
        if mel_slice.shape[1] < _MEL_WIN:
            pad = _MEL_WIN - mel_slice.shape[1]
            mel_slice = np.pad(mel_slice, ((0, 0), (0, pad)), mode="edge")

        avg_mel = mel_slice.mean(axis=1).astype(np.float32)  # (N_MELS,)
        return torch.from_numpy(avg_mel)

    def _load_mel_sequence_interp(self, audio_path, n_frames):
        """Load mel spectrogram and resample to n_frames via linear interpolation.

        Returns:
            (T, N_MELS) tensor where T = n_frames.
        """
        if audio_path is None or not os.path.isfile(audio_path):
            return torch.zeros(n_frames, _N_MELS)

        if audio_path not in self._mel_cache:
            wav = _audio_utils.load_wav(audio_path, sr=_SR)
            mel = _audio_utils.melspectrogram(wav)  # (N_MELS, T_mel)
            self._mel_cache[audio_path] = mel

        mel = self._mel_cache[audio_path]  # (N_MELS, T_mel)

        mel = torch.from_numpy(mel).float().unsqueeze(0)  # (1, N_MELS, T_mel)

        mel_resampled = F.interpolate(
            mel,
            size=n_frames,
            mode="linear",
            align_corners=False
        )  # (1, N_MELS, T_video)

        mel_resampled = mel_resampled.squeeze(0).permute(1, 0)  # (T, N_MELS)
        return mel_resampled

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

        # Fixed T for batch collation.
        T = self.max_frames if self.max_frames > 0 else min(n_spk, n_lis)

        # Speaker: uniform sampling from FULL utterance → T frames.
        # Preserves temporal context of the entire speaker utterance.
        speaker_frames = _read_frames_uniform(cap_spk, self.size, n_spk, T)

        # Listener GT: uniform sampling from FULL utterance → T frames.
        # Paired with speaker — both cover their entire respective utterances.
        listener_targets = _read_frames_uniform(cap_lis, self.size, n_lis, T)

        # Listener identity: random frame from the listener video
        gap = random.randint(self.identity_gap_min, self.identity_gap_max)
        identity_idx = random.randint(0, max(0, n_lis - 1))
        listener_source = _read_frame(cap_lis, identity_idx, self.size)

        cap_spk.release()
        cap_lis.release()

        # Pad to fixed length T if video was shorter
        if speaker_frames.shape[0] < T:
            pad_n = T - speaker_frames.shape[0]
            speaker_frames = torch.cat(
                [speaker_frames, speaker_frames[-1:].expand(pad_n, -1, -1, -1)], dim=0
            )
        if listener_targets.shape[0] < T:
            pad_n = T - listener_targets.shape[0]
            listener_targets = torch.cat(
                [listener_targets, listener_targets[-1:].expand(pad_n, -1, -1, -1)], dim=0
            )

        speaker_mel = self._load_mel_sequence_interp(pair.get("spk_audio_path"), T)
        listener_mel = self._load_mel_sequence_interp(pair.get("lis_audio_path"), T)

        return {
            "speaker_video":   speaker_frames,    # (T, C, H, W)
            "listener_source": listener_source,    # (C, H, W)
            "listener_target": listener_targets,   # (T, C, H, W)
            "speaker_mel":     speaker_mel,        # (T, N_MELS)
            "listener_mel":    listener_mel,       # (T, N_MELS)
            "pair_id": f"{pair['spk_id']}#{pair['lis_id']}@{identity_idx}",
            "dia_num": f"{pair['lis_path']}"
        }
