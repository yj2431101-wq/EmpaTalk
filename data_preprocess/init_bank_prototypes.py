"""Extract listener feature prototypes from training data via K-means.

Runs the frozen pretrained encoder over listener frames from AvaMERG,
collects fc(wa_L) features, then clusters them with K-means to produce
K prototype vectors for each bank (pose, exp, lip, audio).

Usage:
    python data_preprocess/init_bank_prototypes.py \
        --pretrained_ckpt checkpoints/EDTalk.pt \
        --avamerg_root /mnt/data/dataset/AvaMERG \
        --split_json   /mnt/data/dataset/AvaMERG/train.json \
        --num_prototypes 32 \
        --max_samples 5000 \
        --out bank_prototypes.pt

Output:
    bank_prototypes.pt containing:
        prototypes: (K, 512) — K-means centroids of fc(wa_L)
        feature_mean: (512,) — mean of all fc(wa_L) features
        feature_std:  (512,) — std  of all fc(wa_L) features
"""

import argparse
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from networks.encoder import Encoder
from networks.generator import Generator


def _find_video(video_dir, video_id):
    direct = os.path.join(video_dir, video_id)
    if os.path.isfile(direct):
        return direct
    direct_mp4 = direct if direct.endswith(".mp4") else direct + ".mp4"
    if os.path.isfile(direct_mp4):
        return direct_mp4
    for entry in os.scandir(video_dir):
        if entry.is_dir():
            candidate = os.path.join(entry.path, video_id)
            if os.path.isfile(candidate):
                return candidate
            candidate_mp4 = candidate if candidate.endswith(".mp4") else candidate + ".mp4"
            if os.path.isfile(candidate_mp4):
                return candidate_mp4
    return None


def collect_listener_features(args, device):
    """Encode listener frames and return fc(wa_L) features as numpy array."""

    # Build a minimal generator just for the encoder + fc
    gen = Generator(
        size=args.size,
        style_dim=512,
        lip_dim=20, pose_dim=6, exp_dim=10,
        channel_multiplier=1,
    ).to(device)

    # Load pretrained weights
    ckpt = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("gen", ckpt)
    compatible = {k: v for k, v in state.items() if not k.startswith("listener_bank")}
    gen.load_state_dict(compatible, strict=False)
    gen.eval()

    # Parse JSON to get listener video paths
    with open(args.split_json, "r") as f:
        raw = json.load(f)

    video_dir = os.path.join(args.avamerg_root, "video_v5_0")
    lis_paths = set()
    for item in raw:
        if isinstance(item, dict):
            if "listener" in item:
                p = _find_video(video_dir, item["listener"])
                if p: lis_paths.add(p)
            elif "listener_video" in item:
                p = os.path.join(args.avamerg_root, item["listener_video"])
                if os.path.isfile(p): lis_paths.add(p)
            elif "conversation_id" in item:
                # AvaMERG conversation format — extract listener videos
                conv_id = item["conversation_id"]
                lis_num = item.get("listener_profile", {}).get("ID", "")
                for turn in item.get("turns", []):
                    history = turn.get("dialogue_history", [])
                    if not history:
                        continue
                    last_utt = history[-1]
                    lis_utt_idx = last_utt.get("index", 0) + 1
                    if lis_num:
                        stem = f"dia{conv_id}utt{lis_utt_idx}_{lis_num}"
                        p = _find_video(video_dir, stem)
                        if p: lis_paths.add(p)

    lis_paths = list(lis_paths)
    random.shuffle(lis_paths)
    print(f"Found {len(lis_paths)} listener videos")

    # Extract features
    all_features = []
    frames_per_video = max(1, args.max_samples // max(1, len(lis_paths)))

    with torch.no_grad():
        for vpath in tqdm(lis_paths, desc="Extracting features"):
            if len(all_features) >= args.max_samples:
                break

            cap = cv2.VideoCapture(vpath)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n_frames < 2:
                cap.release()
                continue

            # Sample random frames from this video
            indices = random.sample(range(n_frames), min(frames_per_video, n_frames))
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, bgr = cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_AREA)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                tensor = (tensor - 0.5) / 0.5
                tensor = tensor.unsqueeze(0).to(device)

                wa_L, _, _, _ = gen.enc(tensor, None)
                feat = gen.fc(wa_L)  # (1, 512)
                all_features.append(feat.cpu().numpy())

                if len(all_features) >= args.max_samples:
                    break
            cap.release()

    features = np.concatenate(all_features, axis=0)  # (N, 512)
    print(f"Collected {features.shape[0]} feature vectors")
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Extract K-means bank prototypes from listener training data"
    )
    parser.add_argument("--pretrained_ckpt", required=True)
    parser.add_argument("--avamerg_root", required=True)
    parser.add_argument("--split_json", default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_prototypes", type=int, default=32,
                        help="K: number of prototype clusters")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="max listener frames to encode")
    parser.add_argument("--out", default="bank_prototypes.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.split_json is None:
        args.split_json = os.path.join(args.avamerg_root, "train.json")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Collect features
    features = collect_listener_features(args, device)

    # 2. K-means clustering
    print(f"Running K-means (K={args.num_prototypes})...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.num_prototypes,
        batch_size=min(1024, features.shape[0]),
        n_init=3,
        random_state=42,
    )
    kmeans.fit(features)
    centroids = kmeans.cluster_centers_  # (K, 512)

    # 3. Compute statistics
    feat_mean = features.mean(axis=0)
    feat_std  = features.std(axis=0)

    # 4. Save
    torch.save({
        "prototypes":   torch.from_numpy(centroids).float(),   # (K, 512)
        "feature_mean": torch.from_numpy(feat_mean).float(),   # (512,)
        "feature_std":  torch.from_numpy(feat_std).float(),    # (512,)
        "num_samples":  features.shape[0],
    }, args.out)
    print(f"Saved {args.num_prototypes} prototypes -> {args.out}")
    print(f"  feature mean norm: {np.linalg.norm(feat_mean):.4f}")
    print(f"  feature std  mean: {feat_std.mean():.4f}")
    print(f"  centroid norms: min={np.linalg.norm(centroids, axis=1).min():.4f}, "
          f"max={np.linalg.norm(centroids, axis=1).max():.4f}")


if __name__ == "__main__":
    main()
