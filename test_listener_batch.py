"""Batch test script for ListenerBank on AvaMERG test set.

Generates listener reaction videos for each speaker-listener pair in test.json.
Saves predicted, GT, and speaker videos in separate folders for evaluation.

Output structure:
    out_dir/
        pred/       <- generated listener reaction videos
        gt/         <- GT listener videos (copied from dataset)
        speaker/    <- speaker videos (copied from dataset)

Usage:
    python test_listener_batch.py \
        --ckpt       /path/to/002000.pt \
        --pretrained /path/to/EDTalk.pt \
        --test_json  /mnt/HDD_raid1/AvaMERG_jhchoi/AvaMERG/test.json \
        --video_root /mnt/HDD_raid1/AvaMERG_jhchoi/AvaMERG/video_v5_0 \
        --out_dir    /mnt/data/dataset/AvaMERG_test/020000/ \
        --device     cuda:1
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train.trainer_listener import TrainerListener


# ── Helpers ─────────────────────────────────────────────────────────

def read_frame(cap, frame_idx, size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    if not ok:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
        _, bgr = cap.read()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (t - 0.5) / 0.5  # [-1, 1]


def tensor_to_bgr(t, size):
    img = (t.clamp(-1, 1) + 1) / 2
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (size, size))


def find_video(video_dir, video_id):
    """Resolve video_id to a .mp4 path under video_dir."""
    direct = os.path.join(video_dir, video_id)
    if os.path.isfile(direct):
        return direct
    direct_mp4 = direct if direct.endswith(".mp4") else direct + ".mp4"
    if os.path.isfile(direct_mp4):
        return direct_mp4
    return None


def build_video_index(video_dir):
    """Scan video_dir once and build a filename -> path lookup dict.

    This avoids repeated glob/scandir calls on large directories.
    """
    print(f"Building video index from {video_dir} ...")
    index = {}
    for entry in os.scandir(video_dir):
        if entry.is_file() and entry.name.endswith(".mp4"):
            stem = os.path.splitext(entry.name)[0]
            index[entry.name] = entry.path
            index[stem] = entry.path
        elif entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_file() and sub.name.endswith(".mp4"):
                    stem = os.path.splitext(sub.name)[0]
                    index[sub.name] = sub.path
                    index[stem] = sub.path
    print(f"  Indexed {len(index) // 2} video files")
    return index


def find_video_from_index(index, video_id):
    """Look up video_id in the pre-built index."""
    stem = os.path.splitext(video_id)[0]
    return index.get(video_id) or index.get(stem)


def find_video_glob_from_index(index, conv_id, utt_idx):
    """Find dia{conv_id}utt{utt_idx}_*.mp4 using the pre-built index."""
    prefix = f"dia{conv_id}utt{utt_idx}_"
    for key, path in index.items():
        if key.startswith(prefix) and key.endswith(".mp4"):
            return path
    return None


def copy_video_resized(src_path, dst_path, size):
    """Copy a video, resizing each frame to (size, size)."""
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_path, fourcc, fps, (size, size))

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        writer.write(bgr)

    cap.release()
    writer.release()


# ── Parse test.json ─────────────────────────────────────────────────

def parse_test_json(json_path, video_dir, video_index):
    """Parse test.json into list of {'spk_id', 'lis_id', 'spk_path', 'lis_path'}.

    Uses pre-built video_index for fast lookups (no repeated glob/scandir).
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    pairs = []
    for item in raw:
        if isinstance(item, str):
            parts = item.split("#")
            spk_id, lis_id = parts[0], parts[-1]
            spk_path = find_video_from_index(video_index, spk_id)
            lis_path = find_video_from_index(video_index, lis_id)
            if spk_path and lis_path:
                pairs.append({"spk_id": spk_id, "lis_id": lis_id,
                               "spk_path": spk_path, "lis_path": lis_path})

        elif isinstance(item, dict) and "conversation_id" in item:
            conv_id = item["conversation_id"]
            spk_num = item.get("speaker_profile", {}).get("ID", "")
            lis_num = item.get("listener_profile", {}).get("ID", "")
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
                    spk_path = find_video_from_index(video_index, spk_stem)
                    lis_path = find_video_from_index(video_index, lis_stem)
                else:
                    spk_path = find_video_glob_from_index(video_index, conv_id, spk_utt_idx)
                    lis_path = find_video_glob_from_index(video_index, conv_id, lis_utt_idx)
                    spk_stem = os.path.splitext(os.path.basename(spk_path))[0] if spk_path else ""
                    lis_stem = os.path.splitext(os.path.basename(lis_path))[0] if lis_path else ""

                if spk_path and lis_path:
                    pairs.append({"spk_id": spk_stem, "lis_id": lis_stem,
                                   "spk_path": spk_path, "lis_path": lis_path})

        elif isinstance(item, dict) and "speaker_video" in item:
            root = os.path.dirname(json_path)
            spk_path = os.path.join(root, item["speaker_video"])
            lis_path = os.path.join(root, item["listener_video"])
            spk_id = os.path.splitext(os.path.basename(spk_path))[0]
            lis_id = os.path.splitext(os.path.basename(lis_path))[0]
            if os.path.isfile(spk_path) and os.path.isfile(lis_path):
                pairs.append({"spk_id": spk_id, "lis_id": lis_id,
                               "spk_path": spk_path, "lis_path": lis_path})

        elif isinstance(item, dict):
            spk_id = item.get("speaker", item.get("speaker_id", ""))
            lis_id = item.get("listener", item.get("listener_id", ""))
            spk_path = find_video_from_index(video_index, spk_id)
            lis_path = find_video_from_index(video_index, lis_id)
            if spk_path and lis_path:
                pairs.append({"spk_id": spk_id, "lis_id": lis_id,
                               "spk_path": spk_path, "lis_path": lis_path})

    return pairs


# ── Generate one pair ───────────────────────────────────────────────

@torch.no_grad()
def generate_one_pair(gen, lb, training_mode, pair, size, device,
                      motion_scale=1.0, smooth=0.0, use_prior=False,
                      z_momentum=0.9):
    """Generate listener reaction video for one speaker-listener pair.

    Args:
        use_prior: If True, sample z from N(0,I) with OU temporal smoothing
                   instead of deterministic VAE inference. Produces more
                   dynamic reactions when the bank is undertrained.
    """
    cap_spk = cv2.VideoCapture(pair["spk_path"])
    cap_lis = cv2.VideoCapture(pair["lis_path"])

    n_spk = int(cap_spk.get(cv2.CAP_PROP_FRAME_COUNT))
    n_lis = int(cap_lis.get(cv2.CAP_PROP_FRAME_COUNT))

    if n_spk < 2:
        cap_spk.release()
        cap_lis.release()
        return [], 25.0

    fps = cap_spk.get(cv2.CAP_PROP_FPS) or 25.0

    # Listener identity: random frame from the listener video
    identity_idx = random.randint(0, max(0, n_lis - 1))
    listener_src = read_frame(cap_lis, identity_idx, size).unsqueeze(0).to(device)
    pair["identity_idx"] = identity_idx

    # Listener appearance for decoding
    wa_L, _, feats_L, _ = gen.enc(listener_src, None, None)

    frames_out = []
    prev_pred = None
    z_pose = z_exp = None
    ldim = lb.pose_vae.latent_dim

    # Read speaker frames sequentially (seek-based reading can fail on some codecs)
    cap_spk.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(n_spk):
        ok, bgr = cap_spk.read()
        if not ok or bgr is None:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        spk_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        spk_tensor = (spk_tensor - 0.5) / 0.5
        spk_t = spk_tensor.unsqueeze(0).to(device)

        latent_poseD_S, wa_S, f_pose_S, f_exp_S = gen._speaker_latent(spk_t)
        B = 1

        if use_prior:
            # Sample from VAE prior with OU process for temporal smoothness
            noise_scale = (1.0 - z_momentum ** 2) ** 0.5
            if z_pose is None:
                z_pose = torch.randn(B, ldim, device=device)
                z_exp  = torch.randn(B, ldim, device=device)
            else:
                z_pose = z_momentum * z_pose + noise_scale * torch.randn_like(z_pose)
                z_exp  = z_momentum * z_exp  + noise_scale * torch.randn_like(z_exp)

            f_pose = lb.pose_vae.decode(z_pose, wa_L)
            f_exp  = lb.exp_vae.decode(z_exp, wa_L)

        elif training_mode == 'passive':
            f_pose, f_exp = lb.forward_passive(latent_poseD_S, f_pose_S, f_exp_S)

        else:
            f_pose, f_exp, _, *_ = lb.forward_active(
                latent_poseD_S, wa_S=wa_L, training=False,
                listener_mel=None,
            )

        alpha_D_lip  = torch.zeros(B, gen.lip_dim, device=device)
        alpha_D_pose = gen.pose_fc(f_pose) * motion_scale
        alpha_D_exp  = gen.exp_fc(f_exp)   * motion_scale

        alpha_D_L = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a_L = gen.direction_exp.get_shared_out(alpha_D_L, gen.direction_lipnonlip.weight)
        e_L = gen.direction_exp.get_exp_latent(a_L)
        directions_D_L = gen.direction_exp(alpha_D_L, gen.direction_lipnonlip.weight)
        latent_poseD_L = wa_L + directions_D_L

        pred = gen.dec(latent_poseD_L, feats_L, e_L)

        if smooth > 0.0 and prev_pred is not None:
            pred = smooth * prev_pred + (1.0 - smooth) * pred
        prev_pred = pred.clone()

        frames_out.append(tensor_to_bgr(pred[0], size))

    cap_spk.release()
    cap_lis.release()
    return frames_out, fps


def save_video(frames, path, fps, size):
    """Save list of BGR frames as mp4."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (size, size))
    for f in frames:
        writer.write(f)
    writer.release()


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch test ListenerBank on AvaMERG test set")
    parser.add_argument("--ckpt",         required=True, help="Listener checkpoint .pt (e.g. 002000.pt)")
    parser.add_argument("--pretrained",   required=True, help="EDTalk pretrained .pt")
    parser.add_argument("--test_json",    required=True, help="Path to test.json")
    parser.add_argument("--video_root",   required=True, help="Path to video_v5_0 directory")
    parser.add_argument("--out_dir",      required=True, help="Output directory")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--motion_scale", type=float, default=1.0)
    parser.add_argument("--smooth",       type=float, default=0.0,
                        help="Temporal smoothing (0=none, 0.5~0.8 recommended)")
    parser.add_argument("--use_prior",    action="store_true",
                        help="Sample from VAE prior N(0,I) for diverse reactions")
    parser.add_argument("--z_momentum",   type=float, default=0.9,
                        help="Prior z temporal continuity (0=random, 0.9=smooth)")
    parser.add_argument("--max_pairs",    type=int, default=0,
                        help="Max pairs to test (0 = all)")
    parser.add_argument("--copy_gt",      action="store_true", default=True,
                        help="Copy GT and speaker videos to output dir (for evaluation)")
    parser.add_argument("--no_copy_gt",   action="store_false", dest="copy_gt",
                        help="Skip copying GT/speaker videos")
    cli = parser.parse_args()

    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    # Create output directories
    pred_dir    = os.path.join(cli.out_dir, "pred")
    gt_dir      = os.path.join(cli.out_dir, "gt")
    speaker_dir = os.path.join(cli.out_dir, "speaker")
    os.makedirs(pred_dir, exist_ok=True)
    if cli.copy_gt:
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(speaker_dir, exist_ok=True)

    # ── 1. Load model ──
    print(f"Loading checkpoint: {cli.ckpt}")
    ckpt = torch.load(cli.ckpt, map_location=device, weights_only=False)
    args = ckpt["args"]
    args.pretrained_ckpt = cli.pretrained
    args.resume_ckpt = None

    trainer = TrainerListener(args, device)
    missing, _ = trainer.gen.listener_bank.load_state_dict(
        ckpt["listener_bank"], strict=False
    )
    if missing:
        print(f"  [WARN] missing keys: {len(missing)}")
    trainer.gen.eval()

    size = args.size
    gen = trainer._raw_gen
    lb = gen.listener_bank
    training_mode = trainer.training_mode
    print(f"  Model loaded (step {ckpt.get('start_iter', '?')}, mode={training_mode})")

    # ── 2. Parse test.json ──
    video_index = build_video_index(cli.video_root)
    pairs = parse_test_json(cli.test_json, cli.video_root, video_index)
    print(f"Found {len(pairs)} test pairs")

    if cli.max_pairs > 0:
        pairs = pairs[:cli.max_pairs]
        print(f"  Limited to {len(pairs)} pairs")

    # ── 3. Generate ──
    success = 0
    for idx, pair in enumerate(tqdm(pairs, desc="Generating")):
        out_name = f"{pair['spk_id']}_to_{pair['lis_id']}.mp4"

        # Generate predicted listener video
        frames, fps = generate_one_pair(
            gen, lb, training_mode, pair, size, device,
            motion_scale=cli.motion_scale,
            smooth=cli.smooth,
            use_prior=cli.use_prior,
            z_momentum=cli.z_momentum,
        )

        if not frames:
            tqdm.write(f"  [SKIP] {pair['spk_id']} -> {pair['lis_id']} (no frames)")
            continue

        tqdm.write(f"  [{idx+1}/{len(pairs)}] {out_name} ({len(frames)} frames)")

        # Save predicted listener video
        save_video(frames, os.path.join(pred_dir, out_name), fps, size)

        # Copy GT listener and speaker videos (resized to match pred)
        if cli.copy_gt:
            copy_video_resized(pair["lis_path"], os.path.join(gt_dir, out_name), size)
            copy_video_resized(pair["spk_path"], os.path.join(speaker_dir, out_name), size)

        success += 1

    # ── 4. Summary ──
    print(f"\n{'='*60}")
    print(f"Done: {success}/{len(pairs)} pairs generated")
    print(f"  pred/    : {pred_dir}")
    if cli.copy_gt:
        print(f"  gt/      : {gt_dir}")
        print(f"  speaker/ : {speaker_dir}")
    print(f"\nEvaluation (exp/pose only, no lip):")
    print(f"  Compare pred/ vs gt/ using 3DMM coefficient extraction")
    print(f"  e.g. DECA/EMOCA -> extract pose (6-dim) and exp (50-dim)")
    print(f"       then compute FD, MSE, or correlation between pred and gt")


if __name__ == "__main__":
    main()
