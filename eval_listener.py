"""AvaMERG test.json 기반 배치 평가 스크립트

Usage:
    python eval_listener.py \
        --ckpt      listener_exp/v1/checkpoint/166000.pt \
        --pretrained ckpts/ckpts/EDTalk.pt \
        --avamerg_root /mnt/HDD_raid1/AvaMERG_jhchoi/AvaMERG \
        --test_json  /mnt/HDD_raid1/AvaMERG_jhchoi/AvaMERG/test.json \
        --out_dir    ./eval_output/v1 \
        --n_pairs    20 \
        --side_by_side
"""
import argparse
import glob
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

from datasets.dataset_avamerg import _find_video, _find_audio
from train.trainer_listener import TrainerListener


# ── helpers ──────────────────────────────────────────────────────────────────

def read_frame(cap, frame_idx, size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    if not ok:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (t - 0.5) / 0.5  # [-1, 1]


def tensor_to_bgr(t, size):
    img = (t.clamp(-1, 1) + 1) / 2
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (size, size))


def _build_video_index(video_dir):
    """video_dir의 모든 .mp4를 한 번에 스캔 → {stem: path} dict."""
    index = {}
    for fname in os.listdir(video_dir):
        if fname.endswith(".mp4"):
            stem = os.path.splitext(fname)[0]
            index[stem] = os.path.join(video_dir, fname)
    return index


def _find_video_glob(video_index, conv_id, utt_idx):
    """pre-built index에서 dia{conv_id}utt{utt_idx}_* 탐색."""
    prefix = f"dia{conv_id}utt{utt_idx}_"
    for stem, path in video_index.items():
        if stem.startswith(prefix):
            return path
    return None


def parse_test_json(test_json, avamerg_root):
    """test.json → list of (spk_path, lis_path, pair_id)"""
    video_dir = os.path.join(avamerg_root, "video_v5_0")
    with open(test_json) as f:
        raw = json.load(f)

    # 비디오 디렉토리 한 번만 스캔
    video_index = _build_video_index(video_dir)
    print(f"  Video index built: {len(video_index)} files")

    pairs = []
    for item in raw:
        try:
            if isinstance(item, str):
                spk_id, lis_id = item.split("#")[0], item.split("#")[-1]
                spk_path = _find_video(video_dir, spk_id)
                lis_path = _find_video(video_dir, lis_id)
                pairs.append((spk_path, lis_path, f"{spk_id}#{lis_id}"))

            elif "speaker_video" in item:
                spk_id = os.path.splitext(os.path.basename(item["speaker_video"]))[0]
                lis_id = os.path.splitext(os.path.basename(item["listener_video"]))[0]
                spk_path = _find_video(video_dir, spk_id)
                lis_path = _find_video(video_dir, lis_id)
                pairs.append((spk_path, lis_path, f"{spk_id}#{lis_id}"))

            elif "conversation_id" in item:
                conv_id = item["conversation_id"]
                for turn in item.get("turns", []):
                    history = turn.get("dialogue_history", [])
                    if not history:
                        continue
                    last = history[-1]
                    spk_utt_idx = last.get("index", 0)
                    lis_utt_idx = spk_utt_idx + 1

                    spk_path = _find_video_glob(video_index, conv_id, spk_utt_idx)
                    lis_path = _find_video_glob(video_index, conv_id, lis_utt_idx)
                    if spk_path and lis_path:
                        pair_id = f"dia{conv_id}utt{spk_utt_idx}_vs_utt{lis_utt_idx}"
                        pairs.append((spk_path, lis_path, pair_id))

            else:
                spk_id = item.get("speaker", item.get("speaker_id", ""))
                lis_id = item.get("listener", item.get("listener_id", ""))
                spk_path = _find_video(video_dir, spk_id)
                lis_path = _find_video(video_dir, lis_id)
                pairs.append((spk_path, lis_path, f"{spk_id}#{lis_id}"))

        except (FileNotFoundError, KeyError):
            continue
    return pairs


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         required=True,  help="리스너 체크포인트 .pt")
    parser.add_argument("--pretrained",   required=True,  help="EDTalk pretrained .pt")
    parser.add_argument("--avamerg_root", required=True,  help="AvaMERG 루트 디렉토리")
    parser.add_argument("--test_json",    required=True,  help="test.json 경로")
    parser.add_argument("--out_dir",      default="./eval_output")
    parser.add_argument("--n_pairs",      type=int, default=20,  help="평가할 페어 수 (0=전체)")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--side_by_side", action="store_true", help="화자|청자 나란히 저장")
    parser.add_argument("--motion_scale", type=float, default=1.0)
    parser.add_argument("--z_momentum",   type=float, default=0.9)
    parser.add_argument("--use_prior",    action="store_true")
    cli = parser.parse_args()

    os.makedirs(cli.out_dir, exist_ok=True)
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    # ── 1. 체크포인트 로드 ──────────────────────────────────────────────
    print(f"Loading checkpoint: {cli.ckpt}")
    ckpt = torch.load(cli.ckpt, map_location=device, weights_only=False)
    args = ckpt["args"]
    args.pretrained_ckpt = cli.pretrained
    args.resume_ckpt = None

    trainer = TrainerListener(args, device)
    missing, _ = trainer.gen.listener_bank.load_state_dict(ckpt["listener_bank"], strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} missing keys (randomly init'd) — old ckpt without audio modules")
    trainer.gen.eval()
    size = args.size
    print(f"  Loaded (step {ckpt.get('start_iter', '?')})")

    # ── 2. test.json 파싱 ───────────────────────────────────────────────
    pairs = parse_test_json(cli.test_json, cli.avamerg_root)
    if cli.n_pairs > 0:
        pairs = pairs[:cli.n_pairs]
    print(f"Evaluating {len(pairs)} pairs → {cli.out_dir}")

    # ── 3. 페어별 생성 ──────────────────────────────────────────────────
    for idx, (spk_path, lis_path, pair_id) in enumerate(pairs):
        safe_id = pair_id.replace("/", "_").replace("#", "_vs_")
        out_path = os.path.join(cli.out_dir, f"{idx:04d}_{safe_id}.mp4")

        cap_spk = cv2.VideoCapture(spk_path)
        cap_lis = cv2.VideoCapture(lis_path)
        n_frames = int(cap_spk.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_spk.get(cv2.CAP_PROP_FPS) or 25.0

        # 리스너 identity: 첫 프레임
        lis_src = read_frame(cap_lis, 0, size)
        if lis_src is None:
            cap_spk.release(); cap_lis.release()
            continue
        lis_src = lis_src.unsqueeze(0).to(device)

        out_w = size * 2 if cli.side_by_side else size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, size))

        z_pose = z_exp = None
        for i in range(n_frames):
            spk_t = read_frame(cap_spk, i, size)
            if spk_t is None:
                break
            spk_t = spk_t.unsqueeze(0).to(device)

            with torch.no_grad():
                if cli.use_prior:
                    pred, z_pose, z_exp = trainer.sample_prior(
                        spk_t, lis_src,
                        motion_scale=cli.motion_scale,
                        z_pose=z_pose, z_exp=z_exp,
                        z_momentum=cli.z_momentum,
                    )
                else:
                    pred = trainer.sample(spk_t, lis_src, motion_scale=cli.motion_scale)

            pred_bgr = tensor_to_bgr(pred[0], size)
            if cli.side_by_side:
                spk_bgr = tensor_to_bgr(spk_t[0], size)
                frame = np.concatenate([spk_bgr, pred_bgr], axis=1)
            else:
                frame = pred_bgr
            writer.write(frame)

        cap_spk.release()
        cap_lis.release()
        writer.release()
        print(f"  [{idx+1}/{len(pairs)}] {pair_id} → {os.path.basename(out_path)}")

    print(f"\nDone. Results saved to: {cli.out_dir}")


if __name__ == "__main__":
    main()
