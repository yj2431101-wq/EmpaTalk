"""
화자 비디오 → 리스너 반응 비디오 생성 스크립트

Usage:
    python demo_listener.py \
        --ckpt      /path/to/068000.pt \
        --pretrained /path/to/EDTalk.pt \
        --speaker   /path/to/speaker.mp4 \
        --listener_ref /path/to/listener_ref.mp4  (or .jpg/.png) \
        --out       ./listener_output.mp4
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train.trainer_listener import TrainerListener


def read_frame(cap, frame_idx, size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    if not ok:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = (t - 0.5) / 0.5  # [-1, 1]
    return t


def tensor_to_bgr(t, size):
    """(3, H, W) in [-1, 1] → (H, W, 3) uint8 BGR"""
    img = (t.clamp(-1, 1) + 1) / 2  # [0, 1]
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (size, size))
    return img


def load_image_as_tensor(path, size):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = (t - 0.5) / 0.5
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         required=True, help="listener 체크포인트 .pt")
    parser.add_argument("--pretrained",   required=True, help="EDTalk pretrained .pt")
    parser.add_argument("--speaker",      required=True, help="화자 비디오 경로")
    parser.add_argument("--listener_ref", required=True, help="리스너 레퍼런스 (비디오 or 이미지)")
    parser.add_argument("--out",          default="./listener_output.mp4")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--n_frames",     type=int, default=60, help="화자가 이미지일 때 생성할 프레임 수")
    parser.add_argument("--side_by_side", action="store_true", help="화자|리스너 나란히 저장")
    parser.add_argument("--use_prior",    action="store_true", help="VAE prior N(0,I)에서 샘플링 (posterior 대신, 더 다양한 출력)")
    parser.add_argument("--motion_scale", type=float, default=1.0, help="motion 계수 스케일 (>1이면 움직임 증폭, 예: 3.0)")
    parser.add_argument("--smooth",       type=float, default=0.0, help="temporal smoothing (0=없음, 0.5~0.8 권장, 1=완전고정)")
    parser.add_argument("--z_momentum",   type=float, default=0.9, help="prior z 시간축 연속성 (0=매프레임 독립 랜덤, 0.9=부드럽게 변화, 1=완전고정)")
    parser.add_argument("--debug",        action="store_true", help="첫 프레임에서 중간 텐서 norm 출력")
    cli = parser.parse_args()

    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. 체크포인트 로드 ──────────────────────────────────────────────
    print(f"Loading checkpoint: {cli.ckpt}")
    ckpt = torch.load(cli.ckpt, map_location=device, weights_only=False)
    args = ckpt["args"]
    args.pretrained_ckpt = cli.pretrained
    args.resume_ckpt = None

    # ── 2. Trainer 초기화 ───────────────────────────────────────────────
    print("Initialising TrainerListener...")
    trainer = TrainerListener(args, device)
    missing, unexpected = trainer.gen.listener_bank.load_state_dict(ckpt["listener_bank"], strict=False)
    if missing:
        print(f"  [WARN] missing keys (randomly init'd): {len(missing)} — checkpoint may predate VAE")
    if unexpected:
        print(f"  [WARN] unexpected keys: {unexpected}")
    trainer.gen.eval()
    print(f"  listener_bank loaded (step {ckpt.get('start_iter', '?')})")

    size = args.size

    # ── 3. 화자 입력 로드 (비디오 or 이미지) ───────────────────────────
    spk_ext = os.path.splitext(cli.speaker)[1].lower()
    if spk_ext in (".jpg", ".jpeg", ".png", ".bmp"):
        spk_static = load_image_as_tensor(cli.speaker, size).to(device)
        cap_spk = None
        n_frames = cli.n_frames
        fps = 25.0
        print(f"Speaker: image → {n_frames} frames")
    else:
        spk_static = None
        cap_spk = cv2.VideoCapture(cli.speaker)
        n_frames = int(cap_spk.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_spk.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"Speaker video: {n_frames} frames @ {fps:.1f} fps")

    # ── 4. 리스너 레퍼런스 로드 ─────────────────────────────────────────
    ext = os.path.splitext(cli.listener_ref)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp"):
        listener_src = load_image_as_tensor(cli.listener_ref, size).to(device)
        cap_lis = None
        print(f"Listener ref: image ({cli.listener_ref})")
    else:
        cap_lis = cv2.VideoCapture(cli.listener_ref)
        listener_src = read_frame(cap_lis, 0, size).to(device)
        print(f"Listener ref: video first frame ({cli.listener_ref})")

    # ── 5. 출력 비디오 writer ───────────────────────────────────────────
    out_w = size * 2 if cli.side_by_side else size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cli.out, fourcc, fps, (out_w, size))

    # ── 6. 프레임별 추론 ────────────────────────────────────────────────
    print("Generating listener video...")
    prev_pred = None
    z_pose = z_exp = None
    for i in range(n_frames):
        if spk_static is not None:
            spk_t = spk_static.unsqueeze(0)
        else:
            spk_t = read_frame(cap_spk, i, size)
            if spk_t is None:
                break
            spk_t = spk_t.unsqueeze(0).to(device)  # (1, 3, H, W)
        lis_t = listener_src.unsqueeze(0)          # (1, 3, H, W)

        with torch.no_grad():
            if cli.use_prior:
                pred, z_pose, z_exp = trainer.sample_prior(
                    spk_t, lis_t,
                    motion_scale=cli.motion_scale,
                    z_pose=z_pose, z_exp=z_exp,
                    z_momentum=cli.z_momentum,
                )
            else:
                pred = trainer.sample(spk_t, lis_t, motion_scale=cli.motion_scale,
                                      debug=(cli.debug and i == 0))

        if cli.smooth > 0.0 and prev_pred is not None:
            pred = cli.smooth * prev_pred + (1.0 - cli.smooth) * pred
        prev_pred = pred.clone()

        pred_bgr = tensor_to_bgr(pred[0], size)

        if cli.side_by_side:
            spk_bgr = tensor_to_bgr(spk_t[0], size)
            frame = np.concatenate([spk_bgr, pred_bgr], axis=1)
        else:
            frame = pred_bgr

        writer.write(frame)

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{n_frames} frames")

    if cap_spk:
        cap_spk.release()
    if cap_lis:
        cap_lis.release()
    writer.release()
    print(f"Saved → {cli.out}")


if __name__ == "__main__":
    main()
