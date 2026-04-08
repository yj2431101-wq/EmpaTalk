"""
Speaker video + Listener audio -> Listener reaction video generation

Supports two audio use-cases:
  1. Listener audio only  -> pose/exp conditioning via mel_proj (same path as training)
  2. Listener audio + Audio2Lip ckpt -> additionally drives lip sync

Usage:
    python demo_listener.py \
        --ckpt      /path/to/068000.pt \
        --pretrained /path/to/EDTalk.pt \
        --speaker   /path/to/speaker.mp4 \
        --listener_ref /path/to/listener_ref.mp4 \
        --listener_audio /path/to/listener.wav \
        --audio2lip_ckpt /path/to/Audio2Lip.pt \
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
import audio as audio_utils

# Mel parameters (must match hparams.py / dataset)
_SR      = 16000
_FPS     = 25
_HOP     = 200
_N_MELS  = 80
_MEL_WIN = 16  # mel frames per video frame (Audio2Lip window)


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
    """(3, H, W) in [-1, 1] -> (H, W, 3) uint8 BGR"""
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


def load_mel_sequence(audio_path: str, n_frames: int) -> torch.Tensor:
    """Load audio -> mel spectrogram, resample to n_frames via interpolation.

    Returns:
        (n_frames, N_MELS) float32 tensor.
    """
    wav = audio_utils.load_wav(audio_path, sr=_SR)
    mel = audio_utils.melspectrogram(wav)  # (N_MELS, T_mel)
    mel_t = torch.from_numpy(mel).unsqueeze(0)  # (1, N_MELS, T_mel)
    mel_resampled = F.interpolate(
        mel_t, size=n_frames, mode="linear", align_corners=False
    )  # (1, N_MELS, n_frames)
    return mel_resampled.squeeze(0).permute(1, 0)  # (n_frames, N_MELS)


def load_mel_for_audio2lip(audio_path: str, n_frames: int) -> torch.Tensor:
    """Load audio -> mel windows for Audio2Lip input.

    Audio2Lip expects (B*T, 1, 80, 16) where 16 = mel window per video frame.

    Returns:
        (n_frames, 1, N_MELS, MEL_WIN) float32 tensor.
    """
    wav = audio_utils.load_wav(audio_path, sr=_SR)
    mel = audio_utils.melspectrogram(wav)  # (N_MELS, T_mel)

    mel_windows = []
    for i in range(n_frames):
        mel_start = int(i * _SR / _FPS / _HOP)
        mel_end = mel_start + _MEL_WIN
        if mel_start >= mel.shape[1]:
            mel_slice = np.zeros((_N_MELS, _MEL_WIN), dtype=np.float32)
        else:
            mel_slice = mel[:, mel_start:mel_end]
            if mel_slice.shape[1] < _MEL_WIN:
                pad = _MEL_WIN - mel_slice.shape[1]
                mel_slice = np.pad(mel_slice, ((0, 0), (0, pad)), mode="edge")
        mel_windows.append(mel_slice)

    # (n_frames, N_MELS, MEL_WIN) -> (n_frames, 1, N_MELS, MEL_WIN)
    mel_windows = np.stack(mel_windows, axis=0).astype(np.float32)
    return torch.from_numpy(mel_windows).unsqueeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         required=True, help="listener checkpoint .pt")
    parser.add_argument("--pretrained",   required=True, help="EDTalk pretrained .pt")
    parser.add_argument("--speaker",      required=True, help="speaker video or image path")
    parser.add_argument("--listener_ref", required=True, help="listener reference (video or image)")
    parser.add_argument("--listener_audio", default=None,
                        help="Listener audio (.wav). Enables pose/exp conditioning via "
                             "the same mel_proj path used during training (no train/test gap).")
    parser.add_argument("--audio2lip_ckpt", default=None,
                        help="Audio2Lip checkpoint (.pt). When combined with --listener_audio, "
                             "also drives lip sync from listener audio.")
    parser.add_argument("--out",          default="./listener_output.mp4")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--n_frames",     type=int, default=60,
                        help="frames to generate when speaker is an image")
    parser.add_argument("--side_by_side", action="store_true",
                        help="save speaker|listener side-by-side")
    parser.add_argument("--use_prior",    action="store_true",
                        help="sample from VAE prior N(0,I) instead of posterior")
    parser.add_argument("--motion_scale", type=float, default=1.0,
                        help="motion coefficient scale (>1 amplifies movement)")
    parser.add_argument("--smooth",       type=float, default=0.0,
                        help="temporal smoothing (0=none, 0.5~0.8 recommended)")
    parser.add_argument("--z_momentum",   type=float, default=0.9,
                        help="prior z temporal continuity (0=independent, 0.9=smooth)")
    parser.add_argument("--debug",        action="store_true")
    cli = parser.parse_args()

    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- 1. Load checkpoint --
    print(f"Loading checkpoint: {cli.ckpt}")
    ckpt = torch.load(cli.ckpt, map_location=device, weights_only=False)
    args = ckpt["args"]
    args.pretrained_ckpt = cli.pretrained
    args.resume_ckpt = None

    # -- 2. Init Trainer --
    print("Initialising TrainerListener...")
    trainer = TrainerListener(args, device)
    missing, unexpected = trainer.gen.listener_bank.load_state_dict(
        ckpt["listener_bank"], strict=False
    )
    if missing:
        print(f"  [WARN] missing keys (randomly init'd): {len(missing)}")
    if unexpected:
        print(f"  [WARN] unexpected keys: {unexpected}")
    trainer.gen.eval()
    print(f"  listener_bank loaded (step {ckpt.get('start_iter', '?')})")

    size = args.size
    gen = trainer._raw_gen
    lb = gen.listener_bank

    # -- 3. Load Audio2Lip (optional, for lip sync) --
    audio2lip = None
    if cli.audio2lip_ckpt is not None and cli.listener_audio is not None:
        from networks.audio_encoder import Audio2Lip
        audio2lip = Audio2Lip().to(device)
        a2l_ckpt = torch.load(cli.audio2lip_ckpt, map_location=device, weights_only=False)
        audio2lip.load_state_dict(a2l_ckpt.get("audio2lip", a2l_ckpt))
        audio2lip.eval()
        for p in audio2lip.parameters():
            p.requires_grad = False
        print(f"  Audio2Lip loaded from {cli.audio2lip_ckpt}")

    # -- 4. Load speaker input --
    spk_ext = os.path.splitext(cli.speaker)[1].lower()
    if spk_ext in (".jpg", ".jpeg", ".png", ".bmp"):
        spk_static = load_image_as_tensor(cli.speaker, size).to(device)
        cap_spk = None
        n_frames = cli.n_frames
        fps = 25.0
        print(f"Speaker: image -> {n_frames} frames")
    else:
        spk_static = None
        cap_spk = cv2.VideoCapture(cli.speaker)
        n_frames = int(cap_spk.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_spk.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"Speaker video: {n_frames} frames @ {fps:.1f} fps")

    # -- 5. Load listener reference --
    ext = os.path.splitext(cli.listener_ref)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp"):
        listener_src = load_image_as_tensor(cli.listener_ref, size).to(device)
        cap_lis = None
        print(f"Listener ref: image ({cli.listener_ref})")
    else:
        cap_lis = cv2.VideoCapture(cli.listener_ref)
        listener_src = read_frame(cap_lis, 0, size).to(device)
        print(f"Listener ref: video first frame ({cli.listener_ref})")

    # -- 6. Load listener audio mel (optional) --
    mel_seq = None        # (n_frames, N_MELS) for pose/exp conditioning
    a2l_mel_input = None  # (n_frames, 1, N_MELS, MEL_WIN) for Audio2Lip
    if cli.listener_audio is not None:
        print(f"Loading listener audio: {cli.listener_audio}")
        mel_seq = load_mel_sequence(cli.listener_audio, n_frames).to(device)
        print(f"  mel_seq shape: {mel_seq.shape}")
        if audio2lip is not None:
            a2l_mel_input = load_mel_for_audio2lip(cli.listener_audio, n_frames).to(device)
            print(f"  audio2lip mel shape: {a2l_mel_input.shape}")

    # -- 7. Output video writer --
    out_w = size * 2 if cli.side_by_side else size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cli.out, fourcc, fps, (out_w, size))

    # -- 8. Frame-by-frame inference --
    mode_str = "prior" if cli.use_prior else ("active+audio" if mel_seq is not None else "active")
    print(f"Generating listener video ({mode_str})...")
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
        lis_t = listener_src.unsqueeze(0)            # (1, 3, H, W)

        with torch.no_grad():
            if cli.use_prior:
                pred, z_pose, z_exp = trainer.sample_prior(
                    spk_t, lis_t,
                    motion_scale=cli.motion_scale,
                    z_pose=z_pose, z_exp=z_exp,
                    z_momentum=cli.z_momentum,
                )
            else:
                # Compute speaker latent
                latent_poseD_S, wa_S, f_pose_S, f_exp_S = gen._speaker_latent(spk_t)
                wa_L, _, feats_L, _ = gen.enc(lis_t, None, None)
                B = 1

                # Listener mel for this frame (pose/exp conditioning)
                frame_mel = mel_seq[i:i+1] if mel_seq is not None else None  # (1, N_MELS) or None

                if trainer.training_mode == 'passive':
                    f_pose, f_exp = lb.forward_passive(latent_poseD_S, f_pose_S, f_exp_S)
                    alpha_D_lip = torch.zeros(B, gen.lip_dim, device=device)
                else:
                    # Active mode: use same mel_proj path as training
                    f_pose, f_exp, audio_mel_pred, *_ = lb.forward_active(
                        latent_poseD_S, wa_S=wa_L, training=False,
                        listener_mel=frame_mel,
                    )
                    # Lip sync via Audio2Lip if available
                    if audio2lip is not None and a2l_mel_input is not None:
                        mel_window = a2l_mel_input[i:i+1]  # (1, 1, 80, 16)
                        alpha_D_lip = audio2lip(mel_window, 1, 1).squeeze(1)  # (1, 20)
                    else:
                        alpha_D_lip = torch.zeros(B, gen.lip_dim, device=device)

                alpha_D_pose = gen.pose_fc(f_pose) * cli.motion_scale
                alpha_D_exp  = gen.exp_fc(f_exp)   * cli.motion_scale

                alpha_D_L = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
                a_L = gen.direction_exp.get_shared_out(alpha_D_L, gen.direction_lipnonlip.weight)
                e_L = gen.direction_exp.get_exp_latent(a_L)
                directions_D_L = gen.direction_exp(alpha_D_L, gen.direction_lipnonlip.weight)
                latent_poseD_L = wa_L + directions_D_L

                pred = gen.dec(latent_poseD_L, feats_L, e_L)

                if cli.debug and i == 0:
                    print(f"  [DEBUG] mode: active, audio={'YES' if frame_mel is not None else 'NO'}")
                    print(f"  [DEBUG] lip_sync: {'Audio2Lip' if audio2lip is not None else 'zero'}")
                    print(f"  [DEBUG] alpha_D_pose norm: {alpha_D_pose.norm().item():.4f}")
                    print(f"  [DEBUG] alpha_D_exp norm:  {alpha_D_exp.norm().item():.4f}")
                    print(f"  [DEBUG] alpha_D_lip norm:  {alpha_D_lip.norm().item():.4f}")

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
    print(f"Saved -> {cli.out}")


if __name__ == "__main__":
    main()
