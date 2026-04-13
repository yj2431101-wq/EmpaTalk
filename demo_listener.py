"""
Listener reaction inference ? T-matched coefficient interpolation.

Eliminates the train-inference gap by running the model at the same
temporal resolution used during training (T_model), then interpolating
motion coefficients to the original video frame count.

Pipeline:
    1. Uniform-sample speaker video ⊥ T_model frames  (same as training)
    2. Interpolate listener mel     ⊥ T_model frames  (same as training)
    3. forward_listener(T_model)    ⊥ pose/exp coefficients + GRU context
    4. Interpolate coefficients     ⊥ n_frames (original video length)
    5. Audio2Lip at n_frames        ⊥ lip coefficients (optional)
    6. Chunked decode(n_frames)     ⊥ output video

Usage:
    python demo_listener.py \
        --ckpt      /path/to/068000.pt \
        --pretrained /path/to/EDTalk.pt \
        --speaker   /path/to/speaker.mp4 \
        --listener_ref /path/to/listener_ref.mp4 \
        --listener_audio /path/to/listener.wav \
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


# 式式 helpers 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式

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
    """(3, H, W) in [-1, 1] -> (H, W, 3) uint8 BGR"""
    img = (t.clamp(-1, 1) + 1) / 2
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (size, size))


def load_image_as_tensor(path, size):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (t - 0.5) / 0.5


def read_frames_uniform(cap, size, total, T):
    """Uniform-sample T frames from the full video (same as training)."""
    if total <= T:
        indices = list(range(total))
    else:
        indices = [int(i * total / T) for i in range(T)]
    frames = []
    for idx in indices:
        f = read_frame(cap, idx, size)
        if f is not None:
            frames.append(f)
    if not frames:
        return torch.zeros(0, 3, size, size)
    out = torch.stack(frames)
    # Pad if shorter than T
    if out.shape[0] < T:
        pad_n = T - out.shape[0]
        out = torch.cat([out, out[-1:].expand(pad_n, -1, -1, -1)], dim=0)
    return out


def load_mel_sequence(audio_path, n_frames):
    """Load audio ⊥ mel, interpolate to n_frames. Returns (n_frames, N_MELS)."""
    wav = audio_utils.load_wav(audio_path, sr=_SR)
    mel = audio_utils.melspectrogram(wav)  # (N_MELS, T_mel)
    mel_t = torch.from_numpy(mel).unsqueeze(0)  # (1, N_MELS, T_mel)
    mel_resampled = F.interpolate(
        mel_t, size=n_frames, mode="linear", align_corners=False
    )  # (1, N_MELS, n_frames)
    return mel_resampled.squeeze(0).permute(1, 0)  # (n_frames, N_MELS)


def load_mel_for_audio2lip(audio_path, n_frames):
    """Load audio ⊥ mel windows for Audio2Lip. Returns (n_frames, 1, 80, 16)."""
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
    mel_windows = np.stack(mel_windows, axis=0).astype(np.float32)
    return torch.from_numpy(mel_windows).unsqueeze(1)


def interp_coefficients(coeff, target_len):
    """Interpolate (1, T, D) coefficients to (1, target_len, D)."""
    # (1, T, D) ⊥ (1, D, T) ⊥ interpolate ⊥ (1, D, target_len) ⊥ (1, target_len, D)
    return F.interpolate(
        coeff.permute(0, 2, 1),
        size=target_len,
        mode='linear',
        align_corners=False,
    ).permute(0, 2, 1)


# 式式 main 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式

def main():
    parser = argparse.ArgumentParser(
        description="Listener inference ? T-matched coefficient interpolation")
    parser.add_argument("--ckpt",         required=True, help="listener checkpoint .pt")
    parser.add_argument("--pretrained",   required=True, help="EDTalk pretrained .pt")
    parser.add_argument("--speaker",      required=True, help="speaker video or image path")
    parser.add_argument("--listener_ref", required=True, help="listener reference (video or image)")
    parser.add_argument("--listener_audio", default=None,
                        help="Listener audio (.wav). Enables pose/exp conditioning "
                             "via mel_proj (same path as training).")
    parser.add_argument("--audio2lip_ckpt", default=None,
                        help="Audio2Lip checkpoint (.pt). Drives lip sync at "
                             "original frame rate (independent of T_model).")
    parser.add_argument("--out",          default="./listener_output.mp4")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--n_frames",     type=int, default=60,
                        help="frames to generate when speaker is an image")
    parser.add_argument("--side_by_side", action="store_true",
                        help="save speaker|listener side-by-side")
    parser.add_argument("--motion_scale", type=float, default=1.0,
                        help="motion coefficient scale (>1 amplifies movement)")
    parser.add_argument("--decode_chunk", type=int, default=32,
                        help="frames per decode batch (reduce if OOM)")
    parser.add_argument("--debug",        action="store_true")
    cli = parser.parse_args()

    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 式式 1. Load checkpoint 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式
    print(f"Loading checkpoint: {cli.ckpt}")
    ckpt = torch.load(cli.ckpt, map_location=device, weights_only=False)
    args = ckpt["args"]
    args.pretrained_ckpt = cli.pretrained
    args.resume_ckpt = None

    T_MODEL = getattr(args, 'max_frames', 16)
    print(f"T_model (from training): {T_MODEL}")

    # 式式 2. Init Trainer 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式
    print("Initialising TrainerListener...")
    trainer = TrainerListener(args, device)

    # Load listener_bank + GRU states
    missing, _ = trainer.gen.listener_bank.load_state_dict(
        ckpt["listener_bank"], strict=False)
    if missing:
        print(f"  [WARN] missing keys in listener_bank: {len(missing)}")

    if "temp_gru_pose" in ckpt:
        trainer.gen.temporal_gru_pose.load_state_dict(ckpt["temp_gru_pose"])
        trainer.gen.temporal_gru_exp.load_state_dict(ckpt["temp_gru_exp"])
        print("  listener_bank + GRU loaded")
    else:
        print("  [WARN] No GRU states in checkpoint")

    print(f"  Step: {ckpt.get('start_iter', '?')}")
    trainer.gen.eval()

    size = args.size
    gen = trainer._raw_gen

    # 式式 3. Load Audio2Lip (optional, for lip sync) 式式式式式式式式式式式式式式式式式式式式式式
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

    # 式式 4. Load speaker input 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式
    spk_ext = os.path.splitext(cli.speaker)[1].lower()
    if spk_ext in (".jpg", ".jpeg", ".png", ".bmp"):
        spk_static = load_image_as_tensor(cli.speaker, size).to(device)
        cap_spk = None
        n_frames = cli.n_frames
        fps = 25.0
        print(f"Speaker: image ⊥ {n_frames} frames")
    else:
        spk_static = None
        cap_spk = cv2.VideoCapture(cli.speaker)
        n_frames = int(cap_spk.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_spk.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"Speaker video: {n_frames} frames @ {fps:.1f} fps")

    # 式式 5. Load listener reference 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式
    ext = os.path.splitext(cli.listener_ref)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp"):
        listener_src = load_image_as_tensor(cli.listener_ref, size).to(device)
        cap_lis = None
    else:
        cap_lis = cv2.VideoCapture(cli.listener_ref)
        listener_src = read_frame(cap_lis, 0, size).to(device)
    print(f"Listener ref: {cli.listener_ref}")

    # 式式 6. Prepare T_model inputs (matching training distribution) 式式式式式式
    print(f"\n[Step 1] Preparing T_model={T_MODEL} inputs...")

    # Speaker: uniform sample ⊥ T_MODEL frames (same as dataset)
    if spk_static is not None:
        spk_T = spk_static.unsqueeze(0).expand(T_MODEL, -1, -1, -1)
    else:
        spk_T = read_frames_uniform(cap_spk, size, n_frames, T_MODEL).to(device)
    spk_batch = spk_T.unsqueeze(0)  # (1, T_MODEL, C, H, W)
    print(f"  Speaker: (1, {T_MODEL}, 3, {size}, {size})")

    # Listener identity
    lis_batch = listener_src.unsqueeze(0)  # (1, C, H, W)

    # Listener mel at T_MODEL (same as dataset's _load_mel_sequence_interp)
    mel_batch = None
    if cli.listener_audio is not None:
        mel_T = load_mel_sequence(cli.listener_audio, T_MODEL).to(device)
        mel_batch = mel_T.unsqueeze(0)  # (1, T_MODEL, 80)
        print(f"  Listener mel: (1, {T_MODEL}, 80)")
    else:
        print(f"  Listener mel: None (no audio provided)")

    # 式式 7. forward_listener at T_MODEL 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式
    print(f"\n[Step 2] Running forward_listener at T_model={T_MODEL}...")
    with torch.no_grad():
        (img_recon_T,  # (1, T_MODEL, C, H, W) ? discarded
         f_pose, f_exp,
         alpha_D_pose,  # (1, T_MODEL, pose_dim=6)
         alpha_D_exp,   # (1, T_MODEL, exp_dim=10)
         latent_poseD_S,
         mu_p, logvar_p, mu_e, logvar_e,
         ) = gen.forward_listener(
            spk_batch, lis_batch,
            mode=trainer.training_mode,
            training=False,
            listener_mel=mel_batch,
        )

    print(f"  alpha_D_pose: {alpha_D_pose.shape}, norm={alpha_D_pose.norm().item():.4f}")
    print(f"  alpha_D_exp:  {alpha_D_exp.shape},  norm={alpha_D_exp.norm().item():.4f}")

    # 式式 8. Interpolate coefficients to original frame count 式式式式式式式式式式式式式
    print(f"\n[Step 3] Interpolating coefficients {T_MODEL} ⊥ {n_frames} frames...")
    alpha_pose_full = interp_coefficients(alpha_D_pose, n_frames)  # (1, n_frames, 6)
    alpha_exp_full  = interp_coefficients(alpha_D_exp,  n_frames)  # (1, n_frames, 10)
    print(f"  alpha_pose_full: {alpha_pose_full.shape}")
    print(f"  alpha_exp_full:  {alpha_exp_full.shape}")

    # 式式 9. Audio2Lip at original frame rate (optional) 式式式式式式式式式式式式式式式式式式
    if audio2lip is not None and cli.listener_audio is not None:
        print(f"\n[Step 4] Audio2Lip at {n_frames} frames (original rate)...")
        a2l_mel = load_mel_for_audio2lip(cli.listener_audio, n_frames).to(device)
        with torch.no_grad():
            alpha_lip_full = audio2lip(a2l_mel, 1, n_frames)  # (1, n_frames, 20)
        print(f"  alpha_lip_full: {alpha_lip_full.shape}")
    else:
        alpha_lip_full = torch.zeros(1, n_frames, gen.lip_dim, device=device)

    # 式式 10. Listener identity encoding (once) 式式式式式式式式式式式式式式式式式式式式式式式式式式式
    with torch.no_grad():
        wa_L, _, feats_L, _ = gen.enc(lis_batch, None, None)
    # wa_L: (1, 512), feats_L: list of (1, C_i, H_i, W_i)

    # 式式 11. Chunked decode 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式
    CHUNK = cli.decode_chunk
    out_w = size * 2 if cli.side_by_side else size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cli.out, fourcc, fps, (out_w, size))

    # Prepare speaker frames for side-by-side (read at original fps)
    spk_frames_for_display = None
    if cli.side_by_side and cap_spk is not None:
        cap_spk.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"\n[Step 5] Decoding {n_frames} frames (chunk={CHUNK})...")
    frame_count = 0

    for start in range(0, n_frames, CHUNK):
        end = min(start + CHUNK, n_frames)
        chunk_size = end - start

        with torch.no_grad():
            # Coefficients for this chunk
            alpha_lip  = alpha_lip_full[0, start:end]       # (chunk, 20)
            alpha_pose = alpha_pose_full[0, start:end] * cli.motion_scale  # (chunk, 6)
            alpha_exp  = alpha_exp_full[0, start:end]  * cli.motion_scale  # (chunk, 10)

            # Direction mapping
            alpha_D_L = torch.cat([alpha_lip, alpha_pose, alpha_exp], dim=-1)  # (chunk, 36)
            a_L = gen.direction_exp.get_shared_out(
                alpha_D_L, gen.direction_lipnonlip.weight)
            e_L = gen.direction_exp.get_exp_latent(a_L)
            directions_D_L = gen.direction_exp(
                alpha_D_L, gen.direction_lipnonlip.weight)

            # Expand listener identity to chunk_size
            wa_L_exp = wa_L.expand(chunk_size, -1)  # (chunk, 512)
            feats_L_exp = [f.expand(chunk_size, *f.shape[1:]) for f in feats_L]

            # Decode
            latent_poseD_L = wa_L_exp + directions_D_L
            pred_chunk = gen.dec(latent_poseD_L, feats_L_exp, e_L)  # (chunk, 3, H, W)

        # Write frames
        for j in range(chunk_size):
            pred_bgr = tensor_to_bgr(pred_chunk[j], size)

            if cli.side_by_side:
                if spk_static is not None:
                    spk_bgr = tensor_to_bgr(spk_static, size)
                elif cap_spk is not None:
                    ok, bgr = cap_spk.read()
                    if ok:
                        spk_bgr = cv2.resize(bgr, (size, size))
                    else:
                        spk_bgr = np.zeros((size, size, 3), dtype=np.uint8)
                else:
                    spk_bgr = np.zeros((size, size, 3), dtype=np.uint8)
                frame = np.concatenate([spk_bgr, pred_bgr], axis=1)
            else:
                frame = pred_bgr

            writer.write(frame)
            frame_count += 1

        print(f"  {frame_count}/{n_frames} frames decoded")

    if cap_spk:
        cap_spk.release()
    if cap_lis:
        cap_lis.release()
    writer.release()

    print(f"\nDone! Saved ⊥ {cli.out}")
    print(f"  T_model={T_MODEL} ⊥ n_frames={n_frames} (interpolation ratio: {n_frames/T_MODEL:.1f}x)")
    if cli.debug:
        print(f"  [DEBUG] alpha_pose range: [{alpha_pose_full.min().item():.4f}, {alpha_pose_full.max().item():.4f}]")
        print(f"  [DEBUG] alpha_exp  range: [{alpha_exp_full.min().item():.4f}, {alpha_exp_full.max().item():.4f}]")
        print(f"  [DEBUG] alpha_lip  range: [{alpha_lip_full.min().item():.4f}, {alpha_lip_full.max().item():.4f}]")


if __name__ == "__main__":
    main()
