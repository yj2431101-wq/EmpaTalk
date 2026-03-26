"""Training script for ListenerBank on AvaMERG.

Only the ListenerBank parameters are trained; the rest of the Generator is
loaded from a pretrained EDTalk checkpoint and frozen.

Example usage (single GPU):
    python train/train_listener.py \\
        --avamerg_root /mnt/data/dataset/AvaMERG \\
        --split_json   /mnt/data/dataset/AvaMERG/train.json \\
        --pretrained_ckpt checkpoints/pretrained.pt \\
        --exp_path ./listener_exp \\
        --exp_name v1 \\
        --batch_size 8

Example usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 train/train_listener.py \\
        --avamerg_root /mnt/data/dataset/AvaMERG \\
        --split_json   /mnt/data/dataset/AvaMERG/train.json \\
        --pretrained_ckpt checkpoints/pretrained.pt \\
        --distributed
"""

import argparse
import os
import sys
import shutil

# Ensure EDTalk root is on sys.path when run as `python train/train_listener.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from tqdm import tqdm

from datasets.dataset_avamerg import AvaMERGDataset
from train.trainer_listener import TrainerListener

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------

def plot_losses(history: dict, save_path: str):
    """6개 loss를 2x3 그리드로 저장."""
    keys = ["vgg", "l1", "adv", "kl", "mel", "d"]
    labels = ["VGG", "L1", "Adv(G)", "KL", "Mel", "Dis"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("ListenerBank Training Loss", fontsize=13)
    for ax, key, label in zip(axes.flat, keys, labels):
        iters, vals = zip(*history[key]) if history[key] else ([], [])
        ax.plot(iters, vals, linewidth=1.2)
        ax.set_title(label)
        ax.set_xlabel("iter")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def write_loss(step, vgg, l1, adv, kl, mel, d, writer: SummaryWriter):
    writer.add_scalar("listener/vgg_loss", vgg.item(), step)
    writer.add_scalar("listener/l1_loss",  l1.item(),  step)
    writer.add_scalar("listener/adv_g",    adv.item(), step)
    writer.add_scalar("listener/kl_loss",  kl.item(),  step)
    writer.add_scalar("listener/mel_loss", mel.item(), step)
    writer.add_scalar("listener/dis_loss", d.item(),   step)
    writer.flush()


def save_samples(step, epoch, img_spk, img_src, img_tgt, img_recon, path):
    grid = F.interpolate(
        torch.cat([img_spk, img_src, img_tgt, img_recon], dim=0), 256
    )
    utils.save_image(
        grid,
        os.path.join(path, f"epoch_{epoch:05d}_step_{step:06d}.jpg"),
        nrow=img_spk.size(0),
        normalize=True,
        value_range=(-1, 1),
    )


# ---------------------------------------------------------------------------

def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    if args.distributed:
        torch.cuda.set_device(local_rank)

    is_main = (local_rank == 0)

    log_path  = os.path.join(args.exp_path, args.exp_name, "log")
    ckpt_path = os.path.join(args.exp_path, args.exp_name, "checkpoint")
    if is_main:
        os.makedirs(log_path,  exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
    if args.distributed:
        dist.barrier()
    writer = SummaryWriter(log_path) if is_main else None

    # ------------------------------------------------------------------ #
    #  Dataset                                                             #
    # ------------------------------------------------------------------ #
    print("==> preparing AvaMERG dataset")
    dataset = AvaMERGDataset(
        root=args.avamerg_root,
        split_json=args.split_json,
        size=args.size,
        identity_gap_min=args.identity_gap_min,
        identity_gap_max=args.identity_gap_max,
        frames_per_video=args.frames_per_video,
    )

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=sampler, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, drop_last=True,
        )

    # ------------------------------------------------------------------ #
    #  Trainer                                                             #
    # ------------------------------------------------------------------ #
    if is_main:
        print("==> initialising TrainerListener")
    trainer = TrainerListener(args, device)

    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        trainer.gen = DDP(trainer.gen, device_ids=[local_rank], find_unused_parameters=True)
        trainer.dis = DDP(trainer.dis, device_ids=[local_rank], find_unused_parameters=True)

    current_iter = args.start_iter
    if args.resume_ckpt is not None:
        current_iter = trainer.resume(args.resume_ckpt)
        print(f"==> resumed from iteration {current_iter}")

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #
    print("==> training")
    last_sample_path = None
    loss_history = {k: [] for k in ["vgg", "l1", "adv", "kl", "mel", "d"]}
    plot_path = os.path.join(ckpt_path, "loss_plot.png")

    epoch_bar = tqdm(range(args.epoch), desc="Epoch", unit="epoch",
                     position=0, dynamic_ncols=True, file=sys.stderr)

    for epoch in epoch_bar:
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)

        batch_bar = tqdm(
            dataloader,
            desc=f"  Train",
            unit="batch",
            position=1,
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
        )

        for batch in batch_bar:
            current_iter += 1

            img_spk  = batch["speaker_frame"].to(device)
            img_src  = batch["listener_source"].to(device)
            img_tgt  = batch["listener_target"].to(device)
            mel_spk  = batch["speaker_mel"].to(device)    # (B, N_MELS) speaker audio query
            mel_tgt  = batch["listener_mel"].to(device)   # (B, N_MELS) listener mel GT

            # KL warmup: linearly ramp from 0 → lambda_kl over kl_warmup_iters
            if args.training_mode == 'active' and args.lambda_kl > 0:
                kl_weight = min(1.0, current_iter / max(1, args.kl_warmup_iters)) * args.lambda_kl
            else:
                kl_weight = 0.0

            # Generator step
            vgg_loss, l1_loss, adv_loss, kl_loss, mel_loss, img_recon = trainer.gen_update(
                img_spk, img_src, img_tgt,
                kl_weight=kl_weight,
                mel_listener_tgt=mel_tgt,
                speaker_audio_mel=mel_spk,
            )

            # Discriminator step
            d_loss = trainer.dis_update(img_tgt, img_recon)

            # ---- 실시간 loss 출력 + 그래프 업데이트 ----
            if is_main:
                for k, v in zip(
                    ["vgg", "l1", "adv", "kl", "mel", "d"],
                    [vgg_loss, l1_loss, adv_loss, kl_loss, mel_loss, d_loss],
                ):
                    loss_history[k].append((current_iter, v.item()))

            if is_main and current_iter % args.print_freq == 0:
                print(
                    f"[E{epoch:03d}/{args.epoch}][I{current_iter:06d}] "
                    f"vgg={vgg_loss.item():.4f} "
                    f"l1={l1_loss.item():.4f} "
                    f"adv={adv_loss.item():.4f} "
                    f"kl={kl_loss.item():.4f} "
                    f"mel={mel_loss.item():.4f} "
                    f"d={d_loss.item():.4f}",
                    flush=True,
                )
                plot_losses(loss_history, plot_path)

            # ---- sample images (rank 0 only) ----
            if is_main and current_iter % args.image_save_iter == 0:
                with torch.no_grad():
                    img_recon_vis = trainer.sample(img_spk, img_src)
                save_samples(
                    current_iter, epoch,
                    img_spk[:4], img_src[:4], img_tgt[:4], img_recon_vis[:4],
                    ckpt_path,
                )
                last_sample_path = os.path.join(
                    ckpt_path,
                    f"epoch_{epoch:05d}_step_{current_iter:06d}.jpg"
                )

            # ---- checkpoint (rank 0 only) ----
            if is_main and current_iter % args.save_freq == 0:
                trainer.save(current_iter, ckpt_path)
                if last_sample_path and os.path.isfile(last_sample_path):
                    shutil.copy(
                        last_sample_path,
                        os.path.join(ckpt_path, f"step_{current_iter:06d}.jpg"),
                    )

        epoch_bar.set_postfix(iter=current_iter)

    # Final save (rank 0 only)
    if is_main:
        trainer.save(current_iter, ckpt_path)
        writer.close()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ListenerBank on AvaMERG")

    # Dataset
    parser.add_argument("--avamerg_root", type=str,
                        default="/mnt/data/dataset/AvaMERG",
                        help="Root directory of the AvaMERG dataset")
    parser.add_argument("--split_json", type=str,
                        default=None,
                        help="Path to train.json (default: <avamerg_root>/train.json)")
    parser.add_argument("--identity_gap_min", type=int, default=10,
                        help="Min frame gap for listener identity frame")
    parser.add_argument("--identity_gap_max", type=int, default=30,
                        help="Max frame gap for listener identity frame")
    parser.add_argument("--frames_per_video", type=int, default=8,
                        help="Frame pairs sampled per video pair per epoch")
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--lip_dim",  type=int, default=20)
    parser.add_argument("--pose_dim", type=int, default=6)
    parser.add_argument("--exp_dim",  type=int, default=10)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--num_listener_prototypes", type=int, default=32)
    parser.add_argument("--vae_latent_dim",  type=int,   default=64,
                        help="Latent dimension of ReactionVAE (pose + exp heads)")
    parser.add_argument("--audio_dim",       type=int,   default=80,
                        help="Output dimension of audio MLP (mel bins)")
    parser.add_argument("--mirror_alpha_p",  type=float, default=0.2,
                        help="Passive pose mirroring scale α_p ∈ [0, 1]")
    parser.add_argument("--mirror_alpha_e",  type=float, default=0.2,
                        help="Passive expression mirroring scale α_e ∈ [0, 1]")

    # Pretrained Generator checkpoint (required for frozen backbone)
    parser.add_argument("--pretrained_ckpt", type=str, default=None,
                        help="Path to pretrained EDTalk .pt checkpoint")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Path to listener training checkpoint to resume")
    parser.add_argument("--audio2lip_ckpt", type=str, default=None,
                        help="Path to Audio2Lip .pt checkpoint (ckpts/Audio2Lip.pt). "
                             "If provided, predicted mel drives lip motion via Audio2Lip.")

    # Optimiser
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--lambda_vgg", type=float, default=1.0)
    parser.add_argument("--lambda_l1",  type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.1)
    parser.add_argument("--lambda_kl",  type=float, default=0.01,
                        help="KL loss weight for ReactionVAE (active mode only)")
    parser.add_argument("--lambda_mel", type=float, default=1.0,
                        help="Mel reconstruction loss weight for audio_vae output")
    parser.add_argument("--kl_warmup_iters", type=int, default=5000,
                        help="Iterations to linearly ramp KL weight from 0 to lambda_kl")
    parser.add_argument("--training_mode", type=str, default="active",
                        choices=["passive", "active"],
                        help="'active': VAE-sampled reactions; 'passive': mirroring only")

    # Training schedule
    parser.add_argument("--epoch",      type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--start_iter", type=int, default=0)

    # Logging / saving
    parser.add_argument("--exp_path",       type=str, default="./listener_exp")
    parser.add_argument("--exp_name",       type=str, default="v1")
    parser.add_argument("--log_iter",       type=int, default=10)
    parser.add_argument("--print_freq",     type=int, default=10,
                        help="매 N iter마다 loss 출력 (stdout, flush=True)")
    parser.add_argument("--image_save_iter",type=int, default=500)
    parser.add_argument("--save_freq",      type=int, default=2000)

    # DDP
    parser.add_argument("--distributed",  action="store_true")
    parser.add_argument("--local_rank",   type=int, default=0)
    parser.add_argument("--addr",         type=str, default="localhost")
    parser.add_argument("--port",         type=str, default="12345")

    args = parser.parse_args()

    if args.split_json is None:
        args.split_json = os.path.join(args.avamerg_root, "train.json")

    n_gpu = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        os.environ.setdefault("MASTER_ADDR", args.addr)
        os.environ.setdefault("MASTER_PORT", args.port)
        dist.init_process_group(backend="nccl", init_method="env://")

    main(args)
