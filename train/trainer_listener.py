"""Trainer for ListenerBank + ReactionVAE fine-tuning on AvaMERG.

Only the ListenerBank parameters (prototype banks + attention projections +
speaker_fusion + pose_vae + exp_vae) are trained.  The rest of the Generator
(encoder, direction networks, decoder) is loaded from a pretrained EDTalk
checkpoint and kept frozen during this stage.

Loss (active mode):
    L = λ_vgg * VGG_perceptual(pred, gt)
      + λ_l1  * L1(pred, gt)
      + λ_adv * GAN_gen(D(pred))
      + kl_weight * KL(N(μ,σ) || N(0,I))   ← pose_vae + exp_vae + audio_vae
      + λ_mel * L1(audio_mel, mel_gt)       ← mel reconstruction

Loss (passive mode):
    L = λ_vgg * VGG_perceptual(pred, gt)
      + λ_l1  * L1(pred, gt)
      + λ_adv * GAN_gen(D(pred))
      (no KL term — VAE not used in passive mode)

``kl_weight`` is passed per-step from the training loop to support KL warmup.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch import nn, optim

from networks.generator import Generator
from networks.discriminator import Discriminator

from train.vgg19 import VGGLoss


def _requires_grad(net: nn.Module, flag: bool) -> None:
    for p in net.parameters():
        p.requires_grad = flag


def _listener_bank_params(gen: Generator):
    """Return only the ListenerBank parameters (including VAE heads)."""
    return list(gen.listener_bank.parameters())


class TrainerListener(nn.Module):
    """Train only the ListenerBank (+ VAE heads) inside a pretrained Generator.

    Args:
        args:   Parsed argument namespace (see train_listener.py).
        device: torch.device.
    """

    def __init__(self, args, device: torch.device):
        super().__init__()
        self.args          = args
        self.device        = device
        self.training_mode = getattr(args, 'training_mode', 'active')

        # ------------------------------------------------------------------ #
        #  Generator – load pretrained weights, freeze everything except the  #
        #  listener_bank (which includes pose_vae and exp_vae).               #
        # ------------------------------------------------------------------ #
        self.gen = Generator(
            size=args.size,
            style_dim=args.latent_dim_style,
            lip_dim=args.lip_dim,
            pose_dim=args.pose_dim,
            exp_dim=args.exp_dim,
            channel_multiplier=args.channel_multiplier,
            num_listener_prototypes=getattr(args, 'num_listener_prototypes', 16),
            vae_latent_dim=getattr(args, 'vae_latent_dim', 64),
            audio_dim=getattr(args, 'audio_dim', 80),
            audio2lip_ckpt=getattr(args, 'audio2lip_ckpt', None),
            mirror_alpha_p=getattr(args, 'mirror_alpha_p', 0.3),
            mirror_alpha_e=getattr(args, 'mirror_alpha_e', 0.3),
        ).to(device)

        if args.pretrained_ckpt is not None:
            self._load_pretrained(args.pretrained_ckpt)

        # Freeze everything, then unfreeze listener_bank (covers VAE too)
        _requires_grad(self.gen, False)
        _requires_grad(self.gen.listener_bank, True)

        # ------------------------------------------------------------------ #
        #  Discriminator (full, not frozen)                                   #
        # ------------------------------------------------------------------ #
        self.dis = Discriminator(args.size, args.channel_multiplier).to(device)

        # ------------------------------------------------------------------ #
        #  Optimisers                                                          #
        # ------------------------------------------------------------------ #
        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        self.g_optim = optim.Adam(
            _listener_bank_params(self.gen),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        self.criterion_vgg = VGGLoss().to(device)

        self.lambda_vgg = getattr(args, "lambda_vgg", 1.0)
        self.lambda_l1  = getattr(args, "lambda_l1",  1.0)
        self.lambda_adv = getattr(args, "lambda_adv",  0.1)
        self.lambda_mel = getattr(args, "lambda_mel",  1.0)

        self.start_iter = 0

    # ------------------------------------------------------------------ #

    def _load_pretrained(self, ckpt_path: str) -> None:
        """Load Generator weights from an EDTalk checkpoint.

        Keys that belong to listener_bank are skipped (randomly initialised)
        so that only the pretrained backbone is restored.
        """
        print(f"Loading pretrained Generator from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("gen", ckpt)  # support both wrapped and raw state dicts

        # Filter out listener_bank keys – they are newly added
        compatible = {
            k: v for k, v in state.items()
            if not k.startswith("listener_bank")
        }
        missing, unexpected = self.gen.load_state_dict(compatible, strict=False)
        if missing:
            lb_missing = [k for k in missing if "listener_bank" in k]
            other_missing = [k for k in missing if "listener_bank" not in k]
            if other_missing:
                print(f"  [WARNING] Missing non-listener_bank keys: {other_missing}")
            print(f"  listener_bank keys (randomly init'd): {len(lb_missing)}")
        if unexpected:
            print(f"  [WARNING] Unexpected keys: {unexpected}")

    # ------------------------------------------------------------------ #
    #  DDP-safe accessors                                                  #
    # ------------------------------------------------------------------ #

    @property
    def _raw_gen(self):
        return self.gen.module if hasattr(self.gen, "module") else self.gen

    @property
    def _raw_dis(self):
        return self.dis.module if hasattr(self.dis, "module") else self.dis

    # ------------------------------------------------------------------ #
    #  Loss helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.5) -> torch.Tensor:
        """KL( N(μ, σ) || N(0, I) ) with free-bits regularisation.

        ``free_bits`` sets a per-dimension floor below which no KL penalty is
        applied.  This prevents posterior collapse: the VAE is not punished for
        dimensions it has already compressed near the prior, so it keeps using
        the latent code for meaningful variation.

        Args:
            mu, logvar:  VAE parameters, shape (B, latent_dim).
            free_bits:   Minimum KL per dimension before penalty kicks in.
                         0.5 nats ≈ 0.72 bits is a common default.
        """
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # (B, latent_dim)
        return torch.clamp(kl_per_dim, min=free_bits).mean()

    # ------------------------------------------------------------------ #
    #  Training steps                                                      #
    # ------------------------------------------------------------------ #

    def gen_update(
        self,
        img_speaker: torch.Tensor,
        img_listener_src: torch.Tensor,
        img_listener_tgt: torch.Tensor,
        kl_weight: float = 0.0,
        mel_listener_tgt: torch.Tensor = None,
        speaker_audio_mel: torch.Tensor = None,
    ):
        """One generator step.

        Args:
            img_speaker:      (B, 3, H, W)  Speaker frame.
            img_listener_src: (B, 3, H, W)  Listener identity/source frame.
            img_listener_tgt: (B, 3, H, W)  Ground-truth listener target frame.
            kl_weight:        Current KL loss weight (supports warmup schedule).
            mel_listener_tgt: (B, audio_dim) Ground-truth listener mel features.
                              If None, mel reconstruction loss is skipped.

        Returns:
            vgg_loss, l1_loss, adv_loss, kl_loss, mel_loss, img_recon
        """
        self.gen.train()
        self.gen.zero_grad()

        _requires_grad(self._raw_gen.listener_bank, True)
        _requires_grad(self._raw_dis, False)

        img_recon, _, _, _, mu_p, logvar_p, mu_e, logvar_e, audio_mel, mu_a, logvar_a = \
            self._raw_gen.forward_listener(
                img_speaker, img_listener_src,
                mode=self.training_mode,
                speaker_audio_mel=speaker_audio_mel,
                training=True,
            )

        adv_pred = self.dis(img_recon)
        vgg_loss = self.criterion_vgg(img_recon, img_listener_tgt).mean()
        l1_loss  = F.l1_loss(img_recon, img_listener_tgt)
        adv_loss = F.softplus(-adv_pred).mean()

        # KL loss — pose + exp + audio VAEs (active mode only)
        kl_loss = torch.zeros(1, device=self.device)
        if mu_p is not None and kl_weight > 0.0:
            kl_loss = (
                self._kl_loss(mu_p, logvar_p)
                + self._kl_loss(mu_e, logvar_e)
                + self._kl_loss(mu_a, logvar_a)
            ) * kl_weight

        # Mel reconstruction loss
        mel_loss = torch.zeros(1, device=self.device)
        if audio_mel is not None and mel_listener_tgt is not None:
            mel_loss = F.l1_loss(audio_mel, mel_listener_tgt) * self.lambda_mel

        g_loss = (
            self.lambda_vgg * vgg_loss
            + self.lambda_l1  * l1_loss
            + self.lambda_adv * adv_loss
            + kl_loss
            + mel_loss
        )
        g_loss.backward()
        self.g_optim.step()

        return vgg_loss, l1_loss, adv_loss, kl_loss, mel_loss, img_recon.detach()

    def dis_update(
        self,
        img_real: torch.Tensor,
        img_recon: torch.Tensor,
    ):
        """One discriminator step."""
        self.dis.zero_grad()

        _requires_grad(self._raw_gen.listener_bank, False)
        _requires_grad(self._raw_dis, True)

        real_pred = self.dis(img_real)
        fake_pred = self.dis(img_recon.detach())

        d_loss = (
            F.softplus(-real_pred).mean()
            + F.softplus(fake_pred).mean()
        )
        d_loss.backward()
        self.d_optim.step()

        return d_loss

    @torch.no_grad()
    def sample_prior(
        self,
        img_speaker: torch.Tensor,
        img_listener_src: torch.Tensor,
        motion_scale: float = 1.0,
        z_pose: torch.Tensor = None,
        z_exp: torch.Tensor = None,
        z_momentum: float = 0.9,
    ):
        """Sample from VAE prior with optional temporally-smooth z (OU process).

        Args:
            z_pose, z_exp: previous frame's z tensors. Pass None for first frame.
            z_momentum:    0 = fresh random every frame (flickering),
                           0.9 = slow smooth evolution (recommended),
                           1.0 = completely fixed (static).
        Returns:
            (img, z_pose_new, z_exp_new)  — pass z_* back next frame for smoothness.
        """
        self.gen.eval()
        latent_poseD_S, wa_S, f_pose_S, f_exp_S = self._raw_gen._speaker_latent(img_speaker)
        wa_L, _, feats_L, _ = self._raw_gen.enc(img_listener_src, None, None)
        lb   = self._raw_gen.listener_bank
        B, device = wa_L.size(0), wa_L.device
        ldim = lb.pose_vae.latent_dim

        # OU process: z_t = momentum * z_{t-1} + sqrt(1 - momentum²) * ε
        noise_scale = (1.0 - z_momentum ** 2) ** 0.5
        if z_pose is None:
            z_pose = torch.randn(B, ldim, device=device)
        else:
            z_pose = z_momentum * z_pose + noise_scale * torch.randn_like(z_pose)
        if z_exp is None:
            z_exp = torch.randn(B, ldim, device=device)
        else:
            z_exp = z_momentum * z_exp + noise_scale * torch.randn_like(z_exp)

        f_pose = lb.pose_vae.decode(z_pose, wa_L)
        f_exp  = lb.exp_vae.decode(z_exp,  wa_L)
        z_audio = torch.randn(B, lb.audio_vae.latent_dim, device=device)
        f_audio = lb.audio_vae.decode(z_audio, wa_L)

        alpha_D_lip  = torch.zeros(B, self._raw_gen.lip_dim, device=device)
        alpha_D_pose = lb.listener_pose_fc(f_pose) * motion_scale
        alpha_D_exp  = lb.listener_exp_fc(f_exp)   * motion_scale
        alpha_D_L    = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a_L          = self._raw_gen.direction_exp.get_shared_out(alpha_D_L, self._raw_gen.direction_lipnonlip.weight)
        e_L          = self._raw_gen.direction_exp.get_exp_latent(a_L)
        directions_D_L = self._raw_gen.direction_exp(alpha_D_L, self._raw_gen.direction_lipnonlip.weight)
        latent_poseD_L = wa_L + directions_D_L
        return self._raw_gen.dec(latent_poseD_L, feats_L, e_L), z_pose, z_exp

    @torch.no_grad()
    def sample(
        self,
        img_speaker: torch.Tensor,
        img_listener_src: torch.Tensor,
        motion_scale: float = 1.0,
        debug: bool = False,
    ) -> torch.Tensor:
        self.gen.eval()
        gen = self._raw_gen
        lb  = gen.listener_bank

        # Always go through the manual path so we can (a) apply motion_scale
        # uniformly and (b) print debug norms when requested.
        latent_poseD_S, wa_S, f_pose_S, f_exp_S = gen._speaker_latent(img_speaker)
        wa_L, _, feats_L, _ = gen.enc(img_listener_src, None, None)
        B, device = latent_poseD_S.size(0), latent_poseD_S.device

        if self.training_mode == 'passive':
            alpha_D_pose, alpha_D_exp = lb.forward_passive(latent_poseD_S, f_pose_S, f_exp_S)
        else:
            f_pose_L,  _ = lb._attend(lb.pose_q,  lb.pose_k,  lb.pose_v,  lb.pose_bank,  latent_poseD_S)
            f_exp_L,   _ = lb._attend(lb.exp_q,   lb.exp_k,   lb.exp_v,   lb.exp_bank,   latent_poseD_S)
            f_audio_L, _ = lb._attend(lb.audio_q, lb.audio_k, lb.audio_v, lb.audio_bank, latent_poseD_S)
            f_pose,    *_ = lb.pose_vae(wa_L, f_pose_L,  deterministic=False)
            f_exp,     *_ = lb.exp_vae(wa_L,  f_exp_L,   deterministic=False)
            f_audio,   *_ = lb.audio_vae(wa_L, f_audio_L, deterministic=False)
            alpha_D_pose = lb.listener_pose_fc(f_pose) * motion_scale
            alpha_D_exp  = lb.listener_exp_fc(f_exp)   * motion_scale

        alpha_D_lip  = torch.zeros(B, gen.lip_dim, device=device)
        alpha_D_L    = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a_L          = gen.direction_exp.get_shared_out(alpha_D_L, gen.direction_lipnonlip.weight)
        e_L          = gen.direction_exp.get_exp_latent(a_L)
        directions_D_L = gen.direction_exp(alpha_D_L, gen.direction_lipnonlip.weight)
        latent_poseD_L = wa_L + directions_D_L

        if debug:
            print(f"  [DEBUG] training_mode      : {self.training_mode}")
            print(f"  [DEBUG] motion_scale        : {motion_scale}")
            print(f"  [DEBUG] wa_L norm           : {wa_L.norm().item():.4f}")
            print(f"  [DEBUG] alpha_D_pose norm   : {alpha_D_pose.norm().item():.4f}")
            print(f"  [DEBUG] alpha_D_exp  norm   : {alpha_D_exp.norm().item():.4f}")
            print(f"  [DEBUG] directions_D_L norm : {directions_D_L.norm().item():.4f}")
            print(f"  [DEBUG] dir/wa_L ratio      : {(directions_D_L.norm()/wa_L.norm()).item():.4f}")

        return gen.dec(latent_poseD_L, feats_L, e_L)

    # ------------------------------------------------------------------ #
    #  Checkpoint I/O                                                      #
    # ------------------------------------------------------------------ #

    def save(self, idx: int, checkpoint_path: str) -> None:
        torch.save(
            {
                "listener_bank": self._raw_gen.listener_bank.state_dict(),
                "dis":           self._raw_dis.state_dict(),
                "g_optim":       self.g_optim.state_dict(),
                "d_optim":       self.d_optim.state_dict(),
                "args":          self.args,
                "start_iter":    idx,
            },
            os.path.join(checkpoint_path, f"{str(idx).zfill(6)}.pt"),
        )

    def resume(self, ckpt_path: str) -> int:
        print(f"Resuming listener training from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Load listener_bank with strict=False so that newly added audio modules
        # (audio_bank, audio_q/k/v, audio_vae, audio_mlp) remain randomly
        # initialised when resuming from a checkpoint trained without them.
        lb_state = ckpt["listener_bank"]
        missing, unexpected = self._raw_gen.listener_bank.load_state_dict(
            lb_state, strict=False
        )
        audio_missing  = [k for k in missing    if "audio" in k]
        other_missing  = [k for k in missing    if "audio" not in k]
        if other_missing:
            print(f"  [WARNING] Unexpected missing keys: {other_missing}")
        if audio_missing:
            print(f"  Audio modules not in ckpt (randomly init'd): {len(audio_missing)} keys")
        if unexpected:
            print(f"  [WARNING] Unexpected keys in ckpt: {unexpected}")

        self._raw_dis.load_state_dict(ckpt["dis"])

        # If audio modules were missing, the optimizer parameter groups have changed
        # (new params added), so the old optimizer state is incompatible — skip it.
        if not audio_missing:
            self.g_optim.load_state_dict(ckpt["g_optim"])
            self.d_optim.load_state_dict(ckpt["d_optim"])
        else:
            print("  Skipping optimizer state (parameter groups changed due to new audio modules)")

        return ckpt.get("start_iter", 0)
