"""Trainer for ListenerBank + ReactionVAE fine-tuning on AvaMERG.

Only the ListenerBank parameters (prototype banks + attention projections +
speaker_fusion + pose_vae + exp_vae) are trained.  The rest of the Generator
(encoder, direction networks, decoder) is loaded from a pretrained EDTalk
checkpoint and kept frozen during this stage.

Loss (active mode):
    L = lambda_vgg * VGG_perceptual(pred, gt)
      + lambda_l1  * L1(pred, gt)
      + lambda_adv * GAN_gen(D(pred))
      + kl_weight * KL(N(mu,sigma) || N(0,I))   <- pose_vae + exp_vae + audio_vae
      + lambda_mel * L1(audio_mel, mel_gt)       <- mel reconstruction
      + lambda_bank * bank_sequence_loss         <- bank coefficient supervision

Loss (passive mode):
    L = lambda_vgg * VGG_perceptual(pred, gt)
      + lambda_l1  * L1(pred, gt)
      + lambda_adv * GAN_gen(D(pred))
      (no KL term -- VAE not used in passive mode)

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
    """Return ListenerBank + temporal GRU parameters (all trainable modules)."""
    params = list(gen.listener_bank.parameters())
    params += list(gen.temporal_gru_pose.parameters())
    params += list(gen.temporal_gru_exp.parameters())
    return params


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
        #  Generator -- load pretrained weights, freeze everything except the #
        #  listener_bank (which includes pose_vae and exp_vae).              #
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
            mirror_alpha_p=getattr(args, 'mirror_alpha_p', 0.3),
            mirror_alpha_e=getattr(args, 'mirror_alpha_e', 0.3),
        ).to(device)

        # ------------------------------------------------------------------ #
        #  Discriminator -- created before _load_pretrained so its weights    #
        #  can be restored from the checkpoint and frozen as a fixed critic.  #
        # ------------------------------------------------------------------ #
        self.dis = Discriminator(args.size, args.channel_multiplier).to(device)

        if args.pretrained_ckpt is not None:
            self._load_pretrained(args.pretrained_ckpt)

        # Freeze everything, then unfreeze listener_bank + temporal GRUs
        _requires_grad(self.gen, False)
        _requires_grad(self.gen.listener_bank, True)
        _requires_grad(self.gen.temporal_gru_pose, True)
        _requires_grad(self.gen.temporal_gru_exp, True)

        # Discriminator is used as a fixed critic (loaded from pretrained, frozen)
        _requires_grad(self.dis, False)

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
        # d_optim kept for checkpoint compatibility, but unused (D is frozen)
        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        self.criterion_vgg = VGGLoss().to(device)

        self.lambda_vgg        = getattr(args, "lambda_vgg",        1.0)
        self.lambda_l1         = getattr(args, "lambda_l1",         0.5)  # reduced: less mean-face regression
        self.lambda_adv        = getattr(args, "lambda_adv",        0.3)  # increased: push for realistic motion
        self.lambda_bank       = getattr(args, "lambda_bank",       5.0)
        self.lambda_bank_tgt   = getattr(args, "lambda_bank_tgt",   2.0)  # increased: encourage larger coefficients
        self.lambda_motion_amp = getattr(args, "lambda_motion_amp", 5.0)  # increased: penalise static faces

        # Expression amplification: re-render GT with amplified expressions
        # so reconstruction losses don't suppress expression intensity.
        self.exp_amp_max = getattr(args, "exp_amp_max", 1.5)

        # Bank prototype initialization (K-means from training data)
        bank_init = getattr(args, "bank_init", None)
        if bank_init is not None and os.path.isfile(bank_init):
            self.gen.listener_bank.init_from_prototypes(bank_init)

        self.start_iter = 0

    # ------------------------------------------------------------------ #

    def _load_pretrained(self, ckpt_path: str) -> None:
        """Load Generator + Discriminator weights from an EDTalk checkpoint.

        Keys that belong to listener_bank are skipped (randomly initialised)
        so that only the pretrained backbone is restored.  The Discriminator
        weights are also loaded when present so it can be used as a fixed
        critic during listener training.
        """
        print(f"Loading pretrained from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        gen_state = ckpt.get("gen", ckpt)  # support both wrapped and raw state dicts

        # --- Generator --------------------------------------------------- #
        compatible = {
            k: v for k, v in gen_state.items()
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

        # --- Discriminator (optional, used as fixed critic) -------------- #
        if isinstance(ckpt, dict) and "dis" in ckpt:
            try:
                self.dis.load_state_dict(ckpt["dis"])
                print("  Pretrained Discriminator loaded (will be frozen)")
            except Exception as e:
                print(f"  [WARNING] Failed to load Discriminator: {e}")
                print("  [WARNING] D stays random -- consider --lambda_adv 0.0")
        else:
            print("  [WARNING] No 'dis' key in checkpoint -- D stays random.")
            print("  [WARNING] Random frozen D gives noisy adv_loss. "
                  "Consider --lambda_adv 0.0")

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
        """KL( N(mu, sigma) || N(0, I) ) with free-bits regularisation.

        ``free_bits`` sets a per-dimension floor below which no KL penalty is
        applied.  This prevents posterior collapse.

        Args:
            mu, logvar:  VAE parameters, shape (B, latent_dim) or (B, T, latent_dim).
            free_bits:   Minimum KL per dimension before penalty kicks in.
        """
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return torch.clamp(kl_per_dim, min=free_bits).mean()

    def _bank_sequence_loss(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Match sequence-level distribution of bank coefficients, not per-frame.

        Instead of enforcing frame-by-frame correspondence, we align the
        mean and variance of pred and tgt across the T dimension so that
        the overall reaction pattern matches without being locked to a
        specific frame index.

        Args:
            pred: (B, T, dim) predicted bank coefficients.
            tgt:  (B, T, dim) ground-truth bank coefficients.
        Returns:
            Scalar loss.
        """
        # Mean across T: overall level of reaction across the utterance.
        mean_loss = F.l1_loss(pred.mean(dim=1), tgt.mean(dim=1))

        # Std across T: how much the reaction varies over the utterance.
        std_loss = F.l1_loss(pred.std(dim=1), tgt.std(dim=1))

        return mean_loss + std_loss

    @staticmethod
    def _motion_amplitude_loss(
        pred: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetric loss that penalises insufficient motion amplitude.

        Only activates when the predicted motion variance is *smaller* than
        the ground-truth variance (per coefficient dimension, per batch).
        This avoids conflicts with the pixel-level reconstruction loss while
        explicitly encouraging the model to generate sufficiently dynamic
        reactions.

        Args:
            pred: (B, T, dim) predicted coefficients.
            tgt:  (B, T, dim) ground-truth coefficients.
        Returns:
            Scalar loss.
        """
        pred_std = pred.std(dim=1)  # (B, dim)
        tgt_std  = tgt.std(dim=1)   # (B, dim)
        # Only penalise when pred_std < tgt_std (motion too small)
        deficit = F.relu(tgt_std - pred_std)  # (B, dim), zero where pred >= tgt
        return deficit.mean()

    # ------------------------------------------------------------------ #
    #  Expression-amplified GT                                             #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _amplify_expression_targets(
        self,
        img_tgt_flat: torch.Tensor,
        amp_scale: float,
    ):
        """Re-render GT listener frames with amplified expression coefficients.

        Uses the frozen encoder+decoder to create GT images whose expression
        intensity matches the amplified coefficient targets.  This eliminates
        the conflict between reconstruction losses (which pull toward the
        original small-expression GT) and coefficient losses (which push
        toward larger expressions).

        Args:
            img_tgt_flat: (B*T, C, H, W) ground-truth listener frames.
            amp_scale:    Expression amplification factor (1.0 = identity).

        Returns:
            img_amp:   (B*T, C, H, W) re-rendered GT with amplified expression.
            alpha_pose: (B*T, pose_dim) pose coefficients (unchanged).
            alpha_exp:  (B*T, exp_dim) amplified expression coefficients.
        """
        gen = self._raw_gen

        wa, _, feats, _ = gen.enc(img_tgt_flat, None)
        shared = gen.fc(wa)

        alpha_lip  = gen.lip_fc(shared)
        alpha_pose = gen.pose_fc(shared)
        alpha_exp  = gen.exp_fc(shared) * amp_scale   # amplify expression only

        alpha_D = torch.cat([alpha_lip, alpha_pose, alpha_exp], dim=-1)
        a = gen.direction_exp.get_shared_out(alpha_D, gen.direction_lipnonlip.weight)
        e = gen.direction_exp.get_exp_latent(a)
        directions = gen.direction_exp(alpha_D, gen.direction_lipnonlip.weight)

        img_amp = gen.dec(wa + directions, feats, e)
        return img_amp, alpha_pose, alpha_exp

    # ------------------------------------------------------------------ #
    #  Training steps                                                      #
    # ------------------------------------------------------------------ #

    def gen_update(
            self,
            img_speaker: torch.Tensor,  # (B, T, C, H, W)
            img_listener_src: torch.Tensor,  # (B, C, H, W)
            img_listener_tgt: torch.Tensor,  # (B, T, C, H, W)
            kl_weight: float = 0.0,
            mel_listener_tgt: torch.Tensor = None,  # (B, T, audio_dim)
    ):
        """One generator step over a full utterance sequence.

        Args:
            img_speaker:      (B, T, C, H, W) Speaker video frames.
            img_listener_src: (B, C, H, W)    Listener identity/source frame.
            img_listener_tgt: (B, T, C, H, W) Ground-truth listener target frames.
            kl_weight:        Current KL loss weight (supports warmup schedule).
            mel_listener_tgt: (B, T, audio_dim) Listener mel (used as K/V input).

        Returns:
            vgg_loss, l1_loss, adv_loss, kl_loss, bank_loss, motion_amp_loss, img_recon
        """
        self.gen.train()
        self.gen.zero_grad()
        _requires_grad(self._raw_gen.listener_bank, True)
        _requires_grad(self._raw_gen.temporal_gru_pose, True)
        _requires_grad(self._raw_gen.temporal_gru_exp, True)
        _requires_grad(self._raw_dis, False)

        B, T, C, H, W = img_listener_tgt.shape

        # Forward -- all outputs carry the T dimension.
        img_recon, f_pose, f_exp, alpha_D_pose, alpha_D_exp, _, \
            mu_p, logvar_p, mu_e, logvar_e = \
            self._raw_gen.forward_listener(
                img_speaker, img_listener_src,
                mode=self.training_mode,
                training=True,
                listener_mel=mel_listener_tgt,
            )

        # GT bank coefficients + optional expression amplification.
        # When exp_amp_max > 1.0, GT frames are re-rendered with amplified
        # expressions so that image losses and coefficient losses are aligned.
        with torch.no_grad():
            tgt_flat = img_listener_tgt.view(B * T, C, H, W)

            if self.exp_amp_max > 1.0:
                amp = 1.0 + torch.rand(1).item() * (self.exp_amp_max - 1.0)
                img_tgt_flat, alpha_pose_flat, alpha_exp_flat = \
                    self._amplify_expression_targets(tgt_flat, amp)
                alpha_D_pose_tgt = alpha_pose_flat.view(B, T, -1)
                alpha_D_exp_tgt  = alpha_exp_flat.view(B, T, -1)
            else:
                wa_tgt, _, _, _ = self._raw_gen.enc(tgt_flat, None)
                shared_tgt = self._raw_gen.fc(wa_tgt)
                alpha_D_pose_tgt = self._raw_gen.pose_fc(shared_tgt).view(B, T, -1)
                alpha_D_exp_tgt  = self._raw_gen.exp_fc(shared_tgt).view(B, T, -1)
                img_tgt_flat = tgt_flat

        # Image losses -- flatten T into batch.
        img_recon_flat = img_recon.view(B * T, C, H, W)

        adv_pred = self.dis(img_recon_flat)
        vgg_loss = self.criterion_vgg(img_recon_flat, img_tgt_flat).mean()
        l1_loss = F.l1_loss(img_recon_flat, img_tgt_flat)
        adv_loss = F.softplus(-adv_pred).mean()

        # Bank loss
        bank_loss = (self._bank_sequence_loss(alpha_D_pose, alpha_D_pose_tgt*self.lambda_bank_tgt)
                     + self._bank_sequence_loss(alpha_D_exp, alpha_D_exp_tgt*self.lambda_bank_tgt)
                    ) * self.lambda_bank

        # KL loss -- pose + exp VAEs only (active mode).
        kl_loss = torch.zeros(1, device=self.device)
        if mu_p is not None and kl_weight > 0.0:
            kl_loss = (
                self._kl_loss(mu_p, logvar_p)
                + self._kl_loss(mu_e, logvar_e)
            ) * kl_weight

        # Motion amplitude loss
        motion_amp_loss = torch.zeros(1, device=self.device)
        if self.lambda_motion_amp > 0.0:
            motion_amp_loss = (
                self._motion_amplitude_loss(alpha_D_pose, alpha_D_pose_tgt)
                + self._motion_amplitude_loss(alpha_D_exp, alpha_D_exp_tgt)
            ) * self.lambda_motion_amp

        g_loss = (
                self.lambda_vgg * vgg_loss
                + self.lambda_l1 * l1_loss
                + self.lambda_adv * adv_loss
                + kl_loss
                + bank_loss
                + motion_amp_loss
        )
        g_loss.backward()
        self.g_optim.step()

        return vgg_loss, l1_loss, adv_loss, kl_loss, bank_loss, motion_amp_loss, img_recon.detach()

    def dis_update(
            self,
            img_real: torch.Tensor,  # (B, T, C, H, W)
            img_recon: torch.Tensor,  # (B, T, C, H, W)
    ):
        """Discriminator is frozen (fixed critic) -- no update performed.

        The pretrained Discriminator loaded from the EDTalk checkpoint acts
        as a fixed realism critic whose gradients still flow to the generator
        via adv_loss in gen_update, but whose own weights are never updated.
        """
        return torch.zeros(1, device=self.device)

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
            (img, z_pose_new, z_exp_new)  -- pass z_* back next frame for smoothness.
        """
        self.gen.eval()
        latent_poseD_S, wa_S, f_pose_S, f_exp_S = self._raw_gen._speaker_latent(img_speaker)
        wa_L, _, feats_L, _ = self._raw_gen.enc(img_listener_src, None, None)
        lb   = self._raw_gen.listener_bank
        B, device = wa_L.size(0), wa_L.device
        ldim = lb.pose_vae.latent_dim

        # OU process: z_t = momentum * z_{t-1} + sqrt(1 - momentum^2) * eps
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

        alpha_D_lip  = torch.zeros(B, self._raw_gen.lip_dim, device=device)
        alpha_D_pose = self._raw_gen.pose_fc(f_pose) * motion_scale
        alpha_D_exp  = self._raw_gen.exp_fc(f_exp)   * motion_scale
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
        """Sample listener frames for visualisation during training.

        Handles both single-frame (B, C, H, W) and utterance-level (B, T, C, H, W)
        speaker inputs.  For utterance-level inputs, only the first frame is used
        for a quick visual check (full-sequence generation uses forward_listener).
        """
        self.gen.eval()
        gen = self._raw_gen
        lb  = gen.listener_bank

        # If utterance-level input, take only the first frame for visualisation.
        if img_speaker.dim() == 5:
            img_speaker = img_speaker[:, 0]  # (B, C, H, W)

        latent_poseD_S, wa_S, f_pose_S, f_exp_S = gen._speaker_latent(img_speaker)
        wa_L, _, feats_L, _ = gen.enc(img_listener_src, None, None)
        B, device = latent_poseD_S.size(0), latent_poseD_S.device

        if self.training_mode == 'passive':
            f_pose, f_exp = lb.forward_passive(latent_poseD_S, f_pose_S, f_exp_S)
        else:
            f_pose_L, _ = lb._attend(lb.pose_q, lb.pose_k, lb.pose_v, lb.pose_bank, latent_poseD_S)
            f_exp_L,  _ = lb._attend(lb.exp_q,  lb.exp_k,  lb.exp_v,  lb.exp_bank,  latent_poseD_S)
            f_pose, *_ = lb.pose_vae(wa_L, f_pose_L, deterministic=False)
            f_exp,  *_ = lb.exp_vae(wa_L, f_exp_L,  deterministic=False)

        alpha_D_lip  = torch.zeros(B, gen.lip_dim, device=device)
        alpha_D_pose = gen.pose_fc(f_pose) * motion_scale
        alpha_D_exp  = gen.exp_fc(f_exp)   * motion_scale
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
                "temp_gru_exp": self._raw_gen.temporal_gru_exp.state_dict(),
                "temp_gru_pose": self._raw_gen.temporal_gru_pose.state_dict(),
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

        lb_state = ckpt["listener_bank"]
        missing, unexpected = self._raw_gen.listener_bank.load_state_dict(
            lb_state, strict=False
        )
        audio_missing  = [k for k in missing if "audio" in k]
        other_missing  = [k for k in missing if k not in audio_missing]
        if other_missing:
            print(f"  [WARNING] Unexpected missing keys: {other_missing}")
        if audio_missing:
            print(f"  Audio modules not in ckpt (randomly init'd): {len(audio_missing)} keys")
        if unexpected:
            print(f"  [WARNING] Unexpected keys in ckpt: {unexpected}")

        self._raw_dis.load_state_dict(ckpt["dis"])

        # Load temporal GRU states (critical for empathy temporal modelling)
        if "temp_gru_pose" in ckpt:
            self._raw_gen.temporal_gru_pose.load_state_dict(ckpt["temp_gru_pose"])
            self._raw_gen.temporal_gru_exp.load_state_dict(ckpt["temp_gru_exp"])
            print("  Temporal GRU states loaded")
        else:
            print("  [WARN] No GRU states in checkpoint (randomly init'd)")

        # Skip optimizer state if new parameters were added (incompatible param groups).
        if not audio_missing:
            self.g_optim.load_state_dict(ckpt["g_optim"])
            self.d_optim.load_state_dict(ckpt["d_optim"])
        else:
            print("  Skipping optimizer state (parameter groups changed due to new modules)")

        return ckpt.get("start_iter", 0)
