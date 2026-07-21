import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EqualLinear


class ReactionVAE(nn.Module):
    """Conditional VAE for stochastic listener motion feature generation.

    Encoder: (speaker_ctx, retrieved_feature) -> (mu, log sigma^2)
    Decoder: (z, speaker_ctx) -> motion feature

    During training  : z = mu + eps * sigma  (reparameterization trick)
    During inference : z ~ N(0, I)  for maximum diversity,
                       or z = mu     for deterministic fallback
    """

    def __init__(self, feature_dim: int = 512, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: cat(speaker_ctx, retrieved) -> hidden
        self.enc_net = nn.Sequential(
            EqualLinear(feature_dim * 2, feature_dim),
            nn.LeakyReLU(0.2),
            EqualLinear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2),
        )
        self.mu_head     = EqualLinear(feature_dim // 2, latent_dim)
        self.logvar_head = EqualLinear(feature_dim // 2, latent_dim)

        # Decoder: cat(z, speaker_ctx) -> feature
        self.dec_net = nn.Sequential(
            EqualLinear(latent_dim + feature_dim, feature_dim),
            nn.LeakyReLU(0.2),
            EqualLinear(feature_dim, feature_dim),
        )

    def encode(self, speaker_ctx: torch.Tensor, retrieved: torch.Tensor):
        h = self.enc_net(torch.cat([speaker_ctx, retrieved], dim=-1))
        logvar = self.logvar_head(h).clamp(-4, 15)
        return self.mu_head(h), logvar

    def decode(self, z: torch.Tensor, speaker_ctx: torch.Tensor) -> torch.Tensor:
        return self.dec_net(torch.cat([z, speaker_ctx], dim=-1))

    def forward(
        self,
        speaker_ctx: torch.Tensor,
        retrieved: torch.Tensor,
        deterministic: bool = False,
    ):
        mu, logvar = self.encode(speaker_ctx, retrieved)
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            z   = mu + std * torch.randn_like(std)
        return self.decode(z, speaker_ctx), mu, logvar

    def sample_prior(self, speaker_ctx: torch.Tensor) -> torch.Tensor:
        """Sample from prior N(0, I) -- maximum diversity at inference."""
        z = torch.randn(speaker_ctx.size(0), self.latent_dim, device=speaker_ctx.device)
        return self.decode(z, speaker_ctx)


class ListenerBank(nn.Module):
    """Listener Feature Bank for empathetic response generation.

    Two forward modes are supported:

    * ``forward_passive`` (Stage 1 -- silent listening):
        Pose and expression subtly mirror the speaker via alpha-scaled addition.

    * ``forward_active`` (Stage 3 -- empathic backchannel):
        Listener mel is projected via mel_proj and used as K/V for pose/exp
        cross-attention. Pose and expression are stochastically sampled
        through independent ReactionVAE modules.
        When listener_mel is not available, falls back to learnable banks.

    Args:
        num_prototypes:  Number of prototype vectors per bank (K).
        feature_dim:     Dimensionality of each prototype vector (default 512).
        vae_latent_dim:  Latent dimension of the ReactionVAE (default 64).
        audio_dim:       Input dimensionality for mel projection (mel bins, default 80).
        pose_dim:        Pose coefficient dimension (default 6).
        exp_dim:         Expression coefficient dimension (default 10).
        mirror_alpha_p:  Fixed scale for passive pose mirroring (default 0.2).
        mirror_alpha_e:  Fixed scale for passive expression mirroring (default 0.2).
    """

    def __init__(
        self,
        num_prototypes: int = 32,
        feature_dim: int = 512,
        vae_latent_dim: int = 64,
        audio_dim: int = 80,
        pose_dim: int = 6,
        exp_dim: int = 10,
        mirror_alpha_p: float = 0.2,
        mirror_alpha_e: float = 0.2,
    ):
        super().__init__()
        self.feature_dim    = feature_dim
        self.mirror_alpha_p = mirror_alpha_p
        self.mirror_alpha_e = mirror_alpha_e

        # --- Learnable prototype banks (K, D) each ---
        # Used as fallback K/V when listener_mel is not available.
        self.lip_bank   = nn.Parameter(torch.randn(num_prototypes, feature_dim) * 0.01)
        self.pose_bank  = nn.Parameter(torch.randn(num_prototypes, feature_dim) * 0.01)
        self.exp_bank   = nn.Parameter(torch.randn(num_prototypes, feature_dim) * 0.01)

        # --- Per-bank cross-attention projections (pose + exp + lip only) ---
        self.lip_q  = EqualLinear(feature_dim, feature_dim)
        self.lip_k  = EqualLinear(feature_dim, feature_dim)
        self.lip_v  = EqualLinear(feature_dim, feature_dim)

        self.pose_q = EqualLinear(feature_dim, feature_dim)
        self.pose_k = EqualLinear(feature_dim, feature_dim)
        self.pose_v = EqualLinear(feature_dim, feature_dim)

        self.exp_q  = EqualLinear(feature_dim, feature_dim)
        self.exp_k  = EqualLinear(feature_dim, feature_dim)
        self.exp_v  = EqualLinear(feature_dim, feature_dim)

        # --- Fusion: cat(latent_poseD_S, directions_D_L) -> motion offset ---
        self.speaker_fusion = nn.Sequential(
            EqualLinear(feature_dim * 2, feature_dim),
            nn.LeakyReLU(0.2),
            EqualLinear(feature_dim, feature_dim),
        )

        # --- Stochastic VAE heads for active reaction (pose + exp only) ---
        self.pose_vae  = ReactionVAE(feature_dim, vae_latent_dim)
        self.exp_vae   = ReactionVAE(feature_dim, vae_latent_dim)

        # --- Mel projection: audio_dim -> feature_dim ---
        # Projects listener mel into the attention space as K/V for pose/exp.
        self.mel_proj = EqualLinear(audio_dim, feature_dim)

    # ------------------------------------------------------------------
    def init_from_prototypes(self, prototype_path: str) -> None:
        """Initialize banks from K-means prototypes extracted from data."""
        data = torch.load(prototype_path, map_location="cpu", weights_only=False)
        prototypes = data["prototypes"]  # (K, D)
        K_file, D = prototypes.shape
        K_self = self.lip_bank.shape[0]

        if K_file != K_self:
            print(f"  [WARN] prototype K={K_file} != bank K={K_self}, "
                  f"using first {min(K_file, K_self)} rows")
            K = min(K_file, K_self)
            prototypes = prototypes[:K]
        else:
            K = K_self

        with torch.no_grad():
            noise_scale = prototypes.std() * 0.05
            self.lip_bank.data[:K]  = prototypes + torch.randn_like(prototypes) * noise_scale
            self.pose_bank.data[:K] = prototypes + torch.randn_like(prototypes) * noise_scale
            self.exp_bank.data[:K]  = prototypes + torch.randn_like(prototypes) * noise_scale

        print(f"  Bank prototypes initialized from {prototype_path} "
              f"(K={K}, D={D}, norm={prototypes.norm(dim=1).mean():.4f})")

    # ------------------------------------------------------------------
    def _attend(self, q_proj, k_proj, v_proj, bank, query):
        """Single cross-attention lookup against a shared prototype bank."""
        Q = q_proj(query)      # (B, D)
        K = k_proj(bank)       # (K, D)
        V = v_proj(bank)       # (K, D)

        scale  = math.sqrt(self.feature_dim)
        attn_w = F.softmax(torch.matmul(Q, K.T) / scale, dim=-1)  # (B, K)
        return torch.matmul(attn_w, V), attn_w                     # (B, D), (B, K)

    # ------------------------------------------------------------------
    def _attend_audio(self, q_proj, k_proj, v_proj, mel_feat, query):
        """Cross-attention where K/V come from projected listener mel feature."""
        Q = q_proj(query)    # (B, D)
        K = k_proj(mel_feat) # (B, D)
        V = v_proj(mel_feat) # (B, D)
        scale = math.sqrt(self.feature_dim)
        gate = torch.sigmoid((Q * K).sum(dim=-1, keepdim=True) / scale)  # (B, 1)
        return gate * V, gate

    # ------------------------------------------------------------------
    def fuse(self, wa_S: torch.Tensor, directions_D_L: torch.Tensor):
        return self.speaker_fusion(torch.cat([wa_S, directions_D_L], dim=-1))

    # ------------------------------------------------------------------
    def forward_passive(
        self,
        latent_poseD_S: torch.Tensor,
        f_pose_S: torch.Tensor,
        f_exp_S: torch.Tensor,
    ):
        """Stage 1 -- passive listening with micro-expression mirroring."""
        f_pose_L, _ = self._attend(self.pose_q, self.pose_k, self.pose_v, self.pose_bank, latent_poseD_S)
        f_exp_L,  _ = self._attend(self.exp_q,  self.exp_k,  self.exp_v,  self.exp_bank,  latent_poseD_S)

        scale_p = f_pose_L.norm(dim=-1, keepdim=True) / (f_pose_S.norm(dim=-1, keepdim=True) + 1e-8)
        scale_e = f_exp_L.norm(dim=-1, keepdim=True)  / (f_exp_S.norm(dim=-1, keepdim=True)  + 1e-8)

        f_pose_final = f_pose_L + self.mirror_alpha_p * scale_p * f_pose_S
        f_exp_final  = f_exp_L  + self.mirror_alpha_e * scale_e * f_exp_S

        return f_pose_final, f_exp_final

    # ------------------------------------------------------------------
    def forward_active(
        self,
        latent_poseD_S: torch.Tensor,
        f_pose_S: torch.Tensor,
        f_exp_S: torch.Tensor,
        wa_S: torch.Tensor = None,
        training: bool = True,
        listener_mel: torch.Tensor = None,
    ):
        """Stage 3 -- active empathic reaction with stochastic pose/expression.

        Listener mel provides real audio conditioning for pose/exp via
        cross-attention. No separate audio VAE/bank needed.

        Args:
            latent_poseD_S: (B, D) speaker pre-decoder latent (cross-attention query).
            wa_S:           (B, D) listener appearance code (VAE context).
            training:       If True, reparameterization sampling; else z = mu.
            listener_mel:   (B, audio_dim) listener mel feature. Used as K/V
                            for pose/exp attention. Falls back to banks when None.

        Returns:
            f_pose, f_exp: (B, D) sampled listener features.
            mu_p, logvar_p, mu_e, logvar_e: VAE parameters.
        """
        vae_ctx = wa_S if wa_S is not None else latent_poseD_S

        if listener_mel is not None:
            mel_feat = self.mel_proj(listener_mel.float())  # (B, D)
            f_pose_L, _ = self._attend_audio(self.pose_q, self.pose_k, self.pose_v, mel_feat, f_pose_S)
            f_exp_L,  _ = self._attend_audio(self.exp_q,  self.exp_k,  self.exp_v,  mel_feat, f_exp_S)
        else:
            f_pose_L, _ = self._attend(self.pose_q, self.pose_k, self.pose_v, self.pose_bank, f_pose_S)
            f_exp_L,  _ = self._attend(self.exp_q,  self.exp_k,  self.exp_v,  self.exp_bank,  f_exp_S)

        f_pose, mu_p, logvar_p = self.pose_vae(
            vae_ctx, f_pose_L, deterministic=not training
        )
        f_exp, mu_e, logvar_e = self.exp_vae(
            vae_ctx, f_exp_L, deterministic=not training
        )

        return f_pose, f_exp, mu_p, logvar_p, mu_e, logvar_e
