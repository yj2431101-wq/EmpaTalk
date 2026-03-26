import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EqualLinear


class ReactionVAE(nn.Module):
    """Conditional VAE for stochastic listener motion feature generation.

    Encoder: (speaker_ctx, retrieved_feature) → (μ, log σ²)
    Decoder: (z, speaker_ctx) → motion feature

    During training  : z = μ + ε·σ  (reparameterization trick)
    During inference : z ~ N(0, I)  for maximum diversity,
                       or z = μ     for deterministic fallback
    """

    def __init__(self, feature_dim: int = 512, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: cat(speaker_ctx, retrieved) → hidden
        self.enc_net = nn.Sequential(
            EqualLinear(feature_dim * 2, feature_dim),
            nn.LeakyReLU(0.2),
            EqualLinear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2),
        )
        self.mu_head     = EqualLinear(feature_dim // 2, latent_dim)
        self.logvar_head = EqualLinear(feature_dim // 2, latent_dim)

        # Decoder: cat(z, speaker_ctx) → feature
        self.dec_net = nn.Sequential(
            EqualLinear(latent_dim + feature_dim, feature_dim),
            nn.LeakyReLU(0.2),
            EqualLinear(feature_dim, feature_dim),
        )

    def encode(self, speaker_ctx: torch.Tensor, retrieved: torch.Tensor):
        h = self.enc_net(torch.cat([speaker_ctx, retrieved], dim=-1))
        # Clamp logvar to prevent exp() explosion during early training
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
        """
        Args:
            speaker_ctx:  (B, feature_dim) speaker pre-decoder latent.
            retrieved:    (B, feature_dim) bank-retrieved listener feature.
            deterministic: if True, z = μ (no sampling).

        Returns:
            out:    (B, feature_dim) generated motion feature.
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
        """
        mu, logvar = self.encode(speaker_ctx, retrieved)
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            z   = mu + std * torch.randn_like(std)
        return self.decode(z, speaker_ctx), mu, logvar

    def sample_prior(self, speaker_ctx: torch.Tensor) -> torch.Tensor:
        """Sample from prior N(0, I) — maximum diversity at inference."""
        z = torch.randn(speaker_ctx.size(0), self.latent_dim, device=speaker_ctx.device)
        return self.decode(z, speaker_ctx)


class ListenerBank(nn.Module):
    """Listener Feature Bank for empathetic response generation.

    Mirrors the speaker's 3-encoder design with three independent prototype
    banks — lip, pose, and expression.  The speaker's **pre-decoder latent**
    (``latent_poseD_S = wa_S + directions_D_S``) is used as the query so that
    the bank responds to the speaker's full motion context, not just raw
    appearance.

    After retrieval, a fusion module combines the speaker's pre-decoder latent
    with the listener's motion directions to produce the final motion offset
    that is added to the listener's appearance code::

        latent_poseD_L = wa_L + fuse(latent_poseD_S, directions_D_L)

    Two forward modes are supported:

    * ``forward_passive`` (Stage 1 – silent listening):
        Lip motion is zeroed out (mouth closed).  Pose and expression subtly
        mirror the speaker via α-scaled addition::

            f_pose = f_l_pose + α_p · f_s_pose
            f_exp  = f_l_exp  + α_e · f_s_exp

    * ``forward_active`` (Stage 3 – empathic backchannel):
        Lip motion comes from the TTS audio (passed externally as
        ``alpha_D_lip``).  Pose and expression are **stochastically sampled**
        through independent :class:`ReactionVAE` modules, giving diverse
        reactions conditioned on the same speaker context.
        Audio features are also stochastically generated via ``audio_vae``
        and projected to mel-bin space through ``audio_mlp``.

    Args:
        num_prototypes:  Number of prototype vectors per bank (K).
        feature_dim:     Dimensionality of each prototype vector (default 512).
        vae_latent_dim:  Latent dimension of the ReactionVAE (default 64).
        audio_dim:       Output dimensionality of the audio MLP (mel bins, default 80).
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

        # --- Learnable prototype banks  (K, D) each ---
        self.lip_bank   = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.pose_bank  = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.exp_bank   = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.audio_bank = nn.Parameter(torch.randn(num_prototypes, feature_dim))

        # --- Per-bank cross-attention projections ---
        self.lip_q  = EqualLinear(feature_dim, feature_dim)
        self.lip_k  = EqualLinear(feature_dim, feature_dim)
        self.lip_v  = EqualLinear(feature_dim, feature_dim)

        self.pose_q = EqualLinear(feature_dim, feature_dim)
        self.pose_k = EqualLinear(feature_dim, feature_dim)
        self.pose_v = EqualLinear(feature_dim, feature_dim)

        self.exp_q  = EqualLinear(feature_dim, feature_dim)
        self.exp_k  = EqualLinear(feature_dim, feature_dim)
        self.exp_v  = EqualLinear(feature_dim, feature_dim)

        self.audio_q = EqualLinear(feature_dim, feature_dim)
        self.audio_k = EqualLinear(feature_dim, feature_dim)
        self.audio_v = EqualLinear(feature_dim, feature_dim)

        # --- Fusion: cat(latent_poseD_S, directions_D_L) → motion offset ---
        self.speaker_fusion = nn.Sequential(
            EqualLinear(feature_dim * 2, feature_dim),
            nn.LeakyReLU(0.2),
            EqualLinear(feature_dim, feature_dim),
        )

        # --- Stochastic VAE heads for active reaction (pose + exp + audio) ---
        # Lip features in active mode come from TTS audio, not the VAE.
        self.pose_vae  = ReactionVAE(feature_dim, vae_latent_dim)
        self.exp_vae   = ReactionVAE(feature_dim, vae_latent_dim)
        self.audio_vae = ReactionVAE(feature_dim, vae_latent_dim)

        # --- Audio MLP: feature_dim → audio_dim (mel reconstruction head) ---
        # Projects VAE output into mel-bin space for audio reconstruction.
        self.audio_mlp = nn.Sequential(
            EqualLinear(feature_dim, feature_dim),
            nn.LeakyReLU(0.2),
            EqualLinear(feature_dim, audio_dim),
        )

        # --- Speaker audio query projection: audio_dim → feature_dim ---
        self.speaker_audio_proj = nn.Sequential(
            EqualLinear(audio_dim, feature_dim),
            nn.LeakyReLU(0.2),
            EqualLinear(feature_dim, feature_dim),
        )

        # --- Listener-specific motion FC heads ---
        # Replaces speaker's pose_fc/exp_fc so the bank output distribution
        # doesn't need to match the speaker's W-space.
        self.listener_pose_fc = nn.Sequential(
            EqualLinear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2),
            EqualLinear(feature_dim // 2, pose_dim),
        )
        self.listener_exp_fc = nn.Sequential(
            EqualLinear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2),
            EqualLinear(feature_dim // 2, exp_dim),
        )

    # ------------------------------------------------------------------
    def _attend(self, q_proj, k_proj, v_proj, bank, query):
        """Single cross-attention lookup.

        Args:
            q_proj, k_proj, v_proj: EqualLinear projection layers.
            bank:  (K, D) prototype matrix.
            query: (B, D) query vector.

        Returns:
            retrieved: (B, D) soft-weighted sum of bank values.
            attn_w:    (B, K) attention weight distribution.
        """
        Q = q_proj(query)      # (B, D)
        K = k_proj(bank)       # (K, D)
        V = v_proj(bank)       # (K, D)

        scale  = math.sqrt(self.feature_dim)
        attn_w = F.softmax(torch.matmul(Q, K.T) / scale, dim=-1)  # (B, K)
        return torch.matmul(attn_w, V), attn_w                     # (B, D), (B, K)

    # ------------------------------------------------------------------
    def forward(self, latent_poseD_S: torch.Tensor):
        """Retrieve four listener features using the speaker's pre-decoder latent.

        Args:
            latent_poseD_S: Speaker's pre-decoder latent ``wa_S + directions_D_S``,
                            shape (B, feature_dim).

        Returns:
            f_lip, f_pose, f_exp, f_audio:                  Retrieved features, each (B, D).
            attn_lip, attn_pose, attn_exp, attn_audio:      Attention weights,   each (B, K).
        """
        f_lip,   attn_lip   = self._attend(self.lip_q,   self.lip_k,   self.lip_v,   self.lip_bank,   latent_poseD_S)
        f_pose,  attn_pose  = self._attend(self.pose_q,  self.pose_k,  self.pose_v,  self.pose_bank,  latent_poseD_S)
        f_exp,   attn_exp   = self._attend(self.exp_q,   self.exp_k,   self.exp_v,   self.exp_bank,   latent_poseD_S)
        f_audio, attn_audio = self._attend(self.audio_q, self.audio_k, self.audio_v, self.audio_bank, latent_poseD_S)

        return f_lip, f_pose, f_exp, f_audio, attn_lip, attn_pose, attn_exp, attn_audio

    # ------------------------------------------------------------------
    def fuse(self, wa_S: torch.Tensor, directions_D_L: torch.Tensor):
        """Fuse speaker appearance code with listener motion directions.

        Args:
            wa_S:           (B, D) speaker appearance code (motion-free, wa_S only).
                            Passing wa_S instead of latent_poseD_S prevents fuse()
                            from copying the speaker's head/expression motion onto
                            the listener.
            directions_D_L: (B, D) listener motion directions from bank.

        Returns:
            fused: (B, D) motion offset to add to ``wa_L``.
        """
        return self.speaker_fusion(torch.cat([wa_S, directions_D_L], dim=-1))

    # ------------------------------------------------------------------
    def forward_passive(
        self,
        latent_poseD_S: torch.Tensor,
        f_pose_S: torch.Tensor,
        f_exp_S: torch.Tensor,
    ):
        """Stage 1 — passive listening with micro-expression mirroring.

        The listener's mouth stays closed (lip feature = 0).
        Pose and expression subtly echo the speaker via fixed α scales::

            f_pose = f_l_pose + α_p · f_s_pose
            f_exp  = f_l_exp  + α_e · f_s_exp

        Args:
            latent_poseD_S: (B, D) speaker pre-decoder latent (bank query).
            f_pose_S:       (B, D) speaker pose direction vector.
            f_exp_S:        (B, D) speaker expression direction vector.

        Returns:
            f_pose_final: (B, D) listener pose feature.
            f_exp_final:  (B, D) listener expression feature.
        """
        f_pose_L, _ = self._attend(self.pose_q, self.pose_k, self.pose_v, self.pose_bank, latent_poseD_S)
        f_exp_L,  _ = self._attend(self.exp_q,  self.exp_k,  self.exp_v,  self.exp_bank,  latent_poseD_S)

        # Normalise f_s to the same magnitude as f_l so α is a true mixing ratio.
        # Without this, the QR-basis scale of f_s may differ wildly from the
        # prototype-bank scale of f_l, making α=0.2 unpredictable in practice.
        scale_p = f_pose_L.norm(dim=-1, keepdim=True) / (f_pose_S.norm(dim=-1, keepdim=True) + 1e-8)
        scale_e = f_exp_L.norm(dim=-1, keepdim=True)  / (f_exp_S.norm(dim=-1, keepdim=True)  + 1e-8)

        f_pose_final = f_pose_L + self.mirror_alpha_p * scale_p * f_pose_S
        f_exp_final  = f_exp_L  + self.mirror_alpha_e * scale_e * f_exp_S

        alpha_D_pose = self.listener_pose_fc(f_pose_final)
        alpha_D_exp  = self.listener_exp_fc(f_exp_final)
        return alpha_D_pose, alpha_D_exp

    # ------------------------------------------------------------------
    def forward_active(
        self,
        latent_poseD_S: torch.Tensor,
        wa_S: torch.Tensor = None,
        training: bool = True,
        speaker_audio_feat: torch.Tensor = None,
    ):
        """Stage 3 — active empathic reaction with stochastic pose/expression.

        Args:
            latent_poseD_S:     (B, D) speaker pre-decoder latent (visual).
            wa_S:               (B, D) speaker appearance code (VAE context).
            training:           If True, reparameterization sampling.
            speaker_audio_feat: (B, audio_dim) speaker mel.
                                Combined with visual query for pose/exp banks
                                so speech content modulates which listener
                                reactions are retrieved.

        Returns:
            alpha_D_pose, alpha_D_exp, audio_mel,
            mu_p, logvar_p, mu_e, logvar_e, mu_a, logvar_a
        """
        vae_ctx = wa_S if wa_S is not None else latent_poseD_S

        # Combined query: visual + audio for pose/exp banks
        # Audio shifts the retrieval toward empathetically relevant reactions.
        if speaker_audio_feat is not None:
            audio_proj = self.speaker_audio_proj(speaker_audio_feat)  # (B, D)
            pose_exp_query = latent_poseD_S + audio_proj
            audio_query    = audio_proj
        else:
            pose_exp_query = latent_poseD_S
            audio_query    = latent_poseD_S

        f_pose_L,  _ = self._attend(self.pose_q,  self.pose_k,  self.pose_v,  self.pose_bank,  pose_exp_query)
        f_exp_L,   _ = self._attend(self.exp_q,   self.exp_k,   self.exp_v,   self.exp_bank,   pose_exp_query)
        f_audio_L, _ = self._attend(self.audio_q, self.audio_k, self.audio_v, self.audio_bank, audio_query)

        f_pose,      mu_p, logvar_p = self.pose_vae(vae_ctx,  f_pose_L,  deterministic=not training)
        f_exp,       mu_e, logvar_e = self.exp_vae(vae_ctx,   f_exp_L,   deterministic=not training)
        f_audio_feat, mu_a, logvar_a = self.audio_vae(vae_ctx, f_audio_L, deterministic=not training)

        audio_mel    = self.audio_mlp(f_audio_feat)         # (B, audio_dim)
        alpha_D_pose = self.listener_pose_fc(f_pose)        # (B, pose_dim)
        alpha_D_exp  = self.listener_exp_fc(f_exp)          # (B, exp_dim)

        return alpha_D_pose, alpha_D_exp, audio_mel, mu_p, logvar_p, mu_e, logvar_e, mu_a, logvar_a
