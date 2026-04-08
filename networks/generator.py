from torch import nn
from .encoder import *
from .styledecoder import Synthesis
from .listener_bank import ListenerBank
import torch

class Direction(nn.Module):
    def __init__(self, lip_dim, pose_dim):
        super(Direction, self).__init__()
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.weight = nn.Parameter(torch.randn(512, lip_dim+pose_dim))

    def forward(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.linalg.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out
    def get_shared_out(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.linalg.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)  # torch.Size([1, 20, 512])
            return out
    def get_lip_latent(self, out):
        lip_latent = torch.sum(out[:,:self.lip_dim], dim=1)
        return lip_latent
    def get_pose_latent(self, out):
        pose_latent = torch.sum(out[:,self.lip_dim:], dim=1)
        return pose_latent

class Direction_exp(nn.Module):
    def __init__(self, lip_dim, pose_dim, exp_dim):
        super(Direction_exp, self).__init__()
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.weight = nn.Parameter(torch.randn(512, exp_dim))

    def forward(self, input, lipnonlip_weight):
        # input: (bs*t) x 512
        weight = torch.cat([lipnonlip_weight, self.weight], -1)
        weight = weight + 1e-8 # torch.Size([512, 36])
        Q, R = torch.linalg.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix torch.Size([1, 36]) torch.Size([1, 36, 36])
            out = torch.matmul(input_diag, Q.T) # Q torch.Size([512, 36]) OUT torch.Size([1, 36, 512])
            out = torch.sum(out, dim=1)

            return out

    def only_exp(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.linalg.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix torch.Size([1, 40, 40])
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out

    def get_shared_out(self, input, lipnonlip_weight):
        # input: (bs*t) x 512
        weight = torch.cat([lipnonlip_weight, self.weight], -1)
        weight = weight + 1e-8
        Q, R = torch.linalg.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)  # torch.Size([1, 20, 512])
            return out
    def get_lip_latent(self, out):
        lip_latent = torch.sum(out[:,:self.lip_dim], dim=1)
        return lip_latent
    def get_pose_latent(self, out):
        pose_latent = torch.sum(out[:,self.lip_dim:self.lip_dim+self.pose_dim], dim=1)
        return pose_latent

    def get_exp_latent(self, out):
        exp_latent = torch.sum(out[:,self.lip_dim+self.pose_dim:], dim=1)
        return exp_latent


class Generator(nn.Module):
    def __init__(self, size, style_dim=512, lip_dim=20, pose_dim=6, exp_dim=10,
                 channel_multiplier=1, blur_kernel=[1, 3, 3, 1],
                 num_listener_prototypes=32, vae_latent_dim=64,
                 audio_dim=80,
                 mirror_alpha_p=0.2, mirror_alpha_e=0.2):
        super(Generator, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
        self.direction_lipnonlip = Direction(lip_dim, pose_dim)
        self.direction_exp = Direction_exp(lip_dim, pose_dim, exp_dim)
        # motion network
        fc = [EqualLinear(style_dim, style_dim)]
        for i in range(3):
            fc.append(EqualLinear(style_dim, style_dim))
        self.fc = nn.Sequential(*fc)

        lip_fc = [EqualLinear(style_dim, style_dim)]
        lip_fc.append(EqualLinear(style_dim, style_dim))
        lip_fc.append(EqualLinear(style_dim, lip_dim))
        self.lip_fc = nn.Sequential(*lip_fc)

        pose_fc = [EqualLinear(style_dim, style_dim)]
        pose_fc.append(EqualLinear(style_dim, style_dim))
        pose_fc.append(EqualLinear(style_dim, pose_dim))
        self.pose_fc = nn.Sequential(*pose_fc)

        exp_fc = [EqualLinear(style_dim, style_dim)]
        exp_fc.append(EqualLinear(style_dim, style_dim))
        exp_fc.append(EqualLinear(style_dim, exp_dim))
        self.exp_fc = nn.Sequential(*exp_fc)

        # listener bank: K prototype listener embeddings queried by speaker feature
        self.listener_bank = ListenerBank(
            num_listener_prototypes, style_dim,
            vae_latent_dim=vae_latent_dim,
            audio_dim=audio_dim,
            pose_dim=pose_dim,
            exp_dim=exp_dim,
            mirror_alpha_p=mirror_alpha_p,
            mirror_alpha_e=mirror_alpha_e,
        )

        # Temporal GRU -- applied after bank retrieval, before coefficient decoding.
        # hidden_size matches style_dim so pose_fc / exp_fc input dims stay unchanged.
        self.temporal_gru_pose = nn.GRU(
            input_size=style_dim,
            hidden_size=style_dim,
            batch_first=True,
        )
        self.temporal_gru_exp = nn.GRU(
            input_size=style_dim,
            hidden_size=style_dim,
            batch_first=True,
        )


    def test_EDTalk_V(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start)
        wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive)
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight)
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon

    def test_EDTalk_V_use_exp_weight(self, img_source, lip_img_drive, pose_img_drive, alpha_D_exp, h_start=None):

        wa, wa_t, feats, _ = self.enc(img_source, lip_img_drive, h_start)
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None)
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight)
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon

    def test_EDTalk_A(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, feats_t = self.enc(img_source, exp_img_drive, h_start)
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight)
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon


    def test_EDTalk_A_use_exp_weight(self, img_source, lip_img_drive, pose_img_drive, alpha_D_exp, h_start=None):

        wa, wa_t_p, feats, _ = self.enc(img_source, pose_img_drive, h_start)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight)
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon

    def _speaker_latent(self, img_speaker):
        """Compute the speaker's pre-decoder latent and per-component direction vectors.

        Args:
            img_speaker: (B, 3, H, W) speaker's current frame.

        Returns:
            latent_poseD_S: (B, 512) speaker's pre-decoder latent (wa_S + directions_D_S).
            wa_S:           (B, 512) speaker's raw appearance code.
            f_pose_S:       (B, 512) speaker's pose direction vector (for passive mirroring).
            f_exp_S:        (B, 512) speaker's expression direction vector (for passive mirroring).
        """
        wa_S, _, _, _ = self.enc(img_speaker, None)          # (B, 512)
        shared = self.fc(wa_S)
        alpha_D_S = torch.cat([
            self.lip_fc(shared),   # (B, lip_dim)
            self.pose_fc(shared),  # (B, pose_dim)
            self.exp_fc(shared),   # (B, exp_dim)
        ], dim=-1)
        # get_shared_out returns (B, total_dim, 512); sum gives the full direction offset
        shared_out = self.direction_exp.get_shared_out(alpha_D_S, self.direction_lipnonlip.weight)
        f_pose_S = self.direction_exp.get_pose_latent(shared_out)  # (B, 512)
        f_exp_S  = self.direction_exp.get_exp_latent(shared_out)   # (B, 512)
        directions_D_S = torch.sum(shared_out, dim=1)              # (B, 512)
        return wa_S + directions_D_S, wa_S, f_pose_S, f_exp_S

    def forward_listener_frame(
        self,
        img_speaker,
        img_listener,
        h_start=None,
        mode: str = 'active',
        audio_lip_feat=None,
        training: bool = True,
        listener_mel=None,
    ):
        """Generate an empathetic listener response conditioned on a single speaker frame.

        Args:
            img_speaker:     (B, 3, H, W) speaker's current frame.
            img_listener:    (B, 3, H, W) listener's identity image (source).
            h_start:         Optional starting hidden state for the encoder.
            mode:            ``'passive'`` or ``'active'``.
            audio_lip_feat:  (B, lip_dim) lip coefficients from Audio2Lip.
            training:        Controls VAE reparameterization vs mean-only.
            listener_mel:    (B, audio_dim) listener mel feature for active mode.

        Returns:
            img_recon, f_pose, f_exp, alpha_D_pose, alpha_D_exp,
            latent_poseD_S, mu_p, logvar_p, mu_e, logvar_e, audio_mel, mu_a, logvar_a
        """
        # 1. Speaker pre-decoder latent + per-component direction vectors
        latent_poseD_S, wa_S, f_pose_S, f_exp_S = self._speaker_latent(img_speaker)
        B      = latent_poseD_S.size(0)
        device = latent_poseD_S.device

        # 2. Listener identity appearance + spatial features
        wa_L, _, feats_L, _ = self.enc(img_listener, None, h_start)  # (B, 512)

        # 3. Mode-dependent motion features
        mu_p = mu_e = logvar_p = logvar_e = None

        if mode == 'passive':
            f_pose, f_exp = self.listener_bank.forward_passive(
                latent_poseD_S, f_pose_S, f_exp_S
            )
            alpha_D_lip  = torch.zeros(B, self.lip_dim, device=device)
            alpha_D_pose = self.pose_fc(f_pose)
            alpha_D_exp  = self.exp_fc(f_exp)

        else:  # active
            f_pose, f_exp, mu_p, logvar_p, mu_e, logvar_e = \
                self.listener_bank.forward_active(
                    latent_poseD_S, wa_S=wa_L, training=training, listener_mel=listener_mel
                )

            alpha_D_lip = (
                audio_lip_feat
                if audio_lip_feat is not None
                else torch.zeros(B, self.lip_dim, device=device)
            )
            alpha_D_pose = self.pose_fc(f_pose)
            alpha_D_exp  = self.exp_fc(f_exp)

        # 4. Direction mapping
        alpha_D_L = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a_L = self.direction_exp.get_shared_out(alpha_D_L, self.direction_lipnonlip.weight)
        e_L = self.direction_exp.get_exp_latent(a_L)
        directions_D_L = self.direction_exp(alpha_D_L, self.direction_lipnonlip.weight)

        # 5. Apply listener motion directions directly onto listener appearance.
        latent_poseD_L = wa_L + directions_D_L

        # 6. Decode
        img_recon = self.dec(latent_poseD_L, feats_L, e_L)

        return img_recon, f_pose, f_exp, alpha_D_pose, alpha_D_exp, latent_poseD_S, mu_p, logvar_p, mu_e, logvar_e

    def forward_listener(
        self,
        img_speaker,          # (B, T, C, H, W)
        img_listener,         # (B, C, H, W)  -- identity frame, fixed across T
        h_start=None,
        mode: str = 'active',
        audio_lip_feat=None,  # (B, T, lip_dim)
        training: bool = True,
        listener_mel=None,    # (B, T, N_MELS)
    ):
        """Generate an empathetic listener response over a sequence of speaker frames.

        Extends the single-frame ``forward_listener_frame`` to operate over T frames
        at once.  The listener identity (``img_listener``) is encoded once and shared
        across all timesteps.  Speaker frames are encoded in a single batched pass by
        reshaping to (B*T, ...) before the frozen encoder, then temporal context is
        introduced through a lightweight GRU applied to the bank output features
        before coefficient decoding.

        Args:
            img_speaker:     (B, T, C, H, W) speaker frames for each timestep.
            img_listener:    (B, C, H, W)    listener identity image (source), fixed.
            h_start:         Optional starting hidden state for the listener encoder.
            mode:            ``'passive'`` or ``'active'``.
            audio_lip_feat:  (B, T, lip_dim) lip coefficients from Audio2Lip.
            training:        Controls VAE reparameterization vs mean-only.
            listener_mel:    (B, T, N_MELS) ground-truth listener mel for audio VAE.

        Returns:
            img_recon:      (B, T, C, H, W) generated listener frames.
            f_pose:         (B, T, hidden_dim) temporally-contextualised pose features.
            f_exp:          (B, T, hidden_dim) temporally-contextualised exp  features.
            alpha_D_pose:   (B, T, pose_dim) pose motion coefficients.
            alpha_D_exp:    (B, T, exp_dim)  expression motion coefficients.
            latent_poseD_S: (B, T, style_dim) speaker pre-decoder latent.
            mu_p / logvar_p (B, T, latent_dim) pose VAE params,  or None in passive mode.
            mu_e / logvar_e (B, T, latent_dim) exp  VAE params,  or None in passive mode.
        """
        B, T, C, H, W = img_speaker.shape
        device = img_speaker.device

        # ------------------------------------------------------------------
        # 1. Listener identity -- encode once; appearance features are reused
        #    at every timestep via expand, avoiding redundant forward passes.
        # ------------------------------------------------------------------
        wa_L, _, feats_L, _ = self.enc(img_listener, None, h_start)
        # wa_L:    (B, 512)
        # feats_L: list[(B, C_i, H_i, W_i)]  -- skip-connection tensors for dec()

        # ------------------------------------------------------------------
        # 2. Speaker -- encode all T frames in one batched forward pass
        #    by merging the time axis into the batch axis.
        # ------------------------------------------------------------------
        spk_flat = img_speaker.view(B * T, C, H, W)
        latent_poseD_S_flat, wa_S_flat, f_pose_S_flat, f_exp_S_flat = \
            self._speaker_latent(spk_flat)
        # all: (B*T, feat_dim)

        # ------------------------------------------------------------------
        # 3. Bank retrieval -- same interface as single-frame forward_listener;
        #    inputs are (B*T, ...) so the bank processes every frame atomically.
        # ------------------------------------------------------------------
        mu_p = mu_e = logvar_p = logvar_e = None

        if mode == 'passive':
            f_pose_flat, f_exp_flat = self.listener_bank.forward_passive(
                latent_poseD_S_flat, f_pose_S_flat, f_exp_S_flat
            )

        else:  # active
            wa_L_flat = wa_L.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)

            mel_flat = (
                listener_mel.view(B * T, -1)
                if listener_mel is not None else None
            )

            (f_pose_flat, f_exp_flat,
             mu_p_flat, logvar_p_flat,
             mu_e_flat, logvar_e_flat) = \
                self.listener_bank.forward_active(
                    latent_poseD_S_flat,
                    wa_S=wa_L_flat,
                    training=training,
                    listener_mel=mel_flat,
                )
            mu_p     = mu_p_flat.view(B, T, -1)
            logvar_p = logvar_p_flat.view(B, T, -1)
            mu_e     = mu_e_flat.view(B, T, -1)
            logvar_e = logvar_e_flat.view(B, T, -1)

        # ------------------------------------------------------------------
        # 4. Temporal modelling -- a lightweight GRU propagates context from
        #    t=0 -> t=T-1 so that each frame's coefficients depend on the
        #    reaction history, not just the instantaneous speaker frame.
        # ------------------------------------------------------------------
        f_pose_seq = f_pose_flat.view(B, T, -1)   # (B, T, style_dim)
        f_exp_seq  = f_exp_flat.view(B, T, -1)    # (B, T, style_dim)

        f_pose_temporal, _ = self.temporal_gru_pose(f_pose_seq)
        f_pose_temporal = 0.5*f_pose_temporal + 0.5*f_pose_seq # skip connection
        f_exp_temporal, _  = self.temporal_gru_exp(f_exp_seq)
        f_exp_temporal = 0.5*f_exp_temporal + 0.5*f_exp_seq # skip connection

        # ------------------------------------------------------------------
        # 5. Coefficient decoding -- project temporally-enriched features to
        #    the W-space motion coefficients used by direction_exp.
        # ------------------------------------------------------------------
        alpha_D_pose = self.pose_fc(
            f_pose_temporal.reshape(B * T, -1)
        ).view(B, T, -1)   # (B, T, pose_dim)

        alpha_D_exp = self.exp_fc(
            f_exp_temporal.reshape(B * T, -1)
        ).view(B, T, -1)   # (B, T, exp_dim)

        # ------------------------------------------------------------------
        # 6. Lip coefficients -- use audio-derived features when available,
        #    otherwise keep the mouth closed (zero vector) for every frame.
        # ------------------------------------------------------------------
        alpha_D_lip = (
            audio_lip_feat
            if audio_lip_feat is not None
            else torch.zeros(B, T, self.lip_dim, device=device)
        )   # (B, T, lip_dim)

        # ------------------------------------------------------------------
        # 7. Direction mapping -- concatenate all three coefficient types and
        #    run through the shared direction network in one batched pass.
        # ------------------------------------------------------------------
        alpha_D_L_flat = torch.cat([
            alpha_D_lip.reshape(B * T, -1),
            alpha_D_pose.reshape(B * T, -1),
            alpha_D_exp.reshape(B * T, -1),
        ], dim=-1)

        a_L_flat           = self.direction_exp.get_shared_out(
            alpha_D_L_flat, self.direction_lipnonlip.weight
        )
        e_L_flat           = self.direction_exp.get_exp_latent(a_L_flat)
        directions_D_L_flat = self.direction_exp(
            alpha_D_L_flat, self.direction_lipnonlip.weight
        )

        # ------------------------------------------------------------------
        # 8. Latent composition -- add listener motion directions onto the
        #    listener identity latent:
        #      latent_poseD_L = wa_L + directions_D_L
        # ------------------------------------------------------------------
        wa_L_flat      = wa_L.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        latent_poseD_L = wa_L_flat + directions_D_L_flat   # (B*T, style_dim)

        # ------------------------------------------------------------------
        # 9. Decode -- expand skip-connection feature maps along T and decode
        #    all frames in a single call to the generator.
        # ------------------------------------------------------------------
        feats_L_exp = [
            f.unsqueeze(1)
             .expand(B, T, *f.shape[1:])
             .reshape(B * T, *f.shape[1:])
            for f in feats_L
        ]

        img_recon_flat = self.dec(latent_poseD_L, feats_L_exp, e_L_flat)

        img_recon      = img_recon_flat.view(B, T, C, H, W)
        latent_poseD_S = latent_poseD_S_flat.view(B, T, -1)

        return (
            img_recon, f_pose_temporal, f_exp_temporal,
            alpha_D_pose, alpha_D_exp, latent_poseD_S,
            mu_p, logvar_p, mu_e, logvar_e,
        )
