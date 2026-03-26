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
            # out = torch.sum(out, dim=1)

            # return out
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
            # out = torch.sum(out, dim=1)

            # return out
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
                 mirror_alpha_p=0.2, mirror_alpha_e=0.2):
        super(Generator, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
        # self.direction = Direction(motion_dim)
        self.direction_lipnonlip = Direction(lip_dim, pose_dim)
        self.direction_exp = Direction_exp(lip_dim, pose_dim, exp_dim)
        # motion network
        fc = [EqualLinear(style_dim, style_dim)]
        for i in range(3):
            fc.append(EqualLinear(style_dim, style_dim))
        self.fc = nn.Sequential(*fc)
        # self.source_fc = EqualLinear(style_dim, motion_dim)

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
            mirror_alpha_p=mirror_alpha_p,
            mirror_alpha_e=mirror_alpha_e,
        )


    def test_EDTalk_V(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon

    def test_EDTalk_V_use_exp_weight(self, img_source, lip_img_drive, pose_img_drive, alpha_D_exp, h_start=None):

        wa, wa_t, feats, _ = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon

    def test_EDTalk_A(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, feats_t = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon


    def test_EDTalk_A_use_exp_weight(self, img_source, lip_img_drive, pose_img_drive, alpha_D_exp, h_start=None):

        wa, wa_t_p, feats, _ = self.enc(img_source, pose_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
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

    def forward_listener(
        self,
        img_speaker,
        img_listener,
        h_start=None,
        mode: str = 'active',
        audio_lip_feat=None,
        training: bool = True,
    ):
        """Generate an empathetic listener response conditioned on a speaker frame.

        Two modes are supported:

        **passive** (Stage 1 – silent listening / micro-expression mirroring):
            - Lip: zeroed out (mouth stays closed).
            - Pose / Exp: bank retrieval + α-scaled speaker mirroring (deterministic).
            - No VAE; ``mu_*`` and ``logvar_*`` are ``None``.

        **active** (Stage 3 – empathic backchannel):
            - Lip: ``audio_lip_feat`` from the TTS → Audio2Lip path (20-dim).
            - Pose / Exp: stochastically sampled via :class:`ReactionVAE`.
            - Returns VAE parameters for KL loss.

        Args:
            img_speaker:     (B, 3, H, W) speaker's current frame.
            img_listener:    (B, 3, H, W) listener's identity image (source).
            h_start:         Optional starting hidden state for the encoder.
            mode:            ``'passive'`` or ``'active'``.
            audio_lip_feat:  (B, lip_dim) lip coefficients from Audio2Lip.
                             Used only when ``mode='active'``.
            training:        Controls VAE reparameterization vs mean-only.

        Returns:
            img_recon:      (B, 3, H, W) generated listener frame.
            f_pose:         (B, style_dim) pose feature used.
            f_exp:          (B, style_dim) expression feature used.
            latent_poseD_S: (B, style_dim) speaker's pre-decoder latent.
            mu_p:           (B, latent_dim) pose VAE mean,     or None in passive mode.
            logvar_p:       (B, latent_dim) pose VAE log-var,  or None in passive mode.
            mu_e:           (B, latent_dim) exp  VAE mean,     or None in passive mode.
            logvar_e:       (B, latent_dim) exp  VAE log-var,  or None in passive mode.
        """
        # 1. Speaker pre-decoder latent + per-component direction vectors
        latent_poseD_S, wa_S, f_pose_S, f_exp_S = self._speaker_latent(img_speaker)
        B      = latent_poseD_S.size(0)
        device = latent_poseD_S.device

        # 2. Listener identity appearance + spatial features
        wa_L, _, feats_L, _ = self.enc(img_listener, None, h_start)  # (B, 512)
        # wa_L=h_source: final latent from conv
        # feats: middle latents from conv

        # 3. Mode-dependent motion features
        mu_p = mu_e = logvar_p = logvar_e = None

        if mode == 'passive':
            # Stage 1: silent mirroring — lip closed, pose/exp echo speaker softly
            f_pose, f_exp = self.listener_bank.forward_passive(
                latent_poseD_S, f_pose_S, f_exp_S
            )
            # Lip coefficient = 0 → no mouth movement
            alpha_D_lip  = torch.zeros(B, self.lip_dim, device=device)
            alpha_D_pose = self.pose_fc(self.fc(f_pose))
            alpha_D_exp  = self.exp_fc(self.fc(f_exp))

        else:  # active # empathy 학습
            # Stage 3: stochastic pose/exp from VAE; lip driven by TTS audio.
            # Pass wa_L as VAE context: listener identity conditions the reaction,
            # speaker influence comes only through bank retrieval (latent_poseD_S).
            f_pose, f_exp, mu_p, logvar_p, mu_e, logvar_e = \
                self.listener_bank.forward_active(latent_poseD_S, wa_L=wa_L, training=training) # 여기서 cross attention

            # Use audio-derived lip features if provided, otherwise zero
            alpha_D_lip = (
                audio_lip_feat
                if audio_lip_feat is not None
                else torch.zeros(B, self.lip_dim, device=device)
            )
            alpha_D_pose = self.pose_fc(self.fc(f_pose))
            alpha_D_exp  = self.exp_fc(self.fc(f_exp))

        # 4. Direction mapping
        alpha_D_L = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a_L = self.direction_exp.get_shared_out(alpha_D_L, self.direction_lipnonlip.weight) # orthogonal
        e_L = self.direction_exp.get_exp_latent(a_L) # orthogonal
        directions_D_L = self.direction_exp(alpha_D_L, self.direction_lipnonlip.weight)

        # 5. Apply listener motion directions directly onto listener appearance.
        # directions_D_L is in the pretrained W-space and directly encodes
        # head pose / expression motion — identical to how the speaker latent is built:
        #   latent_poseD_S = wa_S + directions_D_S
        # Passing through fuse() (random MLP) destroys the motion signal.
        latent_poseD_L = wa_L + directions_D_L

        # 6. Decode
        img_recon = self.dec(latent_poseD_L, feats_L, e_L)

        return img_recon, f_pose, f_exp, latent_poseD_L, mu_p, logvar_p, mu_e, logvar_e