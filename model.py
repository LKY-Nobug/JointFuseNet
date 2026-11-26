import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------

# -------------------------
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, norm=nn.InstanceNorm3d):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            norm(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, norm=nn.InstanceNorm3d):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 2, 1, bias=False),
            norm(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, norm=nn.InstanceNorm3d):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels * 2, out_channels, norm=norm)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)
            skip = skip[
                :,
                :,
                diff_d // 2: diff_d // 2 + x.size(2),
                diff_h // 2: diff_h // 2 + x.size(3),
                diff_w // 2: diff_w // 2 + x.size(4),
            ]
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# -------------------------

# -------------------------
class CrossModalFusion3D(nn.Module):
    """Exchange seismic and electrical features within their shared axial coverage."""
    def __init__(self, channels, num_heads=4, ffn_expansion=4,p_drop=0.1):
        super().__init__()
        self.channels = channels
        self.attn_s = nn.MultiheadAttention(channels, num_heads, batch_first=True,dropout=p_drop)
        self.attn_e = nn.MultiheadAttention(channels, num_heads, batch_first=True,dropout=p_drop)

        hidden = channels * ffn_expansion
        self.norm_q_s = nn.LayerNorm(channels)
        self.norm_q_e = nn.LayerNorm(channels)
        self.norm_kv_s = nn.LayerNorm(channels)
        self.norm_kv_e = nn.LayerNorm(channels)

        self.ffn_s = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, channels),
        )
        self.ffn_e = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, channels),
        )
        self.norm_ffn_s = nn.LayerNorm(channels)
        self.norm_ffn_e = nn.LayerNorm(channels)

    def forward(self, feat_seis, feat_elec):
        B, C, D, Hs, W = feat_seis.shape
        _, Ce, De, He, We = feat_elec.shape
        assert C == Ce and D == De and W == We and He <= Hs


        s_ol = feat_seis[:, :, :, :He, :]  # [B,C,D,He,W]
        e_all = feat_elec                  # [B,C,D,He,W]

        # -> tokens [B, N, C]
        S0 = s_ol.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        E0 = e_all.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

        # seismic: Q=S, KV=E
        q_s = self.norm_q_s(S0)
        kv_e = self.norm_kv_e(E0)
        attn_s, _ = self.attn_s(q_s, kv_e, kv_e)
        S = S0 + attn_s
        S = S + self.ffn_s(self.norm_ffn_s(S))

        # electric: Q=E, KV=S
        q_e = self.norm_q_e(E0)
        kv_s = self.norm_kv_s(S0)
        attn_e, _ = self.attn_e(q_e, kv_s, kv_s)
        E = E0 + attn_e
        E = E + self.ffn_e(self.norm_ffn_e(E))


        s_ol_fused = S.reshape(B, D, He, W, C).permute(0, 4, 1, 2, 3)
        e_fused = E.reshape(B, D, He, W, C).permute(0, 4, 1, 2, 3)

        joint_seis = feat_seis.clone()
        joint_seis[:, :, :, :He, :] = s_ol_fused

        return joint_seis, e_fused

class ChannelAttention3D(nn.Module):
    """Learn one channel weight per sample from globally pooled features."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)  # [B,C,D,H,W] -> [B,C,1,1,1]

        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, D, H, W = x.shape

        y = self.pool(x).view(B, C)        # [B, C]


        w = self.fc(y).view(B, C, 1, 1, 1) # [B, C, 1, 1, 1]


        return x * w

class AxialTransformerBlock3D(nn.Module):
    """Apply channel attention and self-attention along the tunnel axis."""
    def __init__(self, channels, num_heads=4, ffn_expansion=4, dropout=0.2,ca_reduction=8):
        super().__init__()
        self.channels = channels

        self.channel_attn = ChannelAttention3D(channels, reduction=ca_reduction)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        hidden = channels * ffn_expansion
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
        )
    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape


        x = self.channel_attn(x)           # [B, C, D, H, W]


        # [B, C, D, H, W] -> [B, D, W, H, C] -> [B*D*W, H, C]
        x_seq = x.permute(0, 2, 4, 3, 1).contiguous()   # [B, D, W, H, C]
        x_seq = x_seq.view(B * D * W, H, C)            # [B*D*W, H, C]

        res = x_seq
        h = self.norm1(x_seq)
        attn_out, _ = self.attn(h, h, h)               # [B*D*W, H, C]
        x_seq = res + attn_out


        res2 = x_seq
        h2 = self.norm2(x_seq)
        ffn_out = self.ffn(h2)
        x_seq = res2 + ffn_out                         # [B*D*W, H, C]


        x_seq = x_seq.view(B, D, W, H, C)              # [B, D, W, H, C]
        x_out = x_seq.permute(0, 4, 1, 3, 2).contiguous()

        return x_out


# -------------------------

# -------------------------
VELOCITY_RANGE = (1500.0, 4000.0)
RESISTIVITY_RANGE = (375.0, 1000.0)


class VelocityHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, F_dec):

        vel_pad = self.conv(F_dec)                    # [B,1,32,128,32]
        vel = vel_pad[:, :, 1:31, 4:124, 1:31]        # [B,1,30,120,30]
        lower, upper = VELOCITY_RANGE
        return lower + torch.sigmoid(vel) * (upper - lower)


# -------------------------

# -------------------------
class ResistivityHead(nn.Module):
    def __init__(self, in_dec_channels, in_elec_channels, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.fuse_conv = nn.Conv3d(in_dec_channels + in_elec_channels,
                                   in_dec_channels, kernel_size=1)
        self.out_conv = nn.Conv3d(in_dec_channels, 1, kernel_size=1)


    def forward(self, F_dec_region, E_ref_region):
        """Return the resistivity estimate from the shared-region decoder features."""
        x = torch.cat([F_dec_region, E_ref_region], dim=1)  # [B,Cd+Ce,30,30,30]
        x = self.fuse_conv(x)                               # [B,Cd,30,30,30]
        x = F_dec_region + self.alpha * x
        rho = self.out_conv(x)                              # [B,1,30,30,30]
        lower, upper = RESISTIVITY_RANGE
        return lower + torch.sigmoid(rho) * (upper - lower)


# -------------------------


#   - VelocityHead + ResistivityHead
# -------------------------
class JointFuseNet(nn.Module):
    """Predict velocity over 120 m and resistivity over the shared 30 m."""
    def __init__(
        self,
        base_channels=32,
        # base_channels=16,
        num_heads=4,
        num_epa_blocks=2,
        norm=nn.InstanceNorm3d,
        alpha=0.5,
    ):
        super().__init__()
        self.base_channels = base_channels

        # ---- Seismic Encoder ----
        self.seis_enc0 = ConvBlock3D(1, base_channels, norm)
        self.seis_down1 = DownBlock3D(base_channels, base_channels * 2, norm)
        self.seis_down2 = DownBlock3D(base_channels * 2, base_channels * 4, norm)
        self.seis_down3 = DownBlock3D(base_channels * 4, base_channels * 8, norm)

        # ---- Electric Encoder ----
        self.elec_enc0 = ConvBlock3D(1, base_channels, norm)
        self.elec_down1 = DownBlock3D(base_channels, base_channels * 2, norm)
        self.elec_down2 = DownBlock3D(base_channels * 2, base_channels * 4, norm)

        # ---- Cross-Attention at level2 ----
        self.fusion_l2 = CrossModalFusion3D(base_channels * 4, num_heads)


        self.dropout_bottleneck = nn.Dropout3d(p=0.4)


        self.global_epa = nn.Sequential(
            *[AxialTransformerBlock3D(base_channels * 8, num_heads) for _ in range(num_epa_blocks)]
        )


        self.up3 = UpBlock3D(base_channels * 8, base_channels * 4, norm)  # skip: joint2
        self.up2 = UpBlock3D(base_channels * 4, base_channels * 2, norm)  # skip: s1
        self.up1 = UpBlock3D(base_channels * 2, base_channels, norm)      # skip: s0


        self.up_e2 = UpBlock3D(base_channels * 4, base_channels * 2, norm)  # e2_fused -> e1
        self.up_e1 = UpBlock3D(base_channels * 2, base_channels, norm)      # e1 -> e0

        # ---- Heads ----
        self.vel_head = VelocityHead(in_channels=base_channels)
        self.rho_head = ResistivityHead(
            in_dec_channels=base_channels,
            in_elec_channels=base_channels,
            alpha=alpha,
        )

    def forward(self, seismic, electric):
        """
        seismic:  [B,1,30,120,30]
        electric: [B,1,30, 30,30]
        """
        # ---- padding ----
        seis_pad = F.pad(seismic, (1, 1, 4, 4, 1, 1))   # [B,1,32,128,32]
        elec_pad = F.pad(electric, (1, 1, 1, 1, 1, 1))  # [B,1,32, 32,32]

        # ---- Seismic Encoder ----
        s0 = self.seis_enc0(seis_pad)   # [B,C,32,128,32]
        s1 = self.seis_down1(s0)        # [B,2C,16,64,16]
        s2 = self.seis_down2(s1)        # [B,4C, 8,32, 8]

        # ---- Electric Encoder ----
        e0 = self.elec_enc0(elec_pad)   # [B,C,32,32,32]
        e1 = self.elec_down1(e0)        # [B,2C,16,16,16]
        e2 = self.elec_down2(e1)        # [B,4C, 8, 8, 8]


        joint2, e2_fused = self.fusion_l2(s2, e2)   # joint2: [B,4C,8,32,8]


        s3 = self.seis_down3(joint2)                # [B,8C,4,16,4]


        s3 = self.dropout_bottleneck(s3)


        b = self.global_epa(s3)                     # [B,8C,4,16,4]


        d2 = self.up3(b, joint2)
        d1 = self.up2(d2, s1)           # skip: s1
        F_dec = self.up1(d1, s0)        # skip: s0  -> [B,C,32,128,32]


        de1 = self.up_e2(e2_fused, e1)       # [B,2C,16,16,16]
        de0 = self.up_e1(de1, e0)           # [B, C,32,32,32]
        E_ref_high = de0                    # [B, C,32,32,32]


        vel = self.vel_head(F_dec)          # [B,1,30,120,30]


        F_rho_region = F_dec[:, :, 1:31, 4:34, 1:31]      # [B,C,30,30,30]
        E_rho_region = E_ref_high[:, :, 1:31, 1:31, 1:31] # [B,C,30,30,30]
        rho = self.rho_head(F_rho_region, E_rho_region)   # [B,1,30,30,30]

        return vel, rho
