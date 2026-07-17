import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# =============================================================================
# 1. BLOK FUNDAMENTAL (RRDB)
# =============================================================================
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

# =============================================================================
# 2. MODUL UPSAMPLE (PixelShuffle 2x)
# =============================================================================
class UpsampleBlock(nn.Module):
    def __init__(self, nf):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.pixelshuffle(self.conv(x)))

# =============================================================================
# 3. MODUL ASPP
# =============================================================================
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        mid = 64
        self.aspp1 = nn.Sequential(nn.Conv2d(in_channels, mid, 1, bias=False),
                                   nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.aspp2 = nn.Sequential(nn.Conv2d(in_channels, mid, 3, padding=6, dilation=6, bias=False),
                                   nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.aspp3 = nn.Sequential(nn.Conv2d(in_channels, mid, 3, padding=12, dilation=12, bias=False),
                                   nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.aspp4 = nn.Sequential(nn.Conv2d(in_channels, mid, 3, padding=18, dilation=18, bias=False),
                                   nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                             nn.Conv2d(in_channels, mid, 1, bias=False),
                                             nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(mid * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=x.shape[2:],
                           mode='bilinear', align_corners=False)
        return self.conv_out(torch.cat([x1, x2, x3, x4, x5], dim=1))

# =============================================================================
# 4. MODUL BAA (Boundary Aware Attention) – dengan ReLU
# =============================================================================
class BAA(nn.Module):
    def __init__(self, in_channels):
        super(BAA, self).__init__()
        self.gcb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (7, 1), padding=(3, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, (1, 7), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, (7, 1), padding=(3, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, (1, 7), padding=(0, 3))
        )
        self.brb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )
        self.gate = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        gcb_out = self.gcb(x)
        brb_out = self.brb(gcb_out)
        refined = gcb_out + brb_out
        return x * self.gate(refined)

# =============================================================================
# 5. SPECTRAL PRESERVATION MODULE
# =============================================================================
class SpectralPreservationModule(nn.Module):
    def __init__(self, nf=64):
        super(SpectralPreservationModule, self).__init__()
        self.conv_lr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.weight_map = nn.Sequential(nn.Conv2d(nf, nf, 1), nn.Sigmoid())

    def forward(self, f_lr, f_sr):
        g_spc = self.conv_lr(f_lr)
        w = self.weight_map(f_sr)
        return w * g_spc + (1 - w) * f_sr

# =============================================================================
# 6. GENERATOR (PalmMTGAN) – dengan SPM, progressive upsampling, skip connections
# =============================================================================
class PalmMTGAN_Generator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, use_checkpoint=False):
        super(PalmMTGAN_Generator, self).__init__()
        self.use_checkpoint = use_checkpoint

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.rrdb_shared = nn.ModuleList([RRDB(nf) for _ in range(11)])
        self.rrdb_sr = nn.ModuleList([RRDB(nf) for _ in range(12)])

        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.spm = SpectralPreservationModule(nf)

        self.up1 = UpsampleBlock(nf)
        self.up2 = UpsampleBlock(nf)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last_sr = nn.Conv2d(nf, out_nc, 3, 1, 1)

        # Segmentasi
        self.skip_proj1 = nn.Conv2d(nf, nf, 1)
        self.skip_proj6 = nn.Conv2d(nf, nf, 1)
        self.skip_proj11 = nn.Conv2d(nf, nf, 1)
        self.skip_proj22 = nn.Conv2d(nf, nf, 1)

        self.skip_fusion = nn.Sequential(
            nn.Conv2d(nf * 4, 256, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp = ASPP(256, 256)
        self.baa = BAA(256)

        self.seg_up1 = UpsampleBlock(256)
        self.seg_up2 = UpsampleBlock(256)
        self.seg_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_last_seg = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)

        out = fea
        skips = []
        for idx, blk in enumerate(self.rrdb_shared):
            if self.use_checkpoint:
                out = checkpoint(blk, out)
            else:
                out = blk(out)
            if idx in [0, 5, 10]:
                skips.append(out)
        shared_feat = out
        skip1, skip6, skip11 = skips[0], skips[1], skips[2]

        out_sr = shared_feat
        for blk in self.rrdb_sr:
            if self.use_checkpoint:
                out_sr = checkpoint(blk, out_sr)
            else:
                out_sr = blk(out_sr)
        sr_feat_deep = out_sr

        out_sr = self.trunk_conv(out_sr) + fea
        out_sr = self.spm(fea, out_sr)          # <-- SPM diterapkan

        sr_up = self.up1(out_sr)
        sr_up = self.up2(sr_up)
        sr_up = F.interpolate(sr_up, scale_factor=1.25, mode='bilinear', align_corners=False)
        sr_up = self.conv_hr(sr_up)
        sr_final = self.conv_last_sr(sr_up)

        # Segmentasi
        s1 = F.relu(self.skip_proj1(skip1))
        s6 = F.relu(self.skip_proj6(skip6))
        s11 = F.relu(self.skip_proj11(skip11))
        s22 = F.relu(self.skip_proj22(sr_feat_deep.detach()))

        fused = torch.cat([s1, s6, s11, s22], dim=1)
        fused = self.skip_fusion(fused)

        seg_feat = self.aspp(fused)
        seg_feat = self.baa(seg_feat)

        seg_up = self.seg_up1(seg_feat)
        seg_up = self.seg_up2(seg_up)
        seg_up = F.interpolate(seg_up, scale_factor=1.25, mode='bilinear', align_corners=False)
        seg_up = self.seg_conv(seg_up)
        seg_final = self.conv_last_seg(seg_up)

        return sr_final, seg_final, out_sr, seg_feat

# =============================================================================
# 7. DISCRIMINATORS
# =============================================================================
class Discriminator_SR(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(Discriminator_SR, self).__init__()
        def block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, stride, 1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.model = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1), nn.LeakyReLU(0.2),
            block(nf, nf, 2), block(nf, nf*2, 1), block(nf*2, nf*2, 2),
            block(nf*2, nf*4, 1), block(nf*4, nf*4, 2), block(nf*4, nf*8, 1), block(nf*8, nf*8, 2),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(nf*8, 1, 1)
        )
    def forward(self, x): return self.model(x)

class Discriminator_Seg(nn.Module):
    """
    PatchGAN Discriminator untuk segmentasi.
    Menilai realisme pada level patch lokal, bukan satu skalar global.
    """
    def __init__(self, in_nc=1, nf=64, noise_std=0.2):
        super(Discriminator_Seg, self).__init__()
        self.noise_std = noise_std
        def conv_block(in_f, out_f, stride=2, norm=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(in_nc, nf, stride=2, norm=False),   # 80x80
            conv_block(nf, nf*2, stride=2),                 # 40x40
            conv_block(nf*2, nf*4, stride=2),               # 20x20
            nn.Conv2d(nf*4, 1, kernel_size=4, stride=1, padding=1)  # output: patch logits
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return self.model(x)