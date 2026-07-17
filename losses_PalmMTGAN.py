import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

# =============================================================================
# 0. FEATURE AFFINITY LOSS
# =============================================================================
class FeatureAffinityLoss(nn.Module):
    def __init__(self):
        super(FeatureAffinityLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def get_similarity_matrix(self, features):
        B, C, H, W = features.size()
        feat = F.interpolate(features, size=(40, 40), mode='bilinear', align_corners=False)
        B, C, h, w = feat.size()
        f_flat = feat.view(B, C, -1)
        sim = torch.bmm(f_flat.transpose(1, 2), f_flat)
        return F.normalize(sim, p=2, dim=-1)

    def forward(self, f_sr, f_seg):
        s_sr = self.get_similarity_matrix(f_sr)
        s_seg = self.get_similarity_matrix(f_seg)
        return self.criterion(s_seg, s_sr)

# =============================================================================
# 1. PERCEPTUAL LOSS (VGG-19)
# =============================================================================
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=34):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:feature_layer]).eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_feat = self.features(x)
        y_feat = self.features(y)
        return self.criterion(x_feat, y_feat)

# =============================================================================
# 2. ADVANCED SEGMENTATION LOSS – Focal + Dice
# =============================================================================
class PalmLoss(nn.Module):
    """
    Tversky + Focal + Hole Loss untuk segmentasi sawit.
    - Tversky dengan beta > alpha → menekan false positive.
    - Focal loss fokus pada hard examples.
    - Hole loss menghukum lubang di dalam blob (FN internal).
    """
    def __init__(self,
                 alpha_t=0.3, beta_t=0.7,      # Tversky: alpha=FN, beta=FP
                 gamma_f=2.0,                   # Focal gamma
                 hole_weight=0.2,               # bobot hole loss
                 tversky_weight=0.5,
                 focal_weight=0.5):
        super(PalmLoss, self).__init__()
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.gamma_f = gamma_f
        self.hole_weight = hole_weight
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight

        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_logits, target_mask):
        prob = torch.sigmoid(pred_logits)
        smooth = 1e-6

        # ---------- Tversky Loss ----------
        # true positive, false negative, false positive
        tp = (prob * target_mask).sum()
        fn = ((1 - prob) * target_mask).sum()
        fp = (prob * (1 - target_mask)).sum()
        tversky = (tp + smooth) / (tp + self.alpha_t * fn + self.beta_t * fp + smooth)
        loss_tversky = 1 - tversky

        # ---------- Focal Loss ----------
        bce = self.bce(pred_logits, target_mask)
        pt = torch.exp(-bce)
        focal = self.alpha_t * (1 - pt) ** self.gamma_f * bce  # alpha dari Tversky? lebih baik pakai 0.25
        loss_focal = focal.mean()

        # ---------- Hole Loss (eksperimental) ----------
        # Menghukum piksel GT=1 yang prediksinya rendah, tetapi tetangganya tinggi.
        # Gunakan max pooling 3x3 untuk menangkap lingkungan.
        if self.hole_weight > 0:
            # max pooling pada prob (bukan logits) untuk deteksi hollow
            prob_pooled = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
            # hollow terjadi jika prediksi rendah, tapi pooled tinggi, dan GT=1
            hole_penalty = (1 - prob) * torch.clamp(prob_pooled - prob, min=0) * target_mask
            loss_hole = hole_penalty.mean()
        else:
            loss_hole = 0.0

        # Total
        total = (self.tversky_weight * loss_tversky +
                 self.focal_weight * loss_focal +
                 self.hole_weight * loss_hole)
        return total

# =============================================================================
# 3. GAN LOSS (Least‑Square)
# =============================================================================
class GANLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(GANLoss, self).__init__()
        self.loss = nn.MSELoss() if reduction == 'mean' else nn.MSELoss(reduction=reduction)

    def forward(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss(pred, target)

# =============================================================================
# 4. TOTAL LOSS DENGAN UNCERTAINTY WEIGHTING
# =============================================================================
class PalmMTGANLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1,
                 gan_sr_weight=0.05, gan_seg_weight=0.05,
                 seg_weight=1.0, affinity_weight=0.1,
                 use_uncertainty=True):
        super(PalmMTGANLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()
        # Gunakan PalmLoss default (Focal‑Dice) tanpa parameter usang
        self.seg_criterion = PalmLoss()
        self.affinity = FeatureAffinityLoss()
        self.gan = GANLoss()

        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.gan_sr_weight = gan_sr_weight
        self.gan_seg_weight = gan_seg_weight
        self.seg_weight = seg_weight
        self.affinity_weight = affinity_weight

        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.log_sigma_l1 = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_percep = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_gan_sr = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_seg = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_gan_seg = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_affinity = nn.Parameter(torch.tensor(0.0))

    def forward(self, sr_fake, sr_real, seg_pred, seg_gt,
                sr_disc_fake, seg_disc_fake,
                f_sr, f_seg):
        loss_l1 = self.l1(sr_fake, sr_real)
        loss_percep = self.perceptual(sr_fake, sr_real)
        loss_gan_sr = self.gan(sr_disc_fake, target_is_real=True)
        loss_seg = self.seg_criterion(seg_pred, seg_gt)
        loss_gan_seg = self.gan(seg_disc_fake, target_is_real=True)
        loss_affinity = self.affinity(f_sr, f_seg)

        if self.use_uncertainty:
            prec_l1 = torch.exp(-self.log_sigma_l1)
            prec_percep = torch.exp(-self.log_sigma_percep)
            prec_gan_sr = torch.exp(-self.log_sigma_gan_sr)
            prec_seg = torch.exp(-self.log_sigma_seg)
            prec_gan_seg = torch.exp(-self.log_sigma_gan_seg)
            prec_affinity = torch.exp(-self.log_sigma_affinity)

            total = (prec_l1 * loss_l1 + self.log_sigma_l1) * self.l1_weight
            total += (prec_percep * loss_percep + self.log_sigma_percep) * self.perceptual_weight
            total += (prec_gan_sr * loss_gan_sr + self.log_sigma_gan_sr) * self.gan_sr_weight
            total += (prec_seg * loss_seg + self.log_sigma_seg) * self.seg_weight
            total += (prec_gan_seg * loss_gan_seg + self.log_sigma_gan_seg) * self.gan_seg_weight
            total += (prec_affinity * loss_affinity + self.log_sigma_affinity) * self.affinity_weight
        else:
            total = (self.l1_weight * loss_l1 +
                     self.perceptual_weight * loss_percep +
                     self.gan_sr_weight * loss_gan_sr +
                     self.seg_weight * loss_seg +
                     self.gan_seg_weight * loss_gan_seg +
                     self.affinity_weight * loss_affinity)

        components = {
            'loss_total': total.item() if isinstance(total, torch.Tensor) else total,
            'loss_l1': loss_l1.item(),
            'loss_percep': loss_percep.item(),
            'loss_gan_sr': loss_gan_sr.item(),
            'loss_seg': loss_seg.item(),
            'loss_gan_seg': loss_gan_seg.item(),
            'loss_affinity': loss_affinity.item()
        }
        return total, components

# =============================================================================
# 5. DISCRIMINATOR LOSS
# =============================================================================
def discriminator_sr_loss(disc_sr, sr_real, sr_fake):
    gan = GANLoss()
    pred_real = disc_sr(sr_real)
    pred_fake = disc_sr(sr_fake.detach())
    loss_real = gan(pred_real, target_is_real=True)
    loss_fake = gan(pred_fake, target_is_real=False)
    return (loss_real + loss_fake) * 0.5

def discriminator_seg_loss(disc_seg, seg_real, seg_fake):
    gan = GANLoss()
    pred_real = disc_seg(seg_real)
    pred_fake = disc_seg(seg_fake.detach())
    loss_real = gan(pred_real, target_is_real=True)
    loss_fake = gan(pred_fake, target_is_real=False)
    return (loss_real + loss_fake) * 0.5