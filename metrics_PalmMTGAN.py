import os
import torch
import math
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim_metric
import lpips

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def _calculate_mse(img1, img2):
    return torch.mean((img1.float() - img2.float()) ** 2).item()

def _calculate_rmse(img1, img2):
    return math.sqrt(_calculate_mse(img1, img2))

def _to_3d(img):
    """Pastikan tensor menjadi 3D (C,H,W) jika 4D ambil batch pertama."""
    if img.dim() == 4:
        return img[0]
    return img

# ==========================================
# 8 METRIK SUPER‑RESOLUTION (SR) – tetap sama
# ==========================================
def calculate_psnr(img1, img2, max_val=1.0):
    img1, img2 = _to_3d(img1), _to_3d(img2)
    mse = _calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))

def calculate_ssim(img1, img2, max_val=1.0):
    img1, img2 = _to_3d(img1), _to_3d(img2)
    img1_np = img1.detach().float().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.detach().float().cpu().numpy().transpose(1, 2, 0)
    return ssim_metric(img1_np, img2_np, data_range=max_val, channel_axis=2)

def calculate_sam(img1, img2):
    img1, img2 = _to_3d(img1), _to_3d(img2)
    img1_flat = img1.view(img1.shape[0], -1).detach().float()
    img2_flat = img2.view(img2.shape[0], -1).detach().float()
    dot_product = torch.sum(img1_flat * img2_flat, dim=0)
    norm1 = torch.norm(img1_flat, dim=0)
    norm2 = torch.norm(img2_flat, dim=0)
    cos_sim = torch.clamp(dot_product / (norm1 * norm2 + 1e-8), -1.0, 1.0)
    return torch.mean(torch.acos(cos_sim)).item()

def calculate_ergas(img1, img2, scale_factor=5):
    img1, img2 = _to_3d(img1), _to_3d(img2)
    channels = img1.shape[0]
    rmse_sum = 0
    for c in range(channels):
        rmse_c = _calculate_rmse(img1[c].detach(), img2[c].detach())
        mean_c = torch.mean(img2[c].detach().float()).item()
        if mean_c != 0:
            rmse_sum += (rmse_c / mean_c) ** 2
    return 100 * (1.0 / scale_factor) * math.sqrt(rmse_sum / channels)

def calculate_cc(img1, img2):
    img1, img2 = _to_3d(img1), _to_3d(img2)
    img1, img2 = img1.detach().float(), img2.detach().float()
    mean1, mean2 = torch.mean(img1), torch.mean(img2)
    img1_centered = img1 - mean1
    img2_centered = img2 - mean2
    numerator = torch.sum(img1_centered * img2_centered)
    denominator = math.sqrt(torch.sum(img1_centered ** 2) * torch.sum(img2_centered ** 2))
    return (numerator / denominator).item() if denominator != 0 else 0.0

def calculate_sdi(img1, img2):
    img1, img2 = _to_3d(img1), _to_3d(img2)
    return torch.mean(torch.abs(img1.float() - img2.float())).item()

def calculate_epi(img1, img2):
    img1, img2 = _to_3d(img1), _to_3d(img2)
    x1 = img1.unsqueeze(0).detach().float()
    x2 = img2.unsqueeze(0).detach().float()
    def laplacian(x):
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype).to(x.device).view(1, 1, 3, 3)
        return F.conv2d(x.mean(dim=1, keepdim=True), k, padding=1)
    l_img1 = laplacian(x1)
    l_img2 = laplacian(x2)
    stacked = torch.stack([l_img1.view(-1), l_img2.view(-1)])
    return torch.corrcoef(stacked)[0, 1].item()

_lpips_vgg = None
def calculate_lpips(img1, img2):
    global _lpips_vgg
    device = img1.device
    if _lpips_vgg is None:
        _lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    else:
        _lpips_vgg = _lpips_vgg.to(device)
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    elif img1.dim() == 4:
        pass
    else:
        raise ValueError(f"Input harus 3D atau 4D, tetapi mendapat shape {img1.shape}")
    img1_norm = (img1.detach().float() * 2.0) - 1.0
    img2_norm = (img2.detach().float() * 2.0) - 1.0
    with torch.no_grad():
        lpips_val = _lpips_vgg(img1_norm, img2_norm)
    return lpips_val.mean().item()

# ==========================================
# METRIK SEGMENTASI BINER (DITAMBAHKAN TOTAL AREA ERROR)
# ==========================================
def calculate_segmentation_metrics(pred_logits, true_mask, threshold=0.5, return_total_error=False):
    """
    Input:
        pred_logits: tensor 3D (1,H,W) atau 4D (B,1,H,W)
        true_mask:   tensor 3D (1,H,W) atau 4D (B,1,H,W)
        return_total_error: jika True, mengembalikan tambahan:
            total_error_pixels (FP+FN) dan total_error_percent (terhadap seluruh gambar)
    Output dasar: (iou, f1_score, precision, recall, net_area_error)
    Output tambahan (bila return_total_error=True): + (total_error_pixels, total_error_percent)
    """
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.unsqueeze(0)
    if true_mask.dim() == 3:
        true_mask = true_mask.unsqueeze(0)

    probs = torch.sigmoid(pred_logits.detach().float())
    pred_mask = (probs >= threshold).float()
    true_mask = true_mask.detach().float()

    iou_list, f1_list, prec_list, rec_list, ae_list = [], [], [], [], []
    total_error_px_list, total_error_pct_list = [], []

    for b in range(pred_mask.size(0)):
        pm = pred_mask[b]
        tm = true_mask[b]

        TP = torch.sum((pm == 1) & (tm == 1)).item()
        FP = torch.sum((pm == 1) & (tm == 0)).item()
        FN = torch.sum((pm == 0) & (tm == 1)).item()
        total_pixels = pm.numel()  # H*W

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

        pred_area = torch.sum(pm).item()
        true_area = torch.sum(tm).item()
        if true_area == 0:
            ae = 0.0 if pred_area == 0 else 100.0
        else:
            ae = abs(pred_area - true_area) / true_area * 100.0

        iou_list.append(iou)
        f1_list.append(f1)
        prec_list.append(precision)
        rec_list.append(recall)
        ae_list.append(ae)

        if return_total_error:
            total_err_px = FP + FN
            total_err_pct = total_err_px / total_pixels * 100.0
            total_error_px_list.append(total_err_px)
            total_error_pct_list.append(total_err_pct)

    # Rata-rata atas batch
    n = len(iou_list)
    base = (sum(iou_list)/n, sum(f1_list)/n, sum(prec_list)/n, sum(rec_list)/n, sum(ae_list)/n)

    if return_total_error:
        avg_tot_px = sum(total_error_px_list) / n
        avg_tot_pct = sum(total_error_pct_list) / n
        return base + (avg_tot_px, avg_tot_pct)
    else:
        return base

# ==========================================
# MASTER EVALUATOR (METRIK LENGKAP + TOTAL AREA ERROR)
# ==========================================
def evaluate_batch(pred_sr, target_sr, pred_seg_logits, target_seg_mask, scale_factor=5):
    batch_size = pred_sr.shape[0]

    metrics = {k: 0.0 for k in ['PSNR', 'SSIM', 'SAM', 'CC', 'ERGAS', 'EPI', 'SDI', 'LPIPS',
                                'IoU', 'F1_Score', 'Precision', 'Recall', 'Area_Error',
                                'Total_Error_px', 'Total_Error_percent']}

    for i in range(batch_size):
        # SR
        metrics['PSNR'] += calculate_psnr(pred_sr[i], target_sr[i])
        metrics['SSIM'] += calculate_ssim(pred_sr[i], target_sr[i])
        metrics['SAM'] += calculate_sam(pred_sr[i], target_sr[i])
        metrics['CC'] += calculate_cc(pred_sr[i], target_sr[i])
        metrics['ERGAS'] += calculate_ergas(pred_sr[i], target_sr[i], scale_factor)
        metrics['EPI'] += calculate_epi(pred_sr[i], target_sr[i])
        metrics['SDI'] += calculate_sdi(pred_sr[i], target_sr[i])
        metrics['LPIPS'] += calculate_lpips(pred_sr[i], target_sr[i])

        # Seg dengan total error
        iou, f1, prec, rec, ae, tot_px, tot_pct = calculate_segmentation_metrics(
            pred_seg_logits[i:i+1], target_seg_mask[i:i+1], return_total_error=True
        )
        metrics['IoU'] += iou
        metrics['F1_Score'] += f1
        metrics['Precision'] += prec
        metrics['Recall'] += rec
        metrics['Area_Error'] += ae
        metrics['Total_Error_px'] += tot_px
        metrics['Total_Error_percent'] += tot_pct

    for k in metrics.keys():
        metrics[k] /= batch_size

    return metrics

# ==========================================================
# FUNGSI SAVE VISUAL (TETAP)
# ==========================================================
def save_visuals_grid_header(lr, hr, sr, m_gt, m_pr, name, path):
    os.makedirs(path, exist_ok=True)
    m_pr_sig = torch.sigmoid(m_pr.detach().float())

    with torch.no_grad():
        def prep(t):
            return torch.clamp(t.float(), 0, 1)

        lr_up = F.interpolate(lr.float(), size=hr.shape[2:], mode='bilinear', align_corners=False)

        imgs = [
            prep(lr_up * 3.5)[0],
            prep(sr)[0],
            prep(hr)[0],
            m_gt[0].float().repeat(3, 1, 1),
            (m_pr_sig[0] > 0.5).float().repeat(3, 1, 1)
        ]

        grid = vutils.make_grid(imgs, nrow=5, padding=4).permute(1, 2, 0).mul(255).to('cpu', torch.uint8).numpy()
        canvas = Image.fromarray(grid)

        w, h = canvas.size
        new_canvas = Image.new('RGB', (w, h + 30), (0, 0, 0))
        new_canvas.paste(canvas, (0, 30))

        draw = ImageDraw.Draw(new_canvas)
        labels = ["Sentinel-2", "SR (2m)", "HR-UAV", "Ground Truth", "Segmentation"]

        box_w = (w - (6 * 4)) // 5
        for i, label in enumerate(labels):
            start_x = 4 + i * (box_w + 4)
            text_pos = (start_x + (box_w // 12), 8)
            draw.text(text_pos, label, fill=(255, 255, 255))

        clean_name = name.replace('.tif', '')
        new_canvas.save(os.path.join(path, f"{clean_name}.png"))