import torch
import os
import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

# --- 1. PENGUNCI REPLIKASI ---
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- 2. IMPORT MODUL LOKAL ---
from models_PalmMTGAN import PalmMTGAN_Generator
from dataset import PalmDataset
from metrics_PalmMTGAN import (
    calculate_psnr, calculate_ssim, calculate_sam, calculate_cc,
    calculate_ergas, calculate_epi, calculate_sdi, calculate_lpips,
    calculate_segmentation_metrics      # sekarang bisa return 7 nilai
)

# --- 3. KONFIGURASI PATH & PARAMETER ---
MODEL_PATH = Path(r"D:\S3\Project\UAV_Satelit\Training\PalmMTGAN_V1\best_gan_IoU_tuning_NAE.pth")
DATA_PATH = Path(r"D:\S3\Project\UAV_Satelit\Dataset_PalmSen2UAV_9537data\test")

BASE_OUT = Path(r"D:\S3\Project\UAV_Satelit\Training\PalmMTGAN_V1\data test best_gan_IoU_tuning_NAE")
SR_ONLY_PATH = BASE_OUT / "sr_results"
SEG_ONLY_PATH = BASE_OUT / "segmentation_results"
REPORT_PATH = BASE_OUT / "test_report_best_gan_IoU_tuning_NAE.txt"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5
BATCH_SIZE = 1

def test_palmmtgan_v3d():
    for p in [SR_ONLY_PATH, SEG_ONLY_PATH]:
        p.mkdir(parents=True, exist_ok=True)

    print("="*85)
    print("🚀 EVALUASI MODEL: PalmMTGAN best_gan_IoU_tuning_NAE (Dengan Total Error)")
    print(f"📌 THRESHOLD : {THRESHOLD}")
    print(f"📌 DEVICE    : {DEVICE}")
    print("="*85)

    # Inisialisasi Generator
    netG = PalmMTGAN_Generator().to(DEVICE)
    state_dict = torch.load(str(MODEL_PATH), map_location=DEVICE)
    netG.load_state_dict(state_dict)
    netG.eval()

    test_loader = DataLoader(PalmDataset(DATA_PATH), BATCH_SIZE, shuffle=False)

    # Inisialisasi metrik (tambahkan Total_Error_px dan Total_Error_percent)
    m = {k: 0.0 for k in ['psnr', 'ssim', 'sam', 'cc', 'ergas', 'epi', 'sdi', 'lpips',
                         'iou', 'f1', 'precision', 'recall', 'ae',
                         'total_error_px', 'total_error_pct']}
    n = len(test_loader)

    print(f"🚀 Memproses {n} citra test...")

    with torch.no_grad():
        for i, (lr, hr, mask, filenames) in enumerate(tqdm(test_loader, desc="Testing")):
            lr, hr, mask = lr.to(DEVICE), hr.to(DEVICE), mask.to(DEVICE)
            filename = filenames[0]

            # Inferensi
            sr, seg_logits, _, _ = netG(lr)

            # --- 1. SIMPAN HASIL SR ---
            sr_cpu = torch.clamp(sr[0].detach().cpu(), 0, 1)
            sr_img = Image.fromarray((sr_cpu.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            sr_img.save(SR_ONLY_PATH / filename)

            # --- 2. SIMPAN HASIL SEGMENTASI (BINER) ---
            seg_prob = torch.sigmoid(seg_logits[0]).detach().cpu()
            seg_bin = (seg_prob >= THRESHOLD).float().numpy().squeeze()
            seg_img = Image.fromarray((seg_bin * 255).astype(np.uint8))
            seg_img.save(SEG_ONLY_PATH / filename)

            # --- 3. HITUNG METRIK SR ---
            m['psnr']  += calculate_psnr(sr[0], hr[0])
            m['ssim']  += calculate_ssim(sr[0], hr[0])
            m['lpips'] += calculate_lpips(sr[0], hr[0])
            m['sam']   += calculate_sam(sr[0], hr[0])
            m['cc']    += calculate_cc(sr[0], hr[0])
            m['ergas'] += calculate_ergas(sr[0], hr[0], scale_factor=5)
            m['epi']   += calculate_epi(sr[0], hr[0])
            m['sdi']   += calculate_sdi(sr[0], hr[0])

            # --- 4. HITUNG METRIK SEGMENTASI (dengan total error) ---
            iou, f1, prec, rec, ae, tot_px, tot_pct = calculate_segmentation_metrics(
                seg_logits[0:1], mask[0:1], threshold=THRESHOLD, return_total_error=True
            )
            m['iou'] += iou
            m['f1']  += f1
            m['precision'] += prec
            m['recall'] += rec
            m['ae']  += ae
            m['total_error_px'] += tot_px
            m['total_error_pct'] += tot_pct

            if i % 50 == 0:
                print(f" -> Progress: {filename}")

    # Rata-rata
    for k in m:
        m[k] /= n

    # Laporan akhir (tambahkan Total Error)
    report = (
        f"================ FINAL TEST REPORT (PalmMTGAN V3d) ================\n"
        f"Waktu Uji : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Threshold : {THRESHOLD} | Total Data: {n}\n\n"
        f"--- METRIK SEGMENTASI ---\n"
        f"mIoU: {m['iou']:.4f} | F1: {m['f1']:.4f} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f}\n"
        f"Net Area Error (|FP-FN|/GT): {m['ae']:.2f}%\n"
        f"Total Error (FP+FN): {m['total_error_px']:.1f} px | {m['total_error_pct']:.2f}%\n\n"
        f"--- METRIK CITRA (SR) ---\n"
        f"PSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.4f} | LPIPS: {m['lpips']:.4f} | SAM: {m['sam']:.4f}\n"
        f"CC: {m['cc']:.4f} | ERGAS: {m['ergas']:.2f} | EPI: {m['epi']:.4f} | SDI: {m['sdi']:.4f}\n"
        f"================================================================"
    )

    print("\n" + report)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Selesai. Laporan tersimpan di: {REPORT_PATH}")

if __name__ == "__main__":
    test_palmmtgan_v3d()