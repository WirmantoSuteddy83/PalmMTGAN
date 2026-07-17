import os
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import random
import numpy as np
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import itertools

from models_PalmMTGAN import (
    PalmMTGAN_Generator,
    Discriminator_SR,
    Discriminator_Seg
)
from dataset import PalmDataset
from losses_PalmMTGAN import (
    PalmMTGANLoss,
    discriminator_sr_loss,
    discriminator_seg_loss
)
from metrics_PalmMTGAN import (
    evaluate_batch,
    save_visuals_grid_header
)

# ----------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- KONFIGURASI --------------------------------------------------------
DATA_PATH = "/content/dataset/Dataset_PalmSen2UAV_9537data"
SAVE_PATH = "/content/drive/MyDrive/S3/GAN/Eksperimen_PalmMTGAN_V2"
LOG_PATH = os.path.join(SAVE_PATH, "gan_log.txt")
VISUAL_PATH = os.path.join(SAVE_PATH, "Visuals_gan")
CSV_PATH = os.path.join(SAVE_PATH, "gan_metrics.csv")
PRETRAINED_GEN = os.path.join(SAVE_PATH, "best_pretrained_IoU.pth")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 100
LR_G = 1e-4
LR_D = 4e-4
GRAD_CLIP = 1.0
PATIENCE = 15

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(VISUAL_PATH, exist_ok=True)

# ----------------------------------------------------------------------
def write_log(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(msg)

def main():
    write_log("🎯 MEMULAI PELATIHAN GAN (PalmMTGAN) – Full Metrics + Early Stop + Cosine")

    netG = PalmMTGAN_Generator().to(DEVICE)
    netD_SR = Discriminator_SR().to(DEVICE)
    netD_Seg = Discriminator_Seg().to(DEVICE)

    if os.path.exists(PRETRAINED_GEN):
        netG.load_state_dict(torch.load(PRETRAINED_GEN, map_location=DEVICE))
        write_log(f"✅ Generator dimuat dari {PRETRAINED_GEN}")
    else:
        write_log("⚠️ Pretrained generator tidak ditemukan, lanjut tanpa inisialisasi")

    criterion_gan = PalmMTGANLoss(
        use_uncertainty=True,
        l1_weight=1.0,
        perceptual_weight=0.1,
        gan_sr_weight=0.05,
        gan_seg_weight=0.05,
        seg_weight=2.0,
        affinity_weight=0.0
    ).to(DEVICE)

    optG = optim.Adam(
        itertools.chain(netG.parameters(), criterion_gan.parameters()),
        lr=LR_G, betas=(0.9, 0.999)
    )
    optD_SR = optim.Adam(netD_SR.parameters(), lr=LR_D, betas=(0.9, 0.999))
    optD_Seg = optim.Adam(netD_Seg.parameters(), lr=LR_D, betas=(0.9, 0.999))

    scheduler_G = CosineAnnealingLR(optG, T_max=EPOCHS, eta_min=1e-6)
    scheduler_D_SR = CosineAnnealingLR(optD_SR, T_max=EPOCHS, eta_min=1e-6)
    scheduler_D_Seg = CosineAnnealingLR(optD_Seg, T_max=EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda')

    train_dataset = PalmDataset(os.path.join(DATA_PATH, 'train'))
    val_dataset = PalmDataset(os.path.join(DATA_PATH, 'val'))
    loader_t = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    loader_v = DataLoader(val_dataset, batch_size=8, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # CSV setup
    csv_file = open(CSV_PATH, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_header = ['epoch', 'loss_G', 'loss_D_SR', 'loss_D_Seg',
                  'PSNR', 'SSIM', 'SAM', 'CC', 'ERGAS', 'EPI', 'SDI', 'LPIPS',
                  'IoU', 'F1_Score', 'Precision', 'Recall', 'Area_Error',
                  'log_sigma_l1', 'log_sigma_percep', 'log_sigma_gan_sr',
                  'log_sigma_seg', 'log_sigma_gan_seg', 'log_sigma_affinity']
    csv_writer.writerow(csv_header)

    # Best trackers
    best_iou = 0.0
    best_psnr_iou = 0.0
    best_lpips_ae = float('inf')
    early_stop_counter = 0

    for epoch in range(1, EPOCHS + 1):
        netG.train()
        netD_SR.train()
        netD_Seg.train()
        epoch_loss_G = 0.0
        epoch_loss_D_SR = 0.0
        epoch_loss_D_Seg = 0.0

        pbar = tqdm(loader_t, desc=f"GAN Ep {epoch}/{EPOCHS}")
        for lr, hr, mask, name in pbar:
            lr, hr, mask = lr.to(DEVICE), hr.to(DEVICE), mask.to(DEVICE)

            # ================== DISKRIMINATOR SR ==================
            optD_SR.zero_grad()
            with torch.amp.autocast('cuda'):
                sr_fake, _, _, _ = netG(lr)
                loss_d_sr = discriminator_sr_loss(netD_SR, hr, sr_fake)
            scaler.scale(loss_d_sr).backward()
            scaler.unscale_(optD_SR)
            torch.nn.utils.clip_grad_norm_(netD_SR.parameters(), GRAD_CLIP)
            scaler.step(optD_SR)

            # ================== DISKRIMINATOR SEGMENTASI ==================
            optD_Seg.zero_grad()
            with torch.amp.autocast('cuda'):
                sr_fake, seg_logits, _, _ = netG(lr)
                seg_fake_prob = torch.sigmoid(seg_logits)
                loss_d_seg = discriminator_seg_loss(netD_Seg, mask, seg_fake_prob)
            scaler.scale(loss_d_seg).backward()
            scaler.unscale_(optD_Seg)
            torch.nn.utils.clip_grad_norm_(netD_Seg.parameters(), GRAD_CLIP)
            scaler.step(optD_Seg)

            # ================== GENERATOR ==================
            optG.zero_grad()
            with torch.amp.autocast('cuda'):
                sr_fake, seg_logits, feat_sr, feat_seg = netG(lr)
                disc_sr_fake = netD_SR(sr_fake)
                disc_seg_fake = netD_Seg(torch.sigmoid(seg_logits))
                total_loss_g, _ = criterion_gan(
                    sr_fake, hr,
                    seg_logits, mask,
                    disc_sr_fake, disc_seg_fake,
                    feat_sr, feat_seg
                )

            scaler.scale(total_loss_g).backward()
            scaler.unscale_(optG)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(criterion_gan.parameters(), GRAD_CLIP)
            scaler.step(optG)
            scaler.update()

            epoch_loss_G += total_loss_g.item()
            epoch_loss_D_SR += loss_d_sr.item()
            epoch_loss_D_Seg += loss_d_seg.item()
            pbar.set_postfix({
                "G": f"{total_loss_g.item():.3f}",
                "D_SR": f"{loss_d_sr.item():.3f}",
                "D_Seg": f"{loss_d_seg.item():.3f}"
            })

        # ================== VALIDASI (METRIK LENGKAP) ==================
        netG.eval()
        # Kumpulkan semua prediksi untuk evaluasi batch
        all_metrics_sum = None
        total_batches = 0
        with torch.no_grad():
            for lr_v, hr_v, mask_v, name_v in loader_v:
                lr_v, hr_v, mask_v = lr_v.to(DEVICE), hr_v.to(DEVICE), mask_v.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    sr_v, seg_v, _, _ = netG(lr_v)
                # evaluate_batch menghitung rata-rata per batch
                batch_metrics = evaluate_batch(sr_v, hr_v, seg_v, mask_v)
                if all_metrics_sum is None:
                    all_metrics_sum = {k: 0.0 for k in batch_metrics}
                for k in batch_metrics:
                    all_metrics_sum[k] += batch_metrics[k]
                total_batches += 1

        # Rata-rata seluruh validasi
        avg_metrics = {k: v / total_batches for k, v in all_metrics_sum.items()}

        # Log ke file dan CSV
        log_msg = (f"Ep {epoch} | IoU: {avg_metrics['IoU']:.4f} | F1: {avg_metrics['F1_Score']:.4f} | "
                   f"AE: {avg_metrics['Area_Error']:.2f}% | LPIPS: {avg_metrics['LPIPS']:.4f} | "
                   f"PSNR: {avg_metrics['PSNR']:.2f}")
        write_log(log_msg)

        # Siapkan data CSV
        csv_row = [epoch,
                   epoch_loss_G/len(loader_t),
                   epoch_loss_D_SR/len(loader_t),
                   epoch_loss_D_Seg/len(loader_t),
                   avg_metrics['PSNR'], avg_metrics['SSIM'], avg_metrics['SAM'],
                   avg_metrics['CC'], avg_metrics['ERGAS'], avg_metrics['EPI'],
                   avg_metrics['SDI'], avg_metrics['LPIPS'],
                   avg_metrics['IoU'], avg_metrics['F1_Score'], avg_metrics['Precision'],
                   avg_metrics['Recall'], avg_metrics['Area_Error']]

        # Tambahkan log_sigma
        if criterion_gan.use_uncertainty:
            csv_row += [
                criterion_gan.log_sigma_l1.item(),
                criterion_gan.log_sigma_percep.item(),
                criterion_gan.log_sigma_gan_sr.item(),
                criterion_gan.log_sigma_seg.item(),
                criterion_gan.log_sigma_gan_seg.item(),
                criterion_gan.log_sigma_affinity.item()
            ]
        else:
            csv_row += [0]*6
        csv_writer.writerow(csv_row)
        csv_file.flush()

        # --- Simpan model terbaik ---
        current_iou = avg_metrics['IoU']
        if current_iou > best_iou:
            best_iou = current_iou
            torch.save(netG.state_dict(), os.path.join(SAVE_PATH, 'best_gan_IoU.pth'))
            write_log(f"⭐ Rekor IoU: {best_iou:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        combined_psnr_iou = (avg_metrics['PSNR'] / 35.0) + current_iou
        if combined_psnr_iou > best_psnr_iou:
            best_psnr_iou = combined_psnr_iou
            torch.save(netG.state_dict(), os.path.join(SAVE_PATH, 'best_gan_PSNR_IoU.pth'))

        combined_lpips_ae = avg_metrics['LPIPS'] * 100 + avg_metrics['Area_Error']
        if combined_lpips_ae < best_lpips_ae:
            best_lpips_ae = combined_lpips_ae
            torch.save(netG.state_dict(), os.path.join(SAVE_PATH, 'best_gan_LPIPS_AE.pth'))

        # Early stopping
        if early_stop_counter >= PATIENCE:
            write_log(f"⏹️ Early stopping setelah {PATIENCE} epoch tanpa peningkatan IoU.")
            break

        # Scheduler step
        scheduler_G.step()
        scheduler_D_SR.step()
        scheduler_D_Seg.step()

        # --- Visualisasi acak ---
        rand_idx = random.randint(0, len(val_dataset) - 1)
        lr_vis, hr_vis, mask_vis, name_vis = val_dataset[rand_idx]
        lr_vis = lr_vis.unsqueeze(0).to(DEVICE)
        hr_vis = hr_vis.unsqueeze(0).to(DEVICE)
        mask_vis = mask_vis.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                sr_vis, seg_vis, _, _ = netG(lr_vis)
        save_visuals_grid_header(lr_vis, hr_vis, sr_vis, mask_vis, seg_vis, name_vis, VISUAL_PATH)

    csv_file.close()
    write_log("✅ PELATIHAN GAN SELESAI.")

if __name__ == "__main__":
    try:
        main()
        print("\n[SUCCESS] Pelatihan GAN selesai.")
    except Exception as e:
        print(f"\n[ERROR] Pelatihan GAN terhenti karena: {e}")
    finally:
        try:
            from google.colab import runtime
            runtime.unassign()
        except ImportError:
            pass