"""
PalmMTGAN - Stage 1: Generator Pretraining
--------------------------------------------
Pretrains the PalmMTGAN generator (SPM + Focal-Dice loss) without any
adversarial component. Run this stage first; its best checkpoint is used
to initialize the adversarial training stage (see Train_PalmMTGAN.py).

Example usage:
    python Pretrain_PalmMTGAN.py \
        --data_path ./data/PalmSen2UAV \
        --save_path ./checkpoints/pretrain \
        --epochs 40 --batch_size 64 --lr 1e-4
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import random
import numpy as np
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models_PalmMTGAN import PalmMTGAN_Generator
from dataset import PalmDataset
from losses_PalmMTGAN import PerceptualLoss, PalmLoss
from metrics_PalmMTGAN import (
    calculate_psnr, calculate_lpips,
    calculate_segmentation_metrics,
    save_visuals_grid_header
)


# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="PalmMTGAN - Generator Pretraining")

    # Paths (previously hardcoded, now configurable)
    parser.add_argument('--data_path', type=str, default='./data/PalmSen2UAV',
                         help="Root dataset directory containing 'train/' and 'val/' subfolders, "
                              "each with LR_S2/, HR_UAV/, and Mask_GT/ subfolders.")
    parser.add_argument('--save_path', type=str, default='./checkpoints/pretrain',
                         help="Directory to save logs, CSV metrics, checkpoints, and visualizations.")

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seg_weight', type=float, default=3.0,
                         help="Weight for the segmentation (Focal-Dice) loss term.")
    parser.add_argument('--patience', type=int, default=10,
                         help="Early stopping patience (epochs without IoU improvement).")
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parse_args()
set_seed(args.seed)

# --- KONFIGURASI --------------------------------------------------------
DATA_PATH = args.data_path
SAVE_PATH = args.save_path
LOG_PATH = os.path.join(SAVE_PATH, "pretrain_log.txt")
CSV_PATH = os.path.join(SAVE_PATH, "pretrain_metrics.csv")
VISUAL_PATH = os.path.join(SAVE_PATH, "visuals_pretrain")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
EPOCHS = args.epochs
LR = args.lr
GRAD_CLIP = args.grad_clip
SEG_WEIGHT = args.seg_weight
PATIENCE = args.patience

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(VISUAL_PATH, exist_ok=True)


# ----------------------------------------------------------------------
def write_log(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(msg)


def main():
    write_log("PRETRAINING (SPM + Focal-Dice Loss)")
    write_log(f"Config: data_path={DATA_PATH}, save_path={SAVE_PATH}, "
              f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, seg_weight={SEG_WEIGHT}")

    netG = PalmMTGAN_Generator().to(DEVICE)
    optG = optim.AdamW(netG.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    criterion_l1 = nn.L1Loss().to(DEVICE)
    criterion_percep = PerceptualLoss().to(DEVICE)
    criterion_seg = PalmLoss()  # Focal-Dice dengan parameter default

    scheduler = ReduceLROnPlateau(optG, mode='max', factor=0.5, patience=5)

    train_dataset = PalmDataset(os.path.join(DATA_PATH, 'train'))
    val_dataset = PalmDataset(os.path.join(DATA_PATH, 'val'))
    loader_t = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    loader_v = DataLoader(val_dataset, batch_size=8, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    csv_file = open(CSV_PATH, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'loss', 'psnr', 'lpips', 'iou', 'area_error',
                         'combined_psnr_iou', 'combined_lpips_ae', 'lr'])

    best_iou = 0.0
    best_iou_epoch = 1
    early_stop_counter = 0

    for epoch in range(1, EPOCHS + 1):
        netG.train()
        pbar = tqdm(loader_t, desc=f"Pretrain Ep {epoch}/{EPOCHS}")
        train_loss = 0.0

        for lr_img, hr, mask, name in pbar:
            lr_img, hr, mask = lr_img.to(DEVICE), hr.to(DEVICE), mask.to(DEVICE)

            optG.zero_grad()
            with torch.amp.autocast('cuda'):
                sr, seg_logits, feat_sr, feat_seg = netG(lr_img)
                loss_l1 = criterion_l1(sr, hr)
                loss_percep = criterion_percep(sr, hr)
                loss_seg = criterion_seg(seg_logits, mask)
                loss_G = (1.0 * loss_l1) + (0.1 * loss_percep) + (SEG_WEIGHT * loss_seg)

            scaler.scale(loss_G).backward()
            scaler.unscale_(optG)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), GRAD_CLIP)
            scaler.step(optG)
            scaler.update()

            train_loss += loss_G.item()
            pbar.set_postfix({"Loss": f"{loss_G.item():.4f}"})

        # Validasi
        netG.eval()
        total_psnr = 0.0
        total_lpips = 0.0
        total_iou = 0.0
        total_ae = 0.0
        total_samples = 0

        sample_metrics = []

        with torch.no_grad():
            for lr_v, hr_v, mask_v, name_v in loader_v:
                lr_v, hr_v, mask_v = lr_v.to(DEVICE), hr_v.to(DEVICE), mask_v.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    sr_v, seg_v, _, _ = netG(lr_v)

                B = sr_v.size(0)
                for j in range(B):
                    sr_j = sr_v[j:j+1]
                    hr_j = hr_v[j:j+1]
                    seg_j = seg_v[j:j+1]
                    mask_j = mask_v[j:j+1]

                    total_psnr += calculate_psnr(sr_j[0], hr_j[0])
                    total_lpips += calculate_lpips(sr_j[0], hr_j[0])
                    iou, _, _, _, ae = calculate_segmentation_metrics(seg_j, mask_j)
                    total_iou += iou
                    total_ae += ae

                    sample_metrics.append((iou,
                                           lr_v[j:j+1].cpu(),
                                           hr_v[j:j+1].cpu(),
                                           sr_j.cpu(),
                                           mask_j.cpu(),
                                           seg_j.cpu(),
                                           name_v[j]))
                total_samples += B

        avg_psnr = total_psnr / total_samples
        avg_lpips = total_lpips / total_samples
        avg_iou = total_iou / total_samples
        avg_ae = total_ae / total_samples

        combined_psnr_iou = (avg_psnr / 35.0) + avg_iou
        combined_lpips_ae = avg_lpips * 100 + avg_ae

        log_msg = (f"Ep {epoch} | Loss: {train_loss/len(loader_t):.4f} | "
                   f"PSNR: {avg_psnr:.2f} | LPIPS: {avg_lpips:.4f} | "
                   f"IoU: {avg_iou:.4f} | AE: {avg_ae:.2f}% | "
                   f"Sc_PSNR+IoU: {combined_psnr_iou:.4f} | Sc_LPIPS+AE: {combined_lpips_ae:.4f} | "
                   f"LR: {optG.param_groups[0]['lr']:.2e}")
        write_log(log_msg)

        csv_writer.writerow([epoch, train_loss/len(loader_t), avg_psnr, avg_lpips,
                             avg_iou, avg_ae, combined_psnr_iou, combined_lpips_ae,
                             optG.param_groups[0]['lr']])
        csv_file.flush()

        # Visualisasi terbaik & terburuk
        sample_metrics.sort(key=lambda x: x[0])
        worst = sample_metrics[0]
        best = sample_metrics[-1]

        for tag, (iou_val, lr_img_v, hr_img, sr_img, mask_img, seg_img, name_img) in [
                ("worst", worst), ("best", best)]:
            save_visuals_grid_header(
                lr_img_v.to(DEVICE), hr_img.to(DEVICE), sr_img.to(DEVICE),
                mask_img.to(DEVICE), seg_img.to(DEVICE),
                f"{tag}_ep{epoch}_{name_img}",
                VISUAL_PATH
            )

        scheduler.step(avg_iou)

        if avg_iou > best_iou:
            best_iou = avg_iou
            best_iou_epoch = epoch
            torch.save(netG.state_dict(), os.path.join(SAVE_PATH, 'best_pretrained_IoU.pth'))
            write_log(f"New best IoU: {best_iou:.4f} -> checkpoint saved")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            write_log(f"Early stopping after {PATIENCE} epochs without IoU improvement.")
            break

    csv_file.close()
    write_log(f"PRETRAINING DONE. Best IoU: {best_iou:.4f} at epoch {best_iou_epoch}")


if __name__ == "__main__":
    try:
        main()
        print("\n[SUCCESS] Pretraining completed.")
    except Exception as e:
        print(f"\n[ERROR] Pretraining stopped due to: {e}")
