# PalmMTGAN

Official implementation of **PalmMTGAN**, a multitask Generative Adversarial Network for simultaneous super-resolution and semantic segmentation of oil palm plantations, using paired Sentinel-2 and UAV imagery.

PalmMTGAN upscales Sentinel-2 imagery (10 m) by a factor of 5× to 2 m resolution while simultaneously producing an oil palm canopy segmentation map — all from a single Sentinel-2 input, **without requiring UAV imagery at inference time**.

## Architecture Overview

- **Generator**: A shared RRDB backbone (23 blocks) splits into a super-resolution branch (Spectral Preservation Module, progressive PixelShuffle upsampling) and a segmentation branch (multi-level skip connections, ASPP, Boundary-Aware Attention).
- **Discriminators**: Two independent LSGAN discriminators — a global discriminator for the SR branch and a PatchGAN discriminator for the segmentation branch.
- **Loss**: A multitask loss combining pixel-wise L1, VGG perceptual loss, adversarial losses for both branches, and a Tversky-Focal-Hole loss for segmentation.

Full architectural and training details are described in the accompanying paper (citation below).

## Repository Structure

This repository uses a flat structure — all Python files import each other directly by module name, so keep them in the same directory.

| File | Description |
|---|---|
| `models_PalmMTGAN.py` | Generator, both discriminators, and all architectural sub-modules (RRDB, ASPP, BAA, SPM). |
| `losses_PalmMTGAN.py` | All loss functions: perceptual loss, Tversky-Focal-Hole segmentation loss, LSGAN loss, feature affinity loss, and the combined multitask loss. |
| `metrics_PalmMTGAN.py` | SR metrics (PSNR, SSIM, SAM, ERGAS, CC, EPI, SDI, LPIPS) and segmentation metrics (IoU, Precision, Recall, F1, Area Error), plus visualization utilities. |
| `dataset.py` | PyTorch `Dataset` class for loading paired Sentinel-2 / UAV / mask GeoTIFF patches. |
| `Pretrain_PalmMTGAN.py` | Stage 1: generator pretraining (no adversarial component). |
| `Train_PalmMTGAN.py` | Stage 2: full adversarial training, initialized from the Stage 1 checkpoint. |
| `evaluasi_data_test_PalmMTGAN.py` | Test set evaluation script — computes all metrics and saves SR/segmentation outputs. |

> **Note**: A third training stage (short fine-tuning with a reduced learning rate) is described in the paper. It reuses `Train_PalmMTGAN.py` with a lower `--epochs` / `--lr_g` / `--lr_d` and a checkpoint from Stage 2 as `--pretrained_path`; a dedicated fine-tuning script has not been separately included here, but the same script accepts the corresponding hyperparameter set (see the paper's training configuration table).

## Installation

```bash
git clone https://github.com/WirmantoSuteddy83/PalmMTGAN.git
cd PalmMTGAN
pip install -r requirements.txt
```

Requires Python 3.9+ and a CUDA-capable GPU (training uses mixed-precision AMP; CPU inference is possible but slow).

## Dataset Structure

Scripts expect the dataset organized as follows, with each split containing three subfolders of co-registered GeoTIFF patches sharing the same filenames:

```
data/PalmSen2UAV/
├── train/
│   ├── LR_S2/       # Sentinel-2 patches, 32x32 px, 10 m resolution
│   ├── HR_UAV/      # UAV patches, 160x160 px, 2 m resolution
│   └── Mask_GT/     # Binary ground-truth canopy masks, 160x160 px
├── val/
│   ├── LR_S2/
│   ├── HR_UAV/
│   └── Mask_GT/
└── test/
    ├── LR_S2/
    ├── HR_UAV/
    └── Mask_GT/
```

The full PalmSen2UAV dataset (9,537 triplets from oil palm plantations in Jambi and East Kalimantan, Indonesia) accompanies the paper. The PalmSen2UAV dataset used in this study is available upon reasonable request. [dataset download link/DOI here once released.]

## Usage

### 1. Generator pretraining

```bash
python Pretrain_PalmMTGAN.py \
    --data_path ./data/PalmSen2UAV \
    --save_path ./checkpoints/pretrain \
    --epochs 40 --batch_size 64 --lr 1e-4
```

### 2. Adversarial training

```bash
python Train_PalmMTGAN.py \
    --data_path ./data/PalmSen2UAV \
    --save_path ./checkpoints/adversarial \
    --pretrained_path ./checkpoints/pretrain/best_pretrained_IoU.pth \
    --epochs 100 --batch_size 64
```

### 3. Test set evaluation

```bash
python evaluasi_data_test_PalmMTGAN.py \
    --model_path ./checkpoints/adversarial/best_gan_IoU.pth \
    --data_path ./data/PalmSen2UAV/test \
    --output_dir ./results/test_evaluation
```

Run any script with `--help` to see all available arguments and their default values.

## Citation

If you use this code or the PalmSen2UAV dataset in your research, please cite:

```bibtex
@article{suteddy2026palmmtgan,
  title={PalmMTGAN: Multitask GAN of Super-Resolution and Semantic Segmentation for Oil Palm Plantation Using Sentinel-2 and UAV Imagery},
  author={Suteddy, Wirmanto and Husni, Emir and Darmakusuma, Reza and Harto, Agung Budi and Rahadiansyah, Hanif A and Majid, Fauzan and Al Hadi, Ibadurahman and Ambarwari, Agus},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}
```

*(Update volume/pages/DOI once the paper is formally published.)*

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The authors thank PT Perkebunan Nusantara IV Regional 4 for providing UAV imagery of oil palm plantations in Jambi, Indonesia, and PT Terra Drone Indonesia for providing UAV imagery of oil palm plantations in East Kalimantan, Indonesia.

## Contact

For questions about this repository, please open an issue, or contact the corresponding author (Wirmanto Suteddy, wirmantosuteddy@gmail.com).
