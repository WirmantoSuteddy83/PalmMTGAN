import os, torch, rasterio
import numpy as np
from torch.utils.data import Dataset

class PalmDataset(Dataset):
    def __init__(self, root_dir):
        self.lr_dir = os.path.join(root_dir, 'LR_S2')
        self.hr_dir = os.path.join(root_dir, 'HR_UAV')
        self.mask_dir = os.path.join(root_dir, 'Mask_GT')
        self.files = sorted([f for f in os.listdir(self.lr_dir) if f.endswith('.tif')])

    def __len__(self): return len(self.files)

    def load_raster(self, path, is_mask=False):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
            if is_mask: return torch.from_numpy((img[0:1] > 0).astype(np.float32))
            else:
                v_max = img.max()
                if v_max > 1.0: img /= 255.0 if v_max <= 255.0 else 10000.0
                return torch.from_numpy(np.clip(img[[2, 1, 0], :, :], 0, 1))

    def __getitem__(self, idx):
        name = self.files[idx]
        lr = self.load_raster(os.path.join(self.lr_dir, name))
        hr = self.load_raster(os.path.join(self.hr_dir, name))
        mask = self.load_raster(os.path.join(self.mask_dir, name), True)
        # PASTIKAN bagian ini mengembalikan 4 nilai
        return lr, hr, mask, name