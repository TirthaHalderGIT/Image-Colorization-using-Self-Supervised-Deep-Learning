# =====================================================
# ALL IMPORTS (must be at top)
# =====================================================

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
import pathlib  # for PosixPath monkey-patch


# =====================================================
# USER SETTINGS
# =====================================================

CHECKPOINT_PATH = "colorization_sdae_unet_best_50ep.pth"
INPUT_IMAGE_PATH = "inputs/old.png"
OUTPUT_DIR = "outputs"
DEVICE = "cpu"    # or "cuda" if you have GPU

# =====================================================
# MODEL DEFINITIONS
# =====================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ColorizationSDAEUNet(nn.Module):
    def __init__(self):
        super().__init__()

        base = 64

        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.up4 = UpBlock(base * 16, base * 8)
        self.up3 = UpBlock(base * 8, base * 4)
        self.up2 = UpBlock(base * 4, base * 2)
        self.up1 = UpBlock(base * 2, base)

        self.out_conv = nn.Conv2d(base, 2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2); p3 = self.pool(e3)
        e4 = self.enc4(p3); p4 = self.pool(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.out_conv(d1)


# =====================================================
# IMAGE PROCESSING
# =====================================================

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256))

    bw_pil = img.convert("L")

    img_np = np.array(img).astype(np.float32) / 255.0
    lab = color.rgb2lab(img_np)

    L = lab[:, :, 0]               # [0, 100]
    L_norm = (L / 50.0) - 1.0      # [-1, 1]

    L_tensor = torch.tensor(L_norm).unsqueeze(0).unsqueeze(0).float()

    return L_tensor, L, bw_pil


def postprocess_and_save(L_original, pred_ab, bw_pil, output_dir, base_name):
    os.makedirs(output_dir, exist_ok=True)

    ab = pred_ab * 128.0

    lab_out = np.zeros((256, 256, 3), dtype=np.float32)
    lab_out[:, :, 0] = L_original
    lab_out[:, :, 1:] = ab

    rgb = color.lab2rgb(lab_out)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    color_pil = Image.fromarray(rgb_uint8)

    color_out = os.path.join(output_dir, f"{base_name}_colorized.png")
    color_pil.save(color_out)

    bw_np = np.array(bw_pil)
    bw_3ch = np.stack([bw_np]*3, axis=-1)
    side = np.concatenate([bw_3ch, rgb_uint8], axis=1)
    side_pil = Image.fromarray(side)

    side_out = os.path.join(output_dir, f"{base_name}_side_by_side.png")
    side_pil.save(side_out)

    plt.imshow(side)
    plt.axis("off")
    plt.show()

    print("\nSaved:")
    print(color_out)
    print(side_out)


# =====================================================
# MAIN (with PosixPath monkey-patch for Windows)
# =====================================================

def main():

    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # ---- MONKEY-PATCH PosixPath SO UNPICKLE WORKS ON WINDOWS ----
    # Kaggle saved cfg with pathlib.PosixPath, which cannot be instantiated on Windows.
    # We replace pathlib.PosixPath with a dummy subclass of PurePosixPath that IS allowed.
    class CompatiblePosixPath(pathlib.PurePosixPath):
        """Dummy PosixPath replacement for loading Linux checkpoints on Windows."""
        pass

    pathlib.PosixPath = CompatiblePosixPath
    # ------------------------------------------------------------

    # load model
    model = ColorizationSDAEUNet().to(device)
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print("Model loaded.")

    # process image
    L_tensor, L_original, bw_pil = preprocess_image(INPUT_IMAGE_PATH)
    L_tensor = L_tensor.to(device)

    with torch.no_grad():
        pred_ab = model(L_tensor)[0].permute(1, 2, 0).cpu().numpy()

    base_name = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]

    postprocess_and_save(L_original, pred_ab, bw_pil, OUTPUT_DIR, base_name)


# =====================================================
# RUN PROGRAM
# =====================================================

if __name__ == "__main__":
    main()
