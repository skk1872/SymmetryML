import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import random, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

class SmallConvMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*32, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

def train_and_get_outputs(lambda_field, device, train_loader, test_loader, kernel_x, kernel_y, Vx, Vy, angles, fill_val):
    model = SmallConvMNIST().to(device)
    t = train_loader.dataset.targets
    pos, neg = (t == 7).sum().item(), (t == 8).sum().item()
    pos_weight = torch.tensor(neg / pos, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for _ in range(5):
        for imgs, lbls in train_loader:
            imgs = imgs.to(device); imgs.requires_grad_(True)
            lbl_bin = (lbls == 7).float().unsqueeze(1).to(device)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, lbl_bin)
                if lambda_field > 0:
                    Dx = F.conv2d(imgs, kernel_x, padding=(0,1))
                    Dy = F.conv2d(imgs, kernel_y, padding=(1,0))
                    dR = Vx * Dx + Vy * Dy
                    loss = loss + lambda_field * dR.pow(2).mean()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    with torch.no_grad():
        for im, lb in test_loader:
            if lb.item() == 7:
                s7 = im.to(device); break
        for im, lb in test_loader:
            if lb.item() == 8:
                s8 = im.to(device); break

    p7_curve = [torch.sigmoid(model(TF.rotate(s7, a, fill=(fill_val,)))).item() for a in angles]
    p8_curve = [1 - torch.sigmoid(model(TF.rotate(s8, a, fill=(fill_val,)))).item() for a in angles]

    agg7 = [[] for _ in angles]
    agg8 = [[] for _ in angles]
    with torch.no_grad():
        for im, lb in test_loader:
            im = im.to(device)
            for i, a in enumerate(angles):
                p = torch.sigmoid(model(TF.rotate(im, a, fill=(fill_val,)))).item()
                if lb.item() == 7:
                    agg7[i].append(p)
                else:
                    agg8[i].append(1 - p)

    med7 = [np.median(v) for v in agg7]
    q1_7 = [np.percentile(v, 25) for v in agg7]
    q3_7 = [np.percentile(v, 75) for v in agg7]
    med8 = [np.median(v) for v in agg8]
    q1_8 = [np.percentile(v, 25) for v in agg8]
    q3_8 = [np.percentile(v, 75) for v in agg8]

    return s7, s8, p7_curve, p8_curve, med7, q1_7, q3_7, med8, q1_8, q3_8

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    mask = (train_ds.targets == 7) | (train_ds.targets == 8)
    train_ds.data, train_ds.targets = train_ds.data[mask], train_ds.targets[mask]
    mask = (test_ds.targets == 7) | (test_ds.targets == 8)
    test_ds.data, test_ds.targets = test_ds.data[mask], test_ds.targets[mask]

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel_x = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=device)
    kernel_y = torch.tensor([[[[-0.5],[0.0],[0.5]]]], device=device)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    fill_val = (0.0 - 0.1307) / 0.3081

    H, W = 28, 28
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')

    Vx7 = (X / torch.sqrt(X**2 + Y**2 + 1e-8))[None,None].to(device)
    Vy7 = (Y / torch.sqrt(X**2 + Y**2 + 1e-8))[None,None].to(device)
    Vx8 = (-Y / torch.sqrt(X**2 + Y**2 + 1e-8))[None,None].to(device)
    Vy8 = (X / torch.sqrt(X**2 + Y**2 + 1e-8))[None,None].to(device)

    base = train_and_get_outputs(0.0, device, train_loader, test_loader, kernel_x, kernel_y, Vx7, Vy7, angles, fill_val)
    out7 = train_and_get_outputs(1.0, device, train_loader, test_loader, kernel_x, kernel_y, Vx7, Vy7, angles, fill_val)
    out8 = train_and_get_outputs(1.0, device, train_loader, test_loader, kernel_x, kernel_y, Vx8, Vy8, angles, fill_val)

    s7b, s8b, p7b, p8b, m7b, q17b, q37b, m8b, q18b, q38b = base
    s7_7, s8_7, p7_7, p8_7, m7_7, q17_7, q37_7, m8_7, q18_7, q38_7 = out7
    s7_8, s8_8, p7_8, p8_8, m7_8, q17_8, q37_8, m8_8, q18_8, q38_8 = out8

    plt.figure(figsize=(6,4))
    plt.plot(angles, p7b, '--o', label="Baseline P(7) on '7'")
    plt.plot(angles, p7_7, '-o', label="Radial-inv P(7) on '7'")
    plt.errorbar(angles, m7b, yerr=[np.subtract(m7b, q17b), np.subtract(q37b, m7b)], fmt='--', capsize=4, linestyle='none')
    plt.errorbar(angles, m7_7, yerr=[np.subtract(m7_7, q17_7), np.subtract(q37_7, m7_7)], fmt='-', capsize=4, linestyle='none')
    plt.xlabel("Rotation Angle (°)")
    plt.ylabel("P(model predicts 7)")
    plt.title("Digit ‘7’: Baseline vs Radial-Invariant")
    plt.ylim(0,1)
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(angles, p8b, '--o', label="Baseline P(8) on '8'")
    plt.plot(angles, p8_8, '-o', label="Tangential-inv P(8) on '8'")
    plt.errorbar(angles, m8b, yerr=[np.subtract(m8b, q18b), np.subtract(q38b, m8b)], fmt='--', capsize=4, linestyle='none')
    plt.errorbar(angles, m8_8, yerr=[np.subtract(m8_8, q18_8), np.subtract(q38_8, m8_8)], fmt='-', capsize=4, linestyle='none')
    plt.xlabel("Rotation Angle (°)")
    plt.ylabel("P(model predicts 8)")
    plt.title("Digit ‘8’: Baseline vs Tangential-Invariant")
    plt.ylim(0,1)
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    main()
