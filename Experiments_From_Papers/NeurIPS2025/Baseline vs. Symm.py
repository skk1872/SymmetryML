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
import matplotlib.gridspec as gridspec

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


def train_and_get_outputs(lambda_sym, lambda_consist, device, train_loader, test_loader, kernel_x, kernel_y, angles, fill_val):
    model = SmallConvMNIST().to(device)
    t = train_loader.dataset.targets
    pos, neg = (t==7).sum().item(), (t==8).sum().item()
    pos_weight = torch.tensor(neg/pos, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler    = torch.cuda.amp.GradScaler()

    model.train()
    for _ in range(5):
        for imgs, lbls in train_loader:
            imgs = imgs.to(device); imgs.requires_grad_(True)
            lbl_bin = (lbls==7).float().unsqueeze(1).to(device)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, lbl_bin)

                 only add symmetry/consistency if lambdas > 0
                if lambda_sym>0:
                    grad_f = torch.autograd.grad(logits, imgs, grad_outputs=torch.ones_like(logits), create_graph=True)[0]
                    phi = random.random()*2*math.pi
                    dR  = F.conv2d(imgs, kernel_x, padding=(0,1))*math.cos(phi) \
                        + F.conv2d(imgs, kernel_y, padding=(1,0))*math.sin(phi)
                    loss = loss + lambda_sym*(grad_f * dR).sum(dim=[1,2,3]).pow(2).mean()
                if lambda_consist>0:
                    ang = random.choice(angles)
                    rot = TF.rotate(imgs, ang, fill=(fill_val,))
                    loss = loss + lambda_consist*F.mse_loss(model(rot), logits)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    with torch.no_grad():
        for im, lb in test_loader:
            if lb.item()==7:
                s7 = im.to(device); break
        for im, lb in test_loader:
            if lb.item()==8:
                s8 = im.to(device); break

    p7_curve = []
    p8_curve = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for a in angles:
            p7_curve.append(torch.sigmoid(model(TF.rotate(s7,a,fill=(fill_val,)))).item())
            p8_curve.append((1- torch.sigmoid(model(TF.rotate(s8,a,fill=(fill_val,))))).item())

    agg7, agg8 = [[] for _ in angles], [[] for _ in angles]
    with torch.no_grad(), torch.cuda.amp.autocast():
        for im, lb in test_loader:
            im = im.to(device)
            for i,a in enumerate(angles):
                p = torch.sigmoid(model(TF.rotate(im,a,fill=(fill_val,)))).item()
                if lb.item()==7:
                    agg7[i].append(p)
                else:
                    agg8[i].append(1-p)
    med7 = [np.median(v) for v in agg7]
    q1_7 = [np.percentile(v,25) for v in agg7]
    q3_7 = [np.percentile(v,75) for v in agg7]
    med8 = [np.median(v) for v in agg8]
    q1_8 = [np.percentile(v,25) for v in agg8]
    q3_8 = [np.percentile(v,75) for v in agg8]

    return (s7, s8, p7_curve, p8_curve, med7, q1_7, q3_7, med8, q1_8, q3_8)


def main():
     data prep (filter to 7 & 8)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    mask = (train_ds.targets==7)|(train_ds.targets==8)
    train_ds.data, train_ds.targets = train_ds.data[mask], train_ds.targets[mask]
    mask = (test_ds.targets==7)|(test_ds.targets==8)
    test_ds.data, test_ds.targets = test_ds.data[mask], test_ds.targets[mask]

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=1,   shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel_x = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=device)
    kernel_y = torch.tensor([[[[-0.5],[0.0],[0.5]]]], device=device)
    angles   = [0,45,90,135,180,225,270,315]
    fill_val = (0.0 - 0.1307)/0.3081

    base = train_and_get_outputs(0.0, 0.0, device, train_loader, test_loader, kernel_x, kernel_y, angles, fill_val)
     symmetry model
    sym  = train_and_get_outputs(0.5, 1.0, device, train_loader, test_loader, kernel_x, kernel_y, angles, fill_val)

    s7b, s8b, p7b, p8b, m7b, q17b, q37b, m8b, q18b, q38b = base
    s7s, s8s, p7s, p8s, m7s, q17s, q37s, m8s, q18s, q38s = sym

    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,4], hspace=0.3)
    ax7 = fig.add_subplot(gs[0,0]); ax8 = fig.add_subplot(gs[0,1]); axR = fig.add_subplot(gs[1,:])
    ax7.imshow(((s7s.cpu()*0.3081)+0.1307).squeeze(), cmap='gray'); ax7.axis('off'); ax7.set_title("Sample '7'")
    ax8.imshow(((s8s.cpu()*0.3081)+0.1307).squeeze(), cmap='gray'); ax8.axis('off'); ax8.set_title("Sample '8'")
    axR.plot(angles, p7b, '--o', label="P(7) on '7' (baseline)")
    axR.plot(angles, p8b, '--o', label="P(8) on '8' (baseline)")
    axR.plot(angles, p7s,  '-o', label="P(7) on '7' (sym)")
    axR.plot(angles, p8s,  '-o', label="P(8) on '8' (sym)")
    axR.set_xlabel("Rotation Angle (°)"); axR.set_ylabel("Probability")
    axR.set_title("Rotation Invariance: Baseline vs Symmetry")
    axR.legend(); axR.grid(True)
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,4))
     baseline
    plt.errorbar(angles, m7b, yerr=[np.subtract(m7b,q17b), np.subtract(q37b,m7b)],
                 fmt='--o', capsize=4, label="P(7) med±IQR (baseline)")
    plt.errorbar(angles, m8b, yerr=[np.subtract(m8b,q18b), np.subtract(q38b,m8b)],
                 fmt='--o', capsize=4, label="P(8) med±IQR (baseline)")
     symmetry
    plt.errorbar(angles, m7s, yerr=[np.subtract(m7s,q17s), np.subtract(q37s,m7s)],
                 fmt='-o', capsize=4, label="P(7) med±IQR (sym)")
    plt.errorbar(angles, m8s, yerr=[np.subtract(m8s,q18s), np.subtract(q38s,m8s)],
                 fmt='-o', capsize=4, label="P(8) med±IQR (sym)")

    plt.xlabel("Rotation Angle (°)"); plt.ylabel("Probability")
    plt.title("Angle-wise Median ± IQR: Baseline vs Symmetry")
    plt.ylim(0,1)
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    main()
