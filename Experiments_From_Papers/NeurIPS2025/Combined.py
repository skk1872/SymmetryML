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


def main():
    torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

    mask = (train_ds.targets == 7) | (train_ds.targets == 8)
    train_ds.data, train_ds.targets = train_ds.data[mask], train_ds.targets[mask]
    mask = (test_ds.targets  == 7) | (test_ds.targets  == 8)
    test_ds.data,  test_ds.targets  = test_ds.data[mask],  test_ds.targets[mask]

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=1,   shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = train_ds.targets
    pos, neg = (t==7).sum().item(), (t==8).sum().item()
    pos_weight = torch.tensor(neg/pos, device=device)

    model     = SmallConvMNIST().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler    = torch.cuda.amp.GradScaler()

    kernel_x = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=device)
    kernel_y = torch.tensor([[[[-0.5], [0.0], [0.5]]]], device=device)

    epochs = 5
    lambda_sym, lambda_consist = 0.5, 1.0
    angles = [0,45,90,135,180,225,270,315]
    fill_val = (0.0 - 0.1307) / 0.3081

    model.train()
    for epoch in range(epochs):
        agg = {"cls":0., "sym":0., "cons":0.}
        for imgs, lbls in train_loader:
            imgs = imgs.to(device); imgs.requires_grad_(True)
            lbl_bin = (lbls==7).float().unsqueeze(1).to(device)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss_c = criterion(logits, lbl_bin)
                grad_f = torch.autograd.grad(logits, imgs, grad_outputs=torch.ones_like(logits), create_graph=True)[0]
                gx = F.conv2d(imgs, kernel_x, padding=(0,1)); gy = F.conv2d(imgs, kernel_y, padding=(1,0))
                phi = random.random()*2*math.pi
                dR  = gx*math.cos(phi) + gy*math.sin(phi)
                loss_s = (grad_f * dR).sum(dim=[1,2,3]).pow(2).mean()
                ang = random.choice(angles)
                rot = TF.rotate(imgs, ang, fill=(fill_val,))
                loss_cons = F.mse_loss(model(rot), logits)
                loss = loss_c + lambda_sym*loss_s + lambda_consist*loss_cons
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            agg["cls"]+=loss_c.item(); agg["sym"]+=loss_s.item(); agg["cons"]+=loss_cons.item()
        n=len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | class={agg['cls']/n:.4f} sym={agg['sym']/n:.4f} cons={agg['cons']/n:.4f}")

    model.eval()
    for im, lb in test_loader:
        if lb.item()==7:
            s7 = im.to(device); break
    for im, lb in test_loader:
        if lb.item()==8:
            s8 = im.to(device); break

    p7_curve, p8_curve = [], []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for a in angles:
            p7_curve.append(torch.sigmoid(model(TF.rotate(s7,a,fill=(fill_val,)))).item())
            p8_curve.append((1-torch.sigmoid(model(TF.rotate(s8,a,fill=(fill_val,))))).item())

    agg7, agg8 = [[] for _ in angles], [[] for _ in angles]
    with torch.no_grad(), torch.cuda.amp.autocast():
        for im, lb in test_loader:
            im = im.to(device)
            for i,a in enumerate(angles):
                p = torch.sigmoid(model(TF.rotate(im,a,fill=(fill_val,)))).item()
                if lb.item()==7: agg7[i].append(p)
                else:         agg8[i].append(1-p)
    means7 = [np.mean(v) for v in agg7]; stds7 = [np.std(v) for v in agg7]
    means8 = [np.mean(v) for v in agg8]; stds8 = [np.std(v) for v in agg8]

    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,4], hspace=0.3)
    ax_img7 = fig.add_subplot(gs[0, 0]); ax_img8 = fig.add_subplot(gs[0, 1]); ax_curve = fig.add_subplot(gs[1, :])
    ax_img7.imshow(((s7.cpu()*0.3081)+0.1307).squeeze(), cmap='gray'); ax_img7.axis('off'); ax_img7.set_title("Sample '7'")
    ax_img8.imshow(((s8.cpu()*0.3081)+0.1307).squeeze(), cmap='gray'); ax_img8.axis('off'); ax_img8.set_title("Sample '8'")
    ax_curve.plot(angles, p7_curve, '-o', label="P(7) on '7'"); ax_curve.plot(angles, p8_curve, '-o', label="P(8) on '8'")
    ax_curve.set_xlabel("Rotation Angle (°)"); ax_curve.set_ylabel("Probability")
    ax_curve.set_title("Rotation Invariance"); ax_curve.legend(); ax_curve.grid(True)
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,4))
    plt.errorbar(angles, means7, yerr=stds7, fmt='-o', capsize=5, label="P(7) mean±std")
    plt.errorbar(angles, means8, yerr=stds8, fmt='-o', capsize=5, label="P(8) mean±std")
    plt.xlabel("Rotation Angle (°)"); plt.ylabel("Probability")
    plt.title("Angle-wise Average ± Std")
    plt.ylim(0, 1)
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

if __name__=='__main__':
    main()
