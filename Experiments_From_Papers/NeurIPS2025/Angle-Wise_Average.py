import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import random, math
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
            nn.Conv2d(1, 16, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*32, 64),  nn.ReLU(),
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

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    targets = train_ds.targets
    pos = (targets == 7).sum().item()
    neg = (targets != 7).sum().item()
    pos_weight = torch.tensor(neg/pos, device=device)

    
    model     = SmallConvMNIST().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler    = torch.cuda.amp.GradScaler()


    kernel_x = torch.tensor([[[[-0.5,0.0,0.5]]]], device=device)
    kernel_y = torch.tensor([[[[-0.5],[0.0],[0.5]]]], device=device)


    epochs         = 5
    lambda_sym     = 0.5
    lambda_consist = 1.0
    K              = 1
    angles         = [0,45,90,135,180,225,270,315]
    fill_val       = (0.0 - 0.1307) / 0.3081

    
    model.train()
    for epoch in range(epochs):
        agg = {"cls":0.0, "sym":0.0, "cons":0.0}
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            images.requires_grad_(True)
            labels_bin = (labels==7).float().unsqueeze(1).to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits     = model(images)
                loss_class = criterion(logits, labels_bin)

                grad_f     = torch.autograd.grad(
                    logits, images,
                    grad_outputs=torch.ones_like(logits),
                    create_graph=True
                )[0]
                grad_x = F.conv2d(images, kernel_x, padding=(0,1))
                grad_y = F.conv2d(images, kernel_y, padding=(1,0))
                phi    = random.random()*2*math.pi
                dx, dy = math.cos(phi), math.sin(phi)
                dR     = grad_x*dx + grad_y*dy
                Xf     = (grad_f * dR).sum(dim=[1,2,3])
                loss_sym = (Xf**2).mean()

                ang = random.choice(angles)
                rot = TF.rotate(images, ang, fill=(fill_val,))
                logits_rot   = model(rot)
                loss_consist = F.mse_loss(logits, logits_rot)

                loss = (
                    loss_class
                    + lambda_sym     * loss_sym
                    + lambda_consist * loss_consist
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            agg["cls"]  += loss_class.item()
            agg["sym"]  += loss_sym.item()
            agg["cons"] += loss_consist.item()

        nb = len(train_loader)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"class={agg['cls']/nb:.4f}  "
            f"sym={agg['sym']/nb:.4f}  "
            f"cons={agg['cons']/nb:.4f}"
        )


    model.eval()
    for sample, label in test_loader:
        if label.item() == 7:
            break
    sample = sample.to(device)

    probs = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for a in angles:
            rot = TF.rotate(sample, a, fill=(fill_val,))
            p   = torch.sigmoid(model(rot)).item()
            probs.append(p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,6),
                                   gridspec_kw={'height_ratios':[2,3]})
    img = ((sample.cpu()*0.3081)+0.1307).squeeze()
    ax1.imshow(img, cmap='gray')
    ax1.axis('off'); ax1.set_title("Reference: '7'")
    ax2.plot(angles, probs, '-o')
    ax2.set_xlabel("Angle (°)"); ax2.set_ylabel("P('7')")
    ax2.set_title("Rotation Check (single sample)")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    sums   = [0.0]*len(angles)
    counts = [0]*len(angles)

    with torch.no_grad(), torch.cuda.amp.autocast():
        for img, lbl in test_loader:
            if lbl.item() != 7: 
                continue
            img = img.to(device)
            for i, a in enumerate(angles):
                rot = TF.rotate(img, a, fill=(fill_val,))
                p = torch.sigmoid(model(rot)).item()
                sums[i]   += p
                counts[i] += 1

    mean_probs = [sums[i]/counts[i] for i in range(len(angles))]

    plt.figure(figsize=(6,4))
    plt.plot(angles, mean_probs, '-o')
    plt.xlabel("Rotation Angle (°)")
    plt.ylabel("Mean P('7') over all test '7's")
    plt.title("Angle‐wise Average Prediction")
    plt.ylim(0,1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
