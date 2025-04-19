import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    targets = train_ds.targets
    pos = (targets == 0).sum().item()
    neg = (targets != 0).sum().item()
    pos_weight = torch.tensor(neg / pos, device=device)

    class SimpleMNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        def forward(self, x):
            return self.fc(x)

    model = SimpleMNISTNet().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    kernel_x = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=device)
    kernel_y = torch.tensor([[[[-0.5],[0.0],[0.5]]]], device=device)

    H, W = 28, 28
    ci = (H - 1) / 2.0
    cj = (W - 1) / 2.0
    i_coords = torch.arange(H, device=device).view(1,1,H,1).expand(1,1,H,W)
    j_coords = torch.arange(W, device=device).view(1,1,1,W).expand(1,1,H,W)
    grid_i = i_coords - ci
    grid_j = j_coords - cj

    lambda_sym = 0.5
    K = 4

    model.train()
    for epoch in range(10):
        for images, labels in train_loader:
            images = images.to(device)
            labels_bin = (labels == 0).float().unsqueeze(1).to(device)

            images.requires_grad_(True)
            logits = model(images)
            loss_class = criterion(logits, labels_bin)

            grad_f = torch.autograd.grad(
                logits, images,
                grad_outputs=torch.ones_like(logits),
                create_graph=True
            )[0]

            grad_img_x = F.conv2d(images, kernel_x, padding=(0,1))
            grad_img_y = F.conv2d(images, kernel_y, padding=(1,0))

            loss_sym = 0.0
            for _ in range(K):
                phi = random.random() * 2 * math.pi
                c, s = math.cos(phi), math.sin(phi)
                gi = c * grid_i - s * grid_j
                gj = s * grid_i + c * grid_j
                dR = grad_img_x * (-gj) + grad_img_y * gi
                Xf = (grad_f * dR).sum(dim=[1,2,3])
                loss_sym += (Xf**2).mean()
            loss_sym /= K

            loss = loss_class + lambda_sym * loss_sym

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: class={loss_class.item():.4f}, sym={loss_sym.item():.4f}")

    model.eval()
    for sample, label in test_loader:
        if label.item() == 0:
            break
    sample = sample.to(device)

    fill_val = (0.0 - 0.1307) / 0.3081

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    probs = []
    with torch.no_grad():
        for a in angles:
            rot = TF.rotate(sample, a, fill=(fill_val,))
            logit = model(rot.unsqueeze(0))
            p = torch.sigmoid(logit).item()
            probs.append(p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [2, 3]})

    img = ((sample.cpu() * 0.3081) + 0.1307).squeeze()
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Reference: '0' Image")
    ax1.axis('off')

    ax2.plot(angles, probs, '-o')
    ax2.set_xlabel("Rotation Angle (°)")
    ax2.set_ylabel("Predicted P('0')")
    ax2.set_title("Rotation Invariance Check")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
