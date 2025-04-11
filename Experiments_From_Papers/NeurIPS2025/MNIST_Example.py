import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import random

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)

class SimpleMNISTNet(nn.Module):
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMNISTNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

kernel_x = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=device)
kernel_y = torch.tensor([[[[-0.5], [0.0], [0.5]]]], device=device)

H, W = 28, 28
center_j = (W - 1) / 2.0
center_i = (H - 1) / 2.0
i_coords = torch.arange(H, dtype=torch.float32, device=device).view(1, 1, H, 1).expand(1, 1, H, W)
j_coords = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, 1, W).expand(1, 1, H, W)
grid_i = i_coords - center_i
grid_j = j_coords - center_j

lambda_sym = 0.5
lambda_consistency = 0.1
fill_val = -0.424

model.train()
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.to(device)
        labels = (labels == 0).float().unsqueeze(1).to(device)
        images.requires_grad_(True)
        outputs = model(images)
        loss_class = criterion(outputs, labels)
        grad_f = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
        grad_img_x = F.conv2d(images, kernel_x, padding=(0, 1))
        grad_img_y = F.conv2d(images, kernel_y, padding=(1, 0))
        dR = grad_img_x * (-grid_j) + grad_img_y * grid_i
        X_f = (grad_f * dR).sum(dim=[1, 2, 3])
        loss_sym = (X_f ** 2).mean()
        epsilon = 2.0
        rotated_images = TF.rotate(images, epsilon, fill=(fill_val,))
        outputs_rot = model(rotated_images)
        loss_consistency = F.mse_loss(outputs, outputs_rot)
        loss = loss_class + lambda_sym * loss_sym + lambda_consistency * loss_consistency
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Total Loss = {loss.item():.4f}, Class Loss = {loss_class.item():.4f}, Sym Loss = {loss_sym.item():.4f}, Consistency Loss = {loss_consistency.item():.4f}")

model.eval()
sample, label = next(iter(test_loader))
sample = sample.to(device)
angles = [0, 45, 90, 135, 180, 225, 270, 315]
predictions = []
with torch.no_grad():
    for angle in angles:
        rotated_image = TF.rotate(sample, angle, fill=(fill_val,))
        output = model(rotated_image)
        predictions.append(output.item())
        print(f"Angle: {angle}° -> Output Probability: {output.item():.4f}")

fig, axs = plt.subplots(1, len(angles), figsize=(15, 3))
for idx, angle in enumerate(angles):
    rotated_image = TF.rotate(sample, angle, fill=(fill_val,))
    axs[idx].imshow(rotated_image.squeeze().cpu(), cmap='gray')
    axs[idx].set_title(f"{angle}°\nProb: {predictions[idx]:.2f}")
    axs[idx].axis('off')
plt.show()
