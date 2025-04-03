import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
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

model = SimpleMNISTNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
lambda_sym = 0.5
NUM_ROTATIONS = 3

for epoch in range(10):
    for images, labels in train_loader:
        labels = (labels == 0).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss_class = criterion(outputs, labels)
        sym_losses = []
        for _ in range(NUM_ROTATIONS):
            angle = random.uniform(0, 360)
            rotated_images = torch.stack([TF.rotate(img, angle) for img in images])
            outputs_rot = model(rotated_images)
            sym_losses.append(torch.mean((outputs - outputs_rot) ** 2))
        loss_sym = torch.mean(torch.stack(sym_losses))
        loss = loss_class + lambda_sym * loss_sym
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Total Loss={loss.item():.4f}, Class Loss={loss_class.item():.4f}, Sym Loss={loss_sym.item():.4f}")

sample, label = next(iter(test_loader))
angles = [0, 45, 90, 135, 180, 225, 270, 315]
predictions = []
model.eval()
with torch.no_grad():
    for angle in angles:
        rotated_image = TF.rotate(sample, angle)
        output = model(rotated_image)
        predictions.append(output.item())
        print(f"Angle: {angle}° -> Output Probability: {output.item():.4f}")

fig, axs = plt.subplots(1, len(angles), figsize=(15, 3))
for idx, angle in enumerate(angles):
    rotated_image = TF.rotate(sample, angle)
    axs[idx].imshow(rotated_image.squeeze(), cmap='gray')
    axs[idx].set_title(f"{angle}°\nProb: {predictions[idx]:.2f}")
    axs[idx].axis('off')
plt.show()
