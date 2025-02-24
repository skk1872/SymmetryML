import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

N = 1000
theta = np.random.uniform(0, 2*np.pi, N)
r = np.sqrt(np.random.uniform(0, 1, N))
x = r * np.cos(theta)
y = r * np.sin(theta)
data = np.stack([x, y], axis=1)
labels = x**2 + y**2

data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
num_epochs = 5000
loss_history = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data_tensor)
    loss = criterion(output, labels_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

plt.figure(figsize=(8,4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss History")
plt.show()

model.eval()
with torch.no_grad():
    preds = model(data_tensor).squeeze().numpy()
plt.figure(figsize=(6,6))
plt.scatter(labels, preds, alpha=0.5)
plt.xlabel("True f(x,y)")
plt.ylabel("Predicted f(x,y)")
plt.title("True vs Predicted Values")
plt.show()

data_tensor.requires_grad = True
output = model(data_tensor)
grad_outputs = torch.ones_like(output)
grads = torch.autograd.grad(outputs=output, inputs=data_tensor, grad_outputs=grad_outputs, create_graph=True)[0]
x_tensor = data_tensor[:, 0]
y_tensor = data_tensor[:, 1]
directional_deriv = -y_tensor * grads[:, 0] + x_tensor * grads[:, 1]
mean_deriv = directional_deriv.abs().mean().item()
std_deriv = directional_deriv.std().item()

print(f"Mean absolute directional derivative along (-y, x): {mean_deriv:.6f}")
print(f"Standard deviation of directional derivative: {std_deriv:.6f}")

plt.figure(figsize=(8,4))
plt.hist(directional_deriv.detach().numpy(), bins=50)
plt.xlabel("Directional Derivative along (-y,x)")
plt.ylabel("Frequency")
plt.title("Histogram of Directional Derivative")
plt.show()
