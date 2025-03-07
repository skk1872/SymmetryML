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

learning_rate = 0.001
num_epochs = 5000
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
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

final_loss = loss_history[-1]
print(f"Training completed: {num_epochs} epochs, learning rate = {learning_rate}, final training loss = {final_loss:.6f}")

plt.figure(figsize=(8,4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss History")
plt.show()

grid_points = 100
x_test = np.linspace(-1, 1, grid_points)
y_test = np.linspace(-1, 1, grid_points)
X_test, Y_test = np.meshgrid(x_test, y_test)
test_data = np.stack([X_test.ravel(), Y_test.ravel()], axis=1)
test_tensor = torch.tensor(test_data, dtype=torch.float32)
model.eval()
with torch.no_grad():
    preds_test = model(test_tensor).squeeze().numpy().reshape(X_test.shape)

plt.figure(figsize=(6,6))
contour = plt.contourf(X_test, Y_test, preds_test, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour Plot of Network Predictions (f(x,y))")
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

f_x = grads[:, 0].detach().numpy()
f_y = grads[:, 1].detach().numpy()
x_np = data_tensor[:, 0].detach().numpy()
y_np = data_tensor[:, 1].detach().numpy()

F = np.column_stack([x_np * f_x, y_np * f_x, f_x, x_np * f_y, y_np * f_y, f_y])
U, S, Vt = np.linalg.svd(F, full_matrices=False)
P = Vt[-1, :]

print("Recovered infinitesimal generator coefficients (up to scale):")
print(f"a: {P[0]:.4f}, b: {P[1]:.4f}, c: {P[2]:.4f}, d: {P[3]:.4f}, e: {P[4]:.4f}, f: {P[5]:.4f}")
print("Ground truth coefficients for rotation (-y, x): a=0, b=-1, c=0, d=1, e=0, f=0")
