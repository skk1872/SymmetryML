import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

def generate_data(t_vals):
    x = np.cos(t_vals)
    y = np.sin(t_vals)
    px = -np.sin(t_vals)
    py = np.cos(t_vals)
    return np.stack([x, y, px, py], axis=1)

t_train = np.linspace(0, 4*np.pi, 800)
t_test  = np.linspace(0, 6*np.pi, 1200)

train_data = generate_data(t_train)
test_data  = generate_data(t_test)

t_train_tensor = torch.tensor(t_train, dtype=torch.float32).unsqueeze(1)
train_tensor   = torch.tensor(train_data, dtype=torch.float32)
t_test_tensor  = torch.tensor(t_test, dtype=torch.float32).unsqueeze(1)
test_tensor    = torch.tensor(test_data, dtype=torch.float32)

class OrbitNet(nn.Module):
    def __init__(self):
        super(OrbitNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    def forward(self, t):
        return self.net(t)

model = OrbitNet()

def invariant_fn(state):
    x  = state[:, 0]
    y  = state[:, 1]
    px = state[:, 2]
    py = state[:, 3]
    return x*py - y*px - 1.0

mse_loss = nn.MSELoss()
lambda_inv = 10.0

def total_loss(pred, target):
    state_loss = mse_loss(pred, target)
    inv_loss   = mse_loss(invariant_fn(pred), torch.zeros_like(invariant_fn(pred)))
    return state_loss + lambda_inv * inv_loss

optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5000
loss_history = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(t_train_tensor)
    loss = total_loss(pred, train_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    test_pred = model(t_test_tensor)
    test_state_mse = mse_loss(test_pred, test_tensor).item()
    test_inv_mse   = mse_loss(invariant_fn(test_pred), torch.zeros_like(invariant_fn(test_pred))).item()

print(f"Test State MSE: {test_state_mse:.6f}, Test Invariant MSE: {test_inv_mse:.6f}")

plt.figure(figsize=(8,4))
plt.plot(loss_history)
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.show()

train_pred = model(t_train_tensor).detach().numpy()
plt.figure(figsize=(6,6))
plt.plot(train_data[:,0], train_data[:,1], 'o', label='True (train)', alpha=0.6)
plt.plot(train_pred[:,0], train_pred[:,1], 'x', label='Predicted (train)', alpha=0.6)
plt.title('Train Trajectory in x-y Plane')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()
plt.show()
