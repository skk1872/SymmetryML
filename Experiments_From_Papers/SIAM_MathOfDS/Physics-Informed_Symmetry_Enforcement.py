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

t_train = np.linspace(0, 2*np.pi, 400)
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

def invariant_fn(state):
    x  = state[:, 0]
    y  = state[:, 1]
    px = state[:, 2]
    py = state[:, 3]
    ang_mom = x * py - y * px - 1.0
    ham = 0.5 * (x**2 + y**2 + px**2 + py**2) - 1.0
    return torch.stack([ang_mom, ham], dim=1)

mse_loss = nn.MSELoss()

def baseline_loss(pred, target):
    return mse_loss(pred, target)

lambda_inv = 10.0
def symmetry_loss(pred, target):
    state_loss = mse_loss(pred, target)
    inv_loss   = mse_loss(invariant_fn(pred), torch.zeros_like(invariant_fn(pred)))
    return state_loss + lambda_inv * inv_loss

def train_model(model, loss_fn, optimizer, num_epochs, t_train_tensor, train_tensor):
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(t_train_tensor)
        loss = loss_fn(pred, train_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    return loss_history

num_epochs = 5000

baseline_model = OrbitNet()
optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=1e-4)
print("Training baseline model:")
loss_history_baseline = train_model(baseline_model, baseline_loss, optimizer_baseline, num_epochs, t_train_tensor, train_tensor)

symmetry_model = OrbitNet()
optimizer_symmetry = optim.Adam(symmetry_model.parameters(), lr=1e-4)
print("\nTraining symmetry-enforced model:")
loss_history_symmetry = train_model(symmetry_model, symmetry_loss, optimizer_symmetry, num_epochs, t_train_tensor, train_tensor)

baseline_model.eval()
symmetry_model.eval()

with torch.no_grad():
    baseline_test_pred = baseline_model(t_test_tensor)
    symmetry_test_pred = symmetry_model(t_test_tensor)
    
    baseline_state_mse = mse_loss(baseline_test_pred, test_tensor).item()
    symmetry_state_mse = mse_loss(symmetry_test_pred, test_tensor).item()
    
    baseline_inv_mse = mse_loss(invariant_fn(baseline_test_pred), torch.zeros_like(invariant_fn(baseline_test_pred))).item()
    symmetry_inv_mse = mse_loss(invariant_fn(symmetry_test_pred), torch.zeros_like(invariant_fn(symmetry_test_pred))).item()

print(f"\nBaseline Model:  Test State MSE = {baseline_state_mse:.6f},  Test Invariant MSE = {baseline_inv_mse:.6f}")
print(f"Symmetry Model: Test State MSE = {symmetry_state_mse:.6f}, Test Invariant MSE = {symmetry_inv_mse:.6f}")

plt.figure(figsize=(8,4))
plt.plot(loss_history_baseline, label='Baseline')
plt.plot(loss_history_symmetry, label='Symmetry Enforced')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training Loss History')
plt.legend()
plt.show()

baseline_train_pred = baseline_model(t_train_tensor).detach().numpy()
symmetry_train_pred = symmetry_model(t_train_tensor).detach().numpy()

plt.figure(figsize=(6,6))
plt.plot(train_data[:,0], train_data[:,1], 'o', label='True (train)', alpha=0.6)
plt.plot(baseline_train_pred[:,0], baseline_train_pred[:,1], 'x', label='Baseline Pred (train)', alpha=0.6)
plt.plot(symmetry_train_pred[:,0], symmetry_train_pred[:,1], '+', label='Symmetry Pred (train)', alpha=0.6)
plt.title('Train Trajectory in x-y Plane')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()
plt.show()

baseline_test_pred_np = baseline_test_pred.detach().numpy()
symmetry_test_pred_np = symmetry_test_pred.detach().numpy()

plt.figure(figsize=(6,6))
plt.plot(test_data[:,0], test_data[:,1], 'o', label='True (test)', alpha=0.3)
plt.plot(baseline_test_pred_np[:,0], baseline_test_pred_np[:,1], 'x', label='Baseline Pred (test)', alpha=0.6)
plt.plot(symmetry_test_pred_np[:,0], symmetry_test_pred_np[:,1], '+', label='Symmetry Pred (test)', alpha=0.6)
plt.title('Test Trajectory in x-y Plane')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()
plt.show()
