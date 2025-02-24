import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def generate_data(n_points, x_range=(-1,1), y_range=(-0.1,0.1)):
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    y = np.random.uniform(y_range[0], y_range[1], n_points)
    data = np.stack([x, y], axis=1)
    labels = np.sin(np.pi * x) * np.exp(-y**2)
    return data, labels

n_train = 1000
train_data, train_labels = generate_data(n_train)
n_test = 2000
test_data, test_labels = generate_data(n_test)

train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

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

mse_loss = nn.MSELoss()

def prediction_loss(pred, target):
    return mse_loss(pred, target)

def symmetry_loss_fn(model, data):
    pred_pos = model(data)
    data_neg = data.clone()
    data_neg[:, 1] = -data_neg[:, 1]
    pred_neg = model(data_neg)
    return mse_loss(pred_pos, pred_neg)

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 3000
loss_history = []
sym_weight = 0.0
switch_epoch = num_epochs // 2
target_sym_weight = 10.0

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(train_data_tensor)
    loss_pred = prediction_loss(pred, train_labels_tensor)
    if epoch < switch_epoch:
        current_sym_weight = 0.0
    else:
        current_sym_weight = target_sym_weight
    loss_sym = symmetry_loss_fn(model, train_data_tensor)
    total_loss = loss_pred + current_sym_weight * loss_sym
    total_loss.backward()
    optimizer.step()
    loss_history.append(total_loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss.item():.6f}, "
              f"Prediction Loss: {loss_pred.item():.6f}, Symmetry Loss: {loss_sym.item():.6f}, "
              f"Symmetry Weight: {current_sym_weight}")

model.eval()
with torch.no_grad():
    test_pred = model(test_data_tensor)
    test_pred_loss = mse_loss(test_pred, test_labels_tensor).item()
    test_sym_loss = symmetry_loss_fn(model, test_data_tensor).item()
print(f"Test Prediction MSE: {test_pred_loss:.6f}, Test Symmetry MSE: {test_sym_loss:.6f}")

plt.figure(figsize=(8,4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Training Loss History")
plt.show()

grid_x = np.linspace(-1, 1, 200)
grid_y = np.linspace(-0.1, 0.1, 200)
X, Y = np.meshgrid(grid_x, grid_y)
grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

model.eval()
with torch.no_grad():
    grid_preds = model(grid_tensor).cpu().numpy().reshape(X.shape)

plt.figure(figsize=(6,5))
plt.contourf(X, Y, grid_preds, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour Plot of Predicted Function")
plt.show()

with torch.no_grad():
    test_preds = model(test_data_tensor).cpu().numpy().squeeze()
plt.figure(figsize=(6,6))
plt.scatter(test_labels, test_preds, alpha=0.5)
plt.xlabel("True f(x,y)")
plt.ylabel("Predicted f(x,y)")
plt.title("Test: True vs Predicted")
plt.plot([test_labels.min(), test_labels.max()],
         [test_labels.min(), test_labels.max()], 'r--')
plt.show()
