import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def vector_field_loss(X_f, y_target, symmetry_loss_weight=1.0):
    mse_loss = nn.MSELoss()
    symmetry_loss = torch.mean(X_f ** 2)
    total_loss = mse_loss(y_target, torch.zeros_like(y_target)) + symmetry_loss_weight * symmetry_loss
    return total_loss

def compute_vector_field_loss(model, X):
    X = X.requires_grad_(True)
    output = model(X)
    grads = torch.autograd.grad(
        outputs=output,
        inputs=X,
        grad_outputs=torch.ones_like(output),
        create_graph=True
    )[0]
    return grads

def train_with_symmetry(X, y, input_dim, symmetry_loss_weight=1.0, lr=1e-3, n_epochs=5000):
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).float()
    model = MLP(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        X_f = compute_vector_field_loss(model, X_tensor)
        loss = vector_field_loss(X_f, y_tensor, symmetry_loss_weight=symmetry_loss_weight)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.6f}")

    return model

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.uniform(-1, 1, (100, 2))
    y = X[:, 0]**2 + X[:, 1]**2
    trained_model = train_with_symmetry(X, y, input_dim=X.shape[1], symmetry_loss_weight=1.0, n_epochs=5000)
    X_test = torch.tensor(np.random.uniform(-1, 1, (10, 2))).float()
    y_test_pred = trained_model(X_test).detach().numpy()
    print("Predicted Outputs:", y_test_pred)
