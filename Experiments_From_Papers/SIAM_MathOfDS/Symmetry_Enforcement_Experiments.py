import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class VectorField(nn.Module):
    def __init__(self, funcs):
        self.comp_array = funcs
        self.dim = np.shape(funcs)[0]

    def apply_comps(self, X):
        ans = torch.stack([self.comp_array[i](X) for i in range(self.dim)])
        ans = torch.transpose(ans,0,1)
        return ans

    def compute_vector_field_loss(self, model, X):
        X = X.requires_grad_(True)
        output = model(X)
        grads = torch.autograd.grad(
            outputs=output,
            inputs=X,
            grad_outputs=torch.ones_like(output),
            create_graph=True
        )[0]
        return grads

    def apply(self, model, X):
        comp_funcs = self.apply_comps(X)
        grad_f = self.compute_vector_field_loss(model, X)
        X_f = torch.mul(comp_funcs, grad_f)
        return X_f

def vector_field_loss(y_pred, y_target, X_f, symmetry_loss_weight=0.5, criterion_model=nn.MSELoss(), criterion_sym=nn.MSELoss()):
    model_loss = criterion_model
    symmetry_loss = criterion_sym(X_f, torch.zeros_like(X_f))
    total_loss = (1.0 - symmetry_loss_weight) * model_loss(y_pred, y_target) + symmetry_loss_weight * symmetry_loss
    return total_loss

def train_with_symmetry(X, y, VF, input_dim, symmetry_loss_weight=0.5, 
                        criterion_model=nn.MSELoss(), criterion_sym=nn.MSELoss(), 
                        lr=1e-3, n_epochs=1000):
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).float()
    model = MLP(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        X_f = VF.apply(model, X_tensor) 
        loss = vector_field_loss(y_pred, y_tensor.reshape(y_pred.shape), X_f, 
                                 symmetry_loss_weight=symmetry_loss_weight, criterion_model=criterion_model, criterion_sym=criterion_sym)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.6f}")

    return model

np.random.seed(0)
X = np.random.uniform(-3, 3, (3000, 2))
y = -X[:, 0]**2 - X[:, 1]**2 + 10.0

func_array = np.array([lambda x: -x[:, 1], lambda x: x[:, 0]])
myVF = VectorField(func_array)

trained_model = train_with_symmetry(X, y, myVF, input_dim=X.shape[1], symmetry_loss_weight=0.8, 
                                    criterion_sym=nn.CrossEntropyLoss(), n_epochs=5000)

X_test = torch.tensor(np.random.uniform(-3, 3, (1000, 2))).float()
y_test_pred = trained_model(X_test).detach()
y_test = -X_test[:, 0]**2 - X_test[:, 1]**2 + 10.0

sum(torch.add(y_test.reshape(y_test_pred.shape),y_test_pred,alpha=-1)**2)/y_test.shape[0]

y_train_preds = trained_model(torch.tensor(X).float()).detach().numpy()

sns.kdeplot(y_train_preds, label="Pred")
sns.kdeplot(y, label="GT")
plt.legend()
plt.title("Training Data Comparison")
plt.show()

sns.kdeplot(y_test_pred, label="Pred")
sns.kdeplot(y_test, label="GT")
plt.legend()
plt.title("Test Data Comparison")
plt.show()
