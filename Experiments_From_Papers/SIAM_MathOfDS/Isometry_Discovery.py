import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)
N = 1000
X = np.random.uniform(-5, 5, (N, 3))
x, y, z = X[:, 0], X[:, 1], X[:, 2]

def f(x, y, z):
    return x**2 + 0.5*y**2 - y*z + 0.5*z**2 - 16

labels = (f(x, y, z) > 0).astype(int)
print("Class distribution:", np.unique(labels, return_counts=True))

def feature_map(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    phi1 = x**2
    phi2 = np.sqrt(2)*x*y
    phi3 = np.sqrt(2)*x*z
    phi4 = y**2
    phi5 = np.sqrt(2)*y*z
    phi6 = z**2
    return np.vstack([phi1, phi2, phi3, phi4, phi5, phi6]).T

Phi_X = feature_map(X)

clf = LogisticRegression(C=1e6, solver='lbfgs', fit_intercept=True)
clf.fit(Phi_X, labels)
print("Learned coefficients (feature space):")
print(clf.coef_)
print("Learned intercept:", clf.intercept_)

expected = np.array([1, 0, 0, 0.5, -1/np.sqrt(2), 0.5])
expected = expected / np.linalg.norm(expected)
learned = clf.coef_.flatten()
learned_norm = learned / np.linalg.norm(learned)
print("Normalized learned coefficients:", learned_norm)
print("Normalized expected coefficients (ignoring intercept):", expected)

def classify_point(xyz):
    phi = feature_map(xyz.reshape(1, -1))
    score = np.dot(phi, clf.coef_.T) + clf.intercept_
    prob = 1 / (1 + np.exp(-score))
    label = (score > 0).astype(int)
    return prob[0,0], label[0,0]

pt = np.array([1.0, 0.5, -0.5])
prob, lab = classify_point(pt)
print(f"Point {pt} -> probability: {prob:.3f}, label: {lab}")

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    rm = np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
                   [2*(b*c + a*d),     a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                   [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c]])
    return rm

theta = np.deg2rad(10)
R = rotation_matrix([0, 0, 1], theta)

X_rot = (R @ X.T).T
Phi_X_rot = feature_map(X_rot)

pred_original = clf.predict(Phi_X)
pred_rotated = clf.predict(Phi_X_rot)

diff = np.mean(pred_original != pred_rotated)
print(f"Fraction of points with different predictions after rotation: {diff:.3f}")

mask = np.abs(X[:, 2]) < 0.5
plt.figure(figsize=(8,6))
plt.scatter(X_rot[mask, 0], X_rot[mask, 1], c=pred_rotated[mask], cmap='bwr', alpha=0.7)
plt.title('Representative Rotated Predictions (z ~ 0)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
