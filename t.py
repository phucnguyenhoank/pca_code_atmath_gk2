import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic 2D data (e.g., ellipse shape)
np.random.seed(42)
mean = [3, 4]
cov = [[2, 3], 
       [3, 5]]  # Covariance matrix for correlated features
X = np.random.multivariate_normal(mean, cov, 100).T  # Shape: (2, n_samples)

# Step 1: Standardize the data (zero mean, unit variance)
X_mean = np.mean(X, axis=1, keepdims=True)  # Mean of each feature, shape: (2,1)
X_std = np.std(X, axis=1, keepdims=True)     # Std of each feature, shape: (2,1)
X_normalized = (X - X_mean) / X_std            # Standardized data

# Step 2: Compute covariance matrix (sample covariance with n-1)
cov_matrix = X_normalized @ X_normalized.T / (X.shape[1] - 1)  # Shape: (2,2)

# Step 3: Compute eigenvalues and eigenvectors (for symmetric matrices)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 4: Sort eigenvectors by decreasing eigenvalues
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

# Tính explained variance ratio cho các PC
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("Explained Variance Ratio:", explained_variance_ratio)

# Step 5: Project data onto top k eigenvectors (reduce to 1D)
k = 1
W = eigenvectors[:, :k]  # Projection matrix, shape: (2, 1)
lambda_coordinates = W.T @ X_normalized  # Dữ liệu 1D, shape: (1, n_samples)

# Tính phép chiếu ngược (projection onto PC) trong không gian normalized:
X_projections = W @ lambda_coordinates # W.T) @ X_normalized  # Dữ liệu đã chiếu, ở dạng normalized

# --- Visualization ---
plt.figure(figsize=(8, 6))
# Plot dữ liệu gốc (normalized)
plt.scatter(X_normalized[0, :], X_normalized[1, :], alpha=0.3, 
            label='Original Normalized', color='blue')
# Plot dữ liệu đã chiếu (reconstructed from 1D) trên không gian normalized
plt.scatter(X_projections[0, :], X_projections[1, :], alpha=0.7, 
            label='Reconstructed (1D)', color='orange')

# Vì dữ liệu normalized có trung bình 0 nên gốc sẽ là (0,0)
origin = np.array([0, 0])
PC_vector = W.flatten()  # Vector của PC1

# Vẽ vector PC tại gốc (0,0) với độ dài vừa đủ để nhìn rõ
plt.quiver(origin[0], origin[1], PC_vector[0], PC_vector[1], 
           angles='xy', scale_units='xy', scale=1, color='red', width=0.005, label='PC1')

# Vẽ các đường vuông góc từ mỗi điểm gốc normalized đến điểm chiếu trên PC
n_points_show = 20
indices = np.linspace(0, X_normalized.shape[1] - 1, n_points_show).astype(int)
for i in indices:
    x_orig = X_normalized[:, i]
    x_proj = X_projections[:, i]
    plt.plot([x_orig[0], x_proj[0]], [x_orig[1], x_proj[1]], 'g--', linewidth=1)

plt.xlabel("X1 (normalized)")
plt.ylabel("X2 (normalized)")
plt.title("PCA: Original vs Reconstructed from 1D on Normalized Data\nwith Projection Lines")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
