import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# --- Step 0: Generate synthetic 3D data (e.g., stretched blob shape) ---
n_samples = 150
mean = [3, 4, 5]
cov = [[3, 2, 1.5], 
       [2, 4, 2], 
       [1.5, 2, 5]]  # Covariance matrix for correlated 3D data
X = np.random.multivariate_normal(mean, cov, n_samples).T  # Shape: (3, n_samples)

# --- Step 1: Standardize the data ---
X_mean = np.mean(X, axis=1, keepdims=True)
X_std = np.std(X, axis=1, keepdims=True)
X_normalized = (X - X_mean) / X_std  # Shape: (3, n_samples)

# --- Step 2: Covariance matrix ---
cov_matrix = X_normalized @ X_normalized.T / X.shape[1]  # Shape: (3, 3)

# --- Step 3: Eigen decomposition ---
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# --- Step 4: Sort eigenvectors by descending eigenvalues ---
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

for j in range(eigenvectors.shape[1]):
    PC_vector = eigenvectors[:, j].flatten()  # Chuyển thành vector 1D
    print(f"PC vector {j+1}:", PC_vector)

# Explained variance ratio
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("Explained Variance Ratio:", explained_variance_ratio)

# --- Step 5: Project to 1D using top principal component ---
k = 1
W = eigenvectors[:, :k]  # Shape: (3, 1)
lambda_coordinates = W.T @ X_normalized  # Shape: (1, n_samples)

# --- Step 6: Reconstruct from 1D (in normalized space) ---
X_projections = W @ lambda_coordinates  # Shape: (3, n_samples)

# --- 3D Visualization ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Gốc dữ liệu đã normalize
ax.scatter(*X_normalized, alpha=0.3, label='Original Normalized', color='blue')

# Dữ liệu đã chiếu ngược từ 1D
ax.scatter(*X_projections, alpha=0.8, label='Reconstructed (1D)', color='orange')

# Vẽ vector PC1 từ gốc
origin = np.zeros(3)
pc1 = W[:, 0]
ax.quiver(*origin, *pc1, color='red', label='PC1', linewidth=2, arrow_length_ratio=0.1)

# Vẽ các đường chiếu
n_lines = n_samples // 3
indices = np.linspace(0, X.shape[1], n_lines, endpoint=False, dtype=int)
for i in indices:
    x_orig = X_normalized[:, i]
    x_proj = X_projections[:, i]
    ax.plot([x_orig[0], x_proj[0]],
            [x_orig[1], x_proj[1]],
            [x_orig[2], x_proj[2]],
            'g--', linewidth=1)

ax.set_xlabel("X1 (normalized)")
ax.set_ylabel("X2 (normalized)")
ax.set_zlabel("X3 (normalized)")
ax.set_title("PCA: 3D Data → 1D Projection (and Back) in Normalized Space")
ax.legend()
plt.tight_layout()
plt.show()
