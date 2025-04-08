import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic 2D data (e.g., ellipse shape)
# np.random.seed(42)
mean = [3, 4]

cov = [[2, 3], 
       [3, 5]]  # Covariance matrix for correlated features
X = np.random.multivariate_normal(mean, cov, 100).T  # Transpose to get shape (2, n_samples)

# Step 1: Standardize the data (zero mean, unit variance)
X_mean = np.mean(X, axis=1, keepdims=True)  # Mean of each feature
X_std = np.std(X, axis=1, keepdims=True)    # Std of each feature
X_normalized = (X - X_mean) / X_std  # Standardize the data

# Step 2: Compute covariance matrix
# Note: Using (n-1) for sample covariance, (n) would be population covariance
cov_matrix = X_normalized @ X_normalized.T / (X.shape[1] - 1)  # Sample covariance matrix

# Step 3: Compute eigenvalues and eigenvectors
# Note: np.linalg.eigh is used for symmetric matrices (like covariance matrices)
# It returns eigenvalues in ascending order, so we need to sort them
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 4: Sort eigenvectors by decreasing eigenvalues
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

# Step 5: Project data onto top k eigenvectors (let's reduce to 1D)
k = 1
W = eigenvectors[:, :k]
print("Projection Matrix W:\n", W)
print("X_centered shape:", X_normalized.shape)
print("W shape:", W.shape)
P_pi = W @ W.T
lambda_coordinates = W.T @ X_normalized  # coordinates of data in the new space, each column is a point in the new space with basis W
X_projections = P_pi @ X_normalized

# Optional: Reconstruct the data (approximation) for visualization
# This is not part of PCA but useful for visualization
print("X_projections shape:", X_projections.shape)
X_approx = X_projections * X_std + X_mean

# Plot original vs reconstructed
plt.figure(figsize=(8, 6))
plt.scatter(X[0, :], X[1, :], alpha=0.3, label='Original')
plt.scatter(X_approx[0, :], X_approx[1, :], alpha=0.7, label='Reconstructed (1D)')
plt.axis('equal')
plt.legend()
plt.title("PCA: Original vs Reconstructed from 1D")
plt.show()
