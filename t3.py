import numpy as np
import matplotlib.pyplot as plt

# --- Sinh dữ liệu 2D ---
# np.random.seed(42)
n_samples = 100
mean = [3, 4]
cov = [[2, 3], 
       [3, 5]]  # Ma trận hiệp phương sai cho các biến có liên quan
X = np.random.multivariate_normal(mean, cov, n_samples).T  # Shape: (2, n_samples)

# --- Bước 1: Chuẩn hóa dữ liệu ---
X_mean = np.mean(X, axis=1, keepdims=True)  # shape: (2,1)
X_std  = np.std(X, axis=1, keepdims=True)     # shape: (2,1)
X_normalized = (X - X_mean) / X_std            # Dữ liệu chuẩn hóa

# --- Bước 2: Tính ma trận hiệp phương sai trên dữ liệu chuẩn hóa ---
cov_matrix = X_normalized @ X_normalized.T / (X.shape[1])  # shape: (2,2)

# --- Bước 3: Tính eigenvalues và eigenvectors ---
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
# Sắp xếp theo thứ tự giảm dần (eigenvalues lớn nhất trước)
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

# Tính explained variance ratio
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("Explained Variance Ratio:", explained_variance_ratio)

# --- Bước 4: PCA giảm về 1D trên không gian normalized ---
k = 1
W = eigenvectors[:, :k]               # Projection matrix, shape: (2,1)
lambda_coordinates = W.T @ X_normalized # Dữ liệu 1D trên không gian normalized

# Lưu ý: X_projections (trong không gian normalized) = W @ lambda_coordinates
X_projections = W @ lambda_coordinates

# --- Chuyển hướng PC từ không gian normalized sang không gian gốc ---
# Trong quá trình chuẩn hóa, mỗi chiều được biến đổi bởi: x_normalized = (x - mu) / sigma.
# Do đó, để đưa vector PC sang không gian gốc, ta nhân phần tử tương ứng với std.
# (Điều này sẽ cho vector PC "đúng" về hướng theo không gian gốc.)
PC_vector_original = (W * X_std).flatten()  # Đây chưa được chuẩn hóa thành vector đơn vị

# Chuyển thành vector đơn vị:
v_unit = PC_vector_original / np.linalg.norm(PC_vector_original)

# --- Tính hình chiếu theo công thức trên không gian gốc ---
# Với mỗi điểm x (trong X), hình chiếu của nó lên đường thẳng qua mu với hướng v_unit là:
# x_proj = mu + dot(x - mu, v_unit)*v_unit.
X_proj_orig = np.zeros_like(X)
for i in range(X.shape[1]):
    x = X[:, i]
    x_proj = X_mean.flatten() + np.dot(x - X_mean.flatten(), v_unit) * v_unit
    X_proj_orig[:, i] = x_proj

# --- Visualization trên không gian gốc (dữ liệu ban đầu) ---
plt.figure(figsize=(8, 6))
# Vẽ dữ liệu gốc (chưa chuẩn hóa)
plt.scatter(X[0, :], X[1, :], alpha=0.3, label='Original Data', color='blue')
# Vẽ dữ liệu hình chiếu (x_proj) đã được tính trong không gian gốc
plt.scatter(X_proj_orig[0, :], X_proj_orig[1, :], alpha=0.7, label='Projected Data (1D)', color='orange')

# Vẽ vector PC trong không gian gốc tại trung điểm dữ liệu (mu)
origin = X_mean.flatten()
scale_factor = 3  # hệ số scale để hiển thị vector PC rõ ràng hơn
plt.quiver(origin[0], origin[1], v_unit[0]*scale_factor, v_unit[1]*scale_factor, angles='xy', scale_units='xy', scale=1, color='red', width=0.005, label='PC1')

# Vẽ các đường nối vuông góc từ mỗi điểm dữ liệu đến hình chiếu của nó
n_points_show = n_samples // 2
indices = np.linspace(0, X.shape[1], n_points_show, endpoint=False, dtype=int)
for i in indices:
    x_orig = X[:, i]
    x_proj = X_proj_orig[:, i]
    plt.plot([x_orig[0], x_proj[0]], [x_orig[1], x_proj[1]], 'g--', linewidth=1)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("PCA: Original Data vs. 1D Projection in Original Space\nwith Perpendicular Projection Lines and PC1")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
