import numpy as np
import matplotlib.pyplot as plt

# --- Bước 0: Sinh dữ liệu n chiều ---
# np.random.seed(42)
n_dim = 5         # số chiều ban đầu (n)
n_samples = 200   # số mẫu

# Sinh một ma trận ngẫu nhiên để tạo ma trận hiệp phương sai dương xác định
A = np.random.rand(n_dim, n_dim)
cov = A @ A.T  # tạo ra ma trận đối xứng dương xác định
mean = np.random.rand(n_dim) * 10  # vector trung bình ngẫu nhiên, nhân với 10 để có giá trị lớn hơn

# Sinh dữ liệu từ phân phối chuẩn đa biến
X = np.random.multivariate_normal(mean, cov, n_samples).T  # X có shape (n_dim, n_samples)

# --- Bước 1: Chuẩn hóa dữ liệu ---
X_mean = np.mean(X, axis=1, keepdims=True)  # shape: (n_dim, 1)
X_std  = np.std(X, axis=1, keepdims=True)    # shape: (n_dim, 1)
X_normalized = (X - X_mean) / X_std           # Dữ liệu chuẩn hóa: trung bình 0 và std 1 cho mỗi chiều

# --- Bước 2: Tính ma trận hiệp phương sai ---
cov_matrix = X_normalized @ X_normalized.T / (n_samples)  # shape: (n_dim, n_dim)

# --- Bước 3: Tính giá trị riêng và vector riêng ---
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

# --- Bước 4: Tính explained variance ratio cho top 2 PC ---
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues[:2] / total_variance
print("Explained Variance Ratio (PC1, PC2):", explained_variance_ratio)

# --- Bước 5: Chiếu dữ liệu xuống 2 chiều với PCA ---
k = 2
W = eigenvectors[:, :k]         # Ma trận chiếu có shape (n_dim, 2)
X_proj = W.T @ X_normalized       # Dữ liệu sau chiếu: shape (2, n_samples)
# Nếu muốn phục hồi lại trong không gian chuẩn hóa (tức dữ liệu chỉ nằm trên mặt phẳng PC)
X_recon = W @ (W.T @ X_normalized)  # phục hồi dữ liệu trong không gian chuẩn hóa

# --- Bước 6: Visualization trên không gian 2D (PC1 vs PC2) ---
plt.figure(figsize=(8, 6))
# Vẽ dữ liệu sau khi chiếu (tương ứng với tọa độ của PC1 và PC2)
plt.scatter(X_proj[0, :], X_proj[1, :], color='blue', alpha=0.6, label='Projected Data (2D)')

# Vẽ gốc tọa độ (ở dữ liệu chuẩn hóa trung bình = 0)
origin = np.array([0, 0])

# Vẽ vector PC trên không gian 2D: các thành phần này tương ứng với tọa độ của các trục mới
# Trong không gian chiếu, PC1 và PC2 tạo thành hệ tọa độ trực chuẩn.
pc1 = np.array([1, 0])  # trục thứ nhất
pc2 = np.array([0, 1])  # trục thứ hai

plt.quiver(origin[0], origin[1], pc1[0], pc1[1], color='red', scale_units='xy', scale=3, label='PC1')
plt.quiver(origin[0], origin[1], pc2[0], pc2[1], color='green',scale_units='xy', scale=3, label='PC2')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA: Projection to 2D from {}D Data".format(n_dim))
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
