import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# ---------------------------
# Bước 1: Tải và lọc dữ liệu MNIST cho số 8
# ---------------------------

print("Loading MNIST dataset...")
mnist_data = fetch_openml('mnist_784', version=1)

# Lọc lấy các ảnh có nhãn "8". Lấy một số mẫu nhất định (ở đây num_samples = 1000)
images = []
num_samples = 1000  
count = 0

# Lặp qua từng mẫu bằng cách dùng zip: lấy giá trị từ DataFrame/Series
for img, label in zip(mnist_data.data.values, mnist_data.target.values):
    if label == "8":  # So sánh với chuỗi "8"
        # img đã là numpy array, chỉ cần flatten (nếu chưa flat)
        img_np = img.flatten()  
        images.append(img_np)
        count += 1
        if count >= num_samples:
            break

# Ban đầu dữ liệu có kích thước (num_samples, 784).
# Chuyển về dạng (784, num_samples): mỗi cột là một mẫu ảnh.
images = np.array(images).T  
print("Shape của tập dữ liệu thực:", images.shape)  # (784, 1000)
D = images.shape[0]   # Số chiều mỗi mẫu (784)
N = images.shape[1]   # Số mẫu

# Hiển thị một số ảnh gốc
fig, axes = plt.subplots(3, 8, figsize=(16, 6))  # 3 hàng: ảnh sạch và ảnh nhiễu
for i in range(8):
    ori_img = images[:, i].reshape(28, 28)  # Reshape về kích thước 28x28
    axes[0, i].imshow(ori_img, cmap='gray')
    axes[0, i].axis('off')

# ---------------------------
# Bước 2: Tính trung bình và ma trận hiệp phương sai
# ---------------------------
# Tính trung bình theo hàng (mỗi hàng là 1 biến) => kết quả là vector (784,)
mu = np.mean(images, axis=1).reshape(D, 1)  # (784,1)

# Trung tâm hóa dữ liệu: trừ đi vector mu (mỗi cột)
X_centered = images - mu  
# Với dữ liệu dạng (784, N) và mỗi hàng là 1 biến, ta dùng np.cov (rowvar=True là mặc định)
cov = X_centered @ X_centered.T / N  # Ma trận hiệp phương sai
# cov có kích thước (784, 784)

# ---------------------------
# Bước 3: Phân rã giá trị riêng của ma trận hiệp phương sai
# ---------------------------
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# Sắp xếp theo thứ tự giảm dần
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# ---------------------------
# Bước 4: Ước lượng tham số PPCA
# ---------------------------
latent_dim = 2  # Không gian latent chỉ có 2 chiều
m = latent_dim

if m < D:
    # Ước lượng nhiễu: dùng trung bình các giá trị riêng từ (m+1) đến D
    sigma2 = np.mean(eigenvalues[m:])
else:
    sigma2 = 0.0

# Tính ma trận B theo công thức PPCA:
# B = U_m * sqrt(Λ_m - σ^2 I)
# eigenvectors[:, :m] có kích thước (D, m)
B = eigenvectors[:, :m] * np.sqrt(eigenvalues[:m] - sigma2)

print("Kích thước của B:", B.shape)
print("Kích thước trung bình mu:", mu.shape)
print("Giá trị nhiễu sigma^2:", sigma2)

# ---------------------------
# Bước 5: Sinh dữ liệu mới từ PPCA
# ---------------------------
def generate_ppca_samples(num_samples=5, add_noise=False):
    """
    Sinh các mẫu mới theo PPCA:
      x_new = μ + Bz + (optionally) nhiễu Gaussian với phương sai σ².
    Trả về dữ liệu dạng (D, num_samples), mỗi cột là một mẫu mới.
    """
    # Lấy mẫu z từ phân phối chuẩn với kích thước (m, num_samples)
    z = np.random.randn(m, num_samples)   # z có shape (2, num_samples)
    # Sinh dữ liệu mới: x_new = mu + Bz
    generated = mu + np.dot(B, z)          # kết quả có kích thước (D, num_samples)
    if add_noise and sigma2 > 0:
        noise = np.random.randn(D, num_samples) * np.sqrt(sigma2)
        generated += noise
    return generated

# Sinh một số mẫu mới (ở đây chọn add_noise=False)
samples_clean = generate_ppca_samples(num_samples=8, add_noise=False)
samples_noisy = generate_ppca_samples(num_samples=8, add_noise=True)

# ---------------------------
# Bước 6: Hiển thị các ảnh sinh ra (reshape về 28x28)
# ---------------------------
for i in range(samples_clean.shape[1]):  # Duyệt qua các cột, mỗi cột là một mẫu
    # Ảnh sạch (hàng trên)
    img_clean = samples_clean[:, i].reshape(28, 28)
    axes[1, i].imshow(img_clean, cmap='gray')
    axes[1, i].axis('off')

    # Ảnh nhiễu (hàng dưới)
    img_noisy = samples_noisy[:, i].reshape(28, 28)
    axes[2, i].imshow(img_noisy, cmap='gray')
    axes[2, i].axis('off')

# Tiêu đề cho từng hàng
# Điều chỉnh khoảng cách giữa các subplot và figure
plt.subplots_adjust(top=0.88, hspace=0.3)

# Tiêu đề cho từng hàng (tính theo tỷ lệ chiều cao figure)
fig.text(0.5, 0.89, "Gốc", ha='center', fontsize=12)
fig.text(0.5, 0.62, "Sạch", ha='center', fontsize=12)
fig.text(0.5, 0.33, "Có nhiễu", ha='center', fontsize=12)

# Tiêu đề toàn bộ
plt.suptitle("So sánh ảnh sinh ra từ PPCA: Gốc không nhiễu vs. có nhiễu", fontsize=14)

plt.show()
