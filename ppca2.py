import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# ---------------------------
# Bước 1: Tải và lọc dữ liệu MNIST cho số 8
# ---------------------------
# Định nghĩa transform để chuyển ảnh thành tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Chuyển về tensor với giá trị trong [0, 1]
])

# Tải tập dữ liệu MNIST (ví dụ sử dụng test dataset)
mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Lọc lấy các ảnh có nhãn "8". Lấy một số mẫu nhất định (ở đây num_samples = 1000)
images = []
num_samples = 1000  
count = 0
for img, label in mnist_data:
    if label == 8:
        # Chuyển tensor sang numpy, loại bỏ kích thước dư và flatten ảnh (28x28 -> vector 784)
        img_np = img.numpy().squeeze().flatten()
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

# ---------------------------
# Bước 2: Tính trung bình và ma trận hiệp phương sai
# ---------------------------
# Tính trung bình theo hàng (mỗi hàng là 1 biến) => kết quả là vector (784,)
mu = np.mean(images, axis=1).reshape(D, 1)  # (784,1)

# Trung tâm hóa dữ liệu: trừ đi vector mu (mỗi cột)
X_centered = images - mu  
# Với dữ liệu dạng (784, N) và mỗi hàng là 1 biến, ta dùng np.cov (rowvar=True là mặc định)
cov = np.cov(X_centered)
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
new_samples = generate_ppca_samples(num_samples=8, add_noise=False)

# ---------------------------
# Bước 6: Hiển thị các ảnh sinh ra (reshape về 28x28)
# ---------------------------
fig, axes = plt.subplots(1, 8, figsize=(16, 2))
for i in range(new_samples.shape[1]):  # Duyệt qua các cột, mỗi cột là một mẫu
    img_vec = new_samples[:, i]
    # Reshape từ vector (784,) thành ảnh 28x28
    img = img_vec.reshape(28, 28)
    axes[i].imshow(img, cmap='gray')
    axes[i].axis('off')
plt.suptitle("Các ảnh '8' mới được sinh ra từ PPCA (samples theo cột)")
plt.show()
