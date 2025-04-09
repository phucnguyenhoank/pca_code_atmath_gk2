import numpy as np
import matplotlib.pyplot as plt

# Hàm sinh dữ liệu thực: ảnh 2x2 với pattern nhất định.
def generate_real_images(num_samples=1000):
    """
    Sinh ảnh 2x2 với các mẫu:
      - Góc trên trái và dưới phải: giá trị tối (0-100)
      - Góc trên phải và dưới trái: giá trị sáng (150-180)
    Các giá trị được chuẩn hóa về [0, 1]
    
    Trả về:
      Một mảng có kích thước (4, num_samples), mỗi cột là một mẫu ảnh 2x2 (đã flattened).
    """
    dataset = []
    for _ in range(num_samples):
        dark_value_1 = np.random.randint(0, 100)
        dark_value_2 = np.random.randint(max(0, dark_value_1 - 50), min(100, dark_value_1 + 50))
        bright_value_1 = np.random.randint(150, 180)
        bright_value_2 = np.random.randint(150, 180)
        pixel_array = np.array([
            [dark_value_1, bright_value_1],
            [bright_value_2, dark_value_2]
        ], dtype=np.float32) / 255.0
        # Lưu ý: Ảnh flattened có dạng (4,)
        dataset.append(pixel_array.flatten())
    # Ban đầu dataset có shape (num_samples, 4); chuyển về (4, num_samples)
    return np.array(dataset).T

# --- Bước 1: Sinh tập dữ liệu thực ---
real_images = generate_real_images(num_samples=1000)
# Mỗi sample là 1 cột, tổng số chiều của mẫu là D (ở đây D=4)
print("Shape của tập dữ liệu thực:", real_images.shape)  # Sẽ in (4, 1000)
D = real_images.shape[0]     # Số chiều của mỗi mẫu (4)
N = real_images.shape[1]     # Số mẫu (1000)

# Hiển thị một số ảnh gốc
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    img_vec = real_images[:, i]
    # Chuyển vector (độ dài 4) thành ma trận 2x2
    img = img_vec.reshape(2, 2)
    axes[0, i].imshow(img, cmap='gray', interpolation='nearest')
    axes[0, i].axis('off')

# --- Bước 2: Tính trung bình (μ) của dữ liệu ---
# Với dữ liệu theo cột, trung bình tính theo các cột -> axis=1 cho mỗi biến.
mu = np.mean(real_images, axis=1)      # shape: (4,)
# Chuyển mu về dạng cột để thuận tiện cho các phép tính sau.
mu = mu.reshape(D, 1)

# --- Bước 3: Tính ma trận hiệp phương sai (sau khi trừ trung bình) ---
X_centered = real_images - mu          # mỗi cột trừ đi vector trung bình
# Vì mỗi hàng là một biến và mỗi cột là một mẫu, np.cov với rowvar=True (mặc định) là phù hợp.
cov = X_centered @ X_centered.T / N  # Ma trận hiệp phương sai
# Kích thước cov: (4, 4)

# --- Bước 4: Phân rã giá trị riêng của ma trận hiệp phương sai ---
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# Sắp xếp theo thứ tự giảm dần:
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# --- Bước 5: Ước lượng tham số PPCA ---
latent_dim = 2   # Số chiều của không gian latent (m)
m = latent_dim
if m < D:
    # Ước lượng nhiễu sigma2: lấy trung bình của các giá trị riêng không được giữ lại (từ m+1 đến D)
    sigma2 = np.mean(eigenvalues[m:])
else:
    sigma2 = 0.0

# Tính ma trận B theo công thức PPCA:
# B = U_m * sqrt(Λ_m - σ^2 I)
# eigenvectors[:, :m] có kích thước (D, m)
# np.sqrt(eigenvalues[:m] - sigma2) có kích thước (m,)
B = eigenvectors[:, :m] * np.sqrt(eigenvalues[:m] - sigma2)

print("Kích thước của B:", B.shape)
print("Trung bình mu:\n", mu)
print("Giá trị nhiễu sigma^2:", sigma2)

# --- Bước 6: Sinh dữ liệu mới với PPCA ---
def generate_ppca_samples(num_samples=5, add_noise=False):
    """
    Sinh các mẫu mới theo PPCA:
      x_new = μ + B z + (optionally) nhiễu Gaussian với phương sai σ².
      
    Trả về dữ liệu dạng (D, num_samples), mỗi cột là một mẫu mới.
    """
    # Với dữ liệu theo cột, lấy mẫu z với kích thước (m, num_samples)
    z = np.random.randn(m, num_samples)   # N(0, I)
    # Dữ liệu mới: x = mu + B*z (với mu có dạng (D,1) và B có (D, m))
    generated = mu + np.dot(B, z)           # Kết quả có shape (D, num_samples)
    if add_noise and sigma2 > 0:
        noise = np.random.randn(D, num_samples) * np.sqrt(sigma2)
        generated += noise
    return generated

# Sinh một số mẫu mới (ở đây thêm nhiễu để thấy rõ hiệu ứng của PPCA)
new_samples = generate_ppca_samples(num_samples=8, add_noise=True)

# --- Bước 7: Hiển thị các mẫu ảnh 2x2 được sinh ra ---

for i in range(new_samples.shape[1]):  # Duyệt theo cột (mỗi cột là một mẫu)
    img_vec = new_samples[:, i]
    # Chuyển vector (độ dài 4) thành ma trận 2x2
    img = img_vec.reshape(2, 2)
    axes[1, i].imshow(img, cmap='gray', interpolation='nearest')
    axes[1, i].axis('off')

# Thêm tiêu đề cho từng hàng bằng fig.text
fig.text(0.04, 0.72, "Gốc", va='center', ha='center', rotation='vertical', fontsize=12)
fig.text(0.04, 0.28, "Có nhiễu", va='center', ha='center', rotation='vertical', fontsize=12)
plt.suptitle("Các mẫu ảnh mới được sinh ra từ PPCA (samples theo cột)")
plt.show()
