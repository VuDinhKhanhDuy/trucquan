import numpy as np
import matplotlib.pyplot as plt

# 1. Chuan bi du lieu lon (10,000 ban ghi cua song hinh sin co nhiễu)
x = np.linspace(0, 50, 10000)
values = np.sin(x) + np.random.normal(0, 0.1, len(x))

# 2. Bieu dien moi ban ghi bang 1 pixel
size = int(np.ceil(np.sqrt(len(values))))
pixel_matrix = np.zeros(size * size)
pixel_matrix[:len(values)] = values
pixel_matrix = pixel_matrix.reshape(size, size)

# 3. Ve Pixel-based Visualization
plt.figure(figsize=(6, 6))
plt.imshow(pixel_matrix, cmap="viridis")
plt.colorbar()
plt.title("Pixel-based Visualization")
plt.tight_layout()

# Phan tich:
# - Cac hoa tiet duong soc luan phien bieu thi chu ky tuan hoan cua tin hieu sine.
# - Cac hat mau hat li ti dac trung cho nhieu trang duoc them vao.

plt.show()
