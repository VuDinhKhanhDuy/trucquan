import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# 1. Load va chuan hoa du lieu
iris = load_iris()
scaled_data = MinMaxScaler().fit_transform(iris.data)
samples = [scaled_data[0], scaled_data[50], scaled_data[100]]
labels = ["Setosa (Sample 0)", "Versicolor (Sample 50)", "Virginica (Sample 100)"]

# 2. Ham ve Chernoff Face
def draw_face(ax, data, label):
    face_size = 0.5 + data[0] * 0.5
    eye_size = 0.05 + data[1] * 0.05
    mouth_curve = data[2] - 0.5
    nose_size = 0.05 + data[3] * 0.05
    
    # Ve guong mat
    face = plt.Circle((0.5, 0.5), face_size * 0.4, fill=False, linewidth=2)
    ax.add_patch(face)
    
    # Ve mat
    left_eye = plt.Circle((0.35, 0.6), eye_size, color="black")
    right_eye = plt.Circle((0.65, 0.6), eye_size, color="black")
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)
    
    # Ve mui
    nose = plt.Circle((0.5, 0.5), nose_size, color="black")
    ax.add_patch(nose)
    
    # Ve mieng
    x = np.linspace(0.35, 0.65, 100)
    y = 0.35 + mouth_curve * (x - 0.5)**2 * -4
    ax.plot(x, y, color="black", linewidth=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(label, weight='bold', size=12)

# 3. Hien thi cac guong mat
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, sample, name in zip(axes, samples, labels):
    draw_face(ax, sample, name)

plt.tight_layout()
plt.show()
