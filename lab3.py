import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Tải dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

# Vẽ Scatter Plot cho petal length và petal width
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    palette="Set1"
)
plt.title("Scatter Plot - Iris Dataset")
plt.show()

from sklearn.datasets import load_wine
import seaborn as sns
import pandas as pd

# Tải dataset Wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

# Vẽ Pairplot
sns.pairplot(
    df,
    hue="target",
    diag_kind="kde",
    palette="Set2"
)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

# Chuẩn bị dữ liệu
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["class"] = wine.target

# Chọn 5 thuộc tính để trực quan hóa
features = ["alcohol", "malic_acid", "ash", "flavanoids", "proline"]
df_plot = df[features + ["class"]]

# Chuẩn hóa dữ liệu về khoảng [1]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_plot[features])
df_scaled = pd.DataFrame(scaled_features, columns=features)
df_scaled["class"] = df_plot["class"]

# Vẽ biểu đồ Parallel Coordinates
plt.figure(figsize=(10, 6))
parallel_coordinates(
    df_scaled,
    "class",
    colormap=plt.cm.Set1,
    linewidth=1
)
plt.title("Parallel Coordinates Plot - Wine Dataset")
plt.xlabel("Features")
plt.ylabel("Normalized Value")
plt.show()

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Tải dữ liệu MNIST và lấy mẫu 2000 bản ghi
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data[:2000], mnist.target[:2000]

# 1. Áp dụng PCA giảm xuống 2 chiều
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(7, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap="tab10", s=10)
plt.title("PCA Visualization of MNIST")
plt.colorbar()
plt.show()

# 2. Áp dụng t-SNE giảm xuống 2 chiều
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(7, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.astype(int), cmap="tab10", s=10)
plt.title("t-SNE Visualization of MNIST")
plt.colorbar()
plt.show()