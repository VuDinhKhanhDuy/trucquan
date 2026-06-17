import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 1. Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Draw Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap - Iris Dataset")
plt.tight_layout()

# Phan tich cap bien tuong quan cao:
# - Petal length va Petal width tuong quan thuan cuc ky manh (0.96).
# - Sepal length tuong quan thuan manh voi ca hai dac trung petal.
# - Sepal width co xu huong tuong quan nghich voi cac bien con lai.

plt.show()
