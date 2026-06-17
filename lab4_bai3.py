import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# 1. Load va chuan hoa du lieu
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
features = iris.feature_names
scaled_data = MinMaxScaler().fit_transform(df[features])

# 2. Ham ve Star Glyph
def draw_star(ax, values, label):
    num_vars = len(values)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    ax.set_title(label, weight='bold', size=11, pad=15)
    ax.set_ylim(0, 1)

# 3. Ve Star Glyph cho 3 lop khac nhau
fig, axes = plt.subplots(1, 3, figsize=(14, 5), subplot_kw=dict(polar=True))
samples_idx = [0, 50, 100]
species_names = ["Setosa (Sample 0)", "Versicolor (Sample 50)", "Virginica (Sample 100)"]

for ax, idx, name in zip(axes, samples_idx, species_names):
    draw_star(ax, scaled_data[idx], name)

plt.tight_layout()

# Phan tich:
# - Setosa co dang glyph thu hep o goc petal, the hien petal length/width rat nho.
# - Virginica co hinh dang to, mo rong deu o tat ca cac canh.
# - Versicolor co kich thuoc va hinh dang trung gian, hoa hop giua Setosa va Virginica.

plt.show()
