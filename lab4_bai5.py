import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# 1. Load, chuan hoa va tinh trung binh tung lop
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]
features = iris.feature_names

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
means = df.groupby('species')[features].mean()

# 2. Thiet lap thong so ve Radar Chart
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# 3. Ve radar chart cho tung lop
for name, row in means.iterrows():
    values = np.concatenate((row.values, [row.values[0]]))
    ax.plot(angles, values, 'o-', linewidth=2, label=name.capitalize())
    ax.fill(angles, values, alpha=0.15)

ax.set_thetagrids(np.degrees(angles[:-1]), features)
ax.set_title("Average Species Characteristics - Radar Chart", weight='bold', size=13, pad=20)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

plt.tight_layout()
plt.show()
