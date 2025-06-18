import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Correlation matrix
corr = df.corr()

# Plot with resized figure
plt.figure(figsize=(10, 8))  # <--- adjust size here
ax = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)

# Move x-axis labels to top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.xticks(rotation=45, ha='left')
plt.yticks(rotation=0)
plt.title("Correlation Matrix", pad=20)

plt.tight_layout()  # better fit inside figure area
plt.show()
