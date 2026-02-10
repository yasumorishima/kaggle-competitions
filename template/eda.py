import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# W&B init (optional - comment out if not needed)
# run = wandb.init(project="kaggle-competition-name", tags=["eda"])

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(f"Train: {train.shape}")
print(f"Test: {test.shape}")
train.head()

print("=== Data Types ===")
print(train.dtypes.value_counts())
print("\n=== Missing Values ===")
missing = train.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))
print("\n=== Describe ===")
train.describe()

target_col = "target"  # TODO: change to actual target column

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
train[target_col].hist(bins=50, ax=ax)
ax.set_title(f"Target Distribution (mean={train[target_col].mean():.4f})")
ax.set_xlabel(target_col)
plt.tight_layout()
plt.show()

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != target_col]

n = len(num_cols)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    train[col].hist(bins=50, ax=axes[i], alpha=0.7, label="train")
    test[col].hist(bins=50, ax=axes[i], alpha=0.5, label="test")
    axes[i].set_title(col)
    axes[i].legend()

for i in range(n, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

corr = train[num_cols + [target_col]].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=len(num_cols) < 20, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Matrix")
plt.tight_layout()
plt.show()

print("\n=== Top correlations with target ===")
print(corr[target_col].drop(target_col).sort_values(key=abs, ascending=False).head(10))
