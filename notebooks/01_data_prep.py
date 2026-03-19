"""
=============================================================
  PHASE 1 — DATA PREPARATION & EXPLORATORY DATA ANALYSIS
  AI Placement Suite | College Project
=============================================================
Dataset: Kaggle — Campus Recruitment Dataset
URL: https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement

Run this script first. It:
  1. Loads and cleans the raw CSV
  2. Engineers features
  3. Saves a clean processed CSV for model training
  4. Generates EDA charts
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/Placement_Data_Full_Class.csv")
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

# ── 2. Clean ──────────────────────────────────────────────────────────────────
# Fill missing salary (not placed students have no salary)
df["salary"] = df["salary"].fillna(0)

# Encode categoricals
binary_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t",
               "workex", "specialisation", "status"]
for col in binary_cols:
    df[col] = df[col].astype("category").cat.codes

# Target: placed = 1, not placed = 0
df["placed"] = df["status"].apply(lambda x: 1 if x == 1 else 0)

# ── 3. Feature Engineering ────────────────────────────────────────────────────
# Weighted academic score (normalised)
df["academic_score"] = (
    0.25 * df["ssc_p"] +
    0.25 * df["hsc_p"] +
    0.30 * df["degree_p"] +
    0.20 * df["etest_p"]
)

# Work experience bonus
df["exp_bonus"] = df["workex"] * 5

# ── 4. Save Processed Data ────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/processed_placement.csv", index=False)
print("\n✅ Saved data/processed_placement.csv")

# ── 5. EDA Plots ──────────────────────────────────────────────────────────────
os.makedirs("notebooks/plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# Plot 1: Placement rate by gender
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x="gender", hue="placed", ax=axes[0])
axes[0].set_title("Placement by Gender")
axes[0].set_xticklabels(["Female", "Male"])

# Plot 2: Academic score vs Salary scatter
placed = df[df["placed"] == 1]
axes[1].scatter(placed["academic_score"], placed["salary"], alpha=0.6, color="#5DCAA5")
axes[1].set_xlabel("Academic Score")
axes[1].set_ylabel("Salary")
axes[1].set_title("Academic Score vs Salary")

# Plot 3: Work experience impact
sns.countplot(data=df, x="workex", hue="placed", ax=axes[2])
axes[2].set_title("Work Experience vs Placement")
axes[2].set_xticklabels(["No Experience", "Has Experience"])

plt.tight_layout()
plt.savefig("notebooks/plots/eda_overview.png", dpi=150)
plt.close()

# Plot 4: Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p",
                "academic_score", "workex", "placed", "salary"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            square=True, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("notebooks/plots/correlation_heatmap.png", dpi=150)
plt.close()

print("✅ EDA plots saved to notebooks/plots/")
print("\nColumn summary:")
print(df.describe())
