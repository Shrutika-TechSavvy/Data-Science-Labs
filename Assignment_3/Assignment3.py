# ==========================================
# Complete Exploratory Data Analysis (EDA)
# Univariate + Bivariate + Multivariate
# with Outlier Detection
# ==========================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Uncleaned Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Step 3: Dataset Overview
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

# Step 4: Select Columns
numerical_cols = ['Age', 'Fare']
categorical_cols = ['Sex', 'Pclass', 'Survived']

# ======================
# UNIVARIATE ANALYSIS
# ======================

# Mean, Median, Mode
print("\n--- Mean, Median, Mode ---")
for col in numerical_cols:
    print(f"\nColumn: {col}")
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Mode:", df[col].mode()[0])

# Standard Deviation & Variance
print("\n--- Standard Deviation & Variance ---")
for col in numerical_cols:
    print(f"\nColumn: {col}")
    print("Standard Deviation:", df[col].std())
    print("Variance:", df[col].var())

# Z-Score
print("\n--- Z-Score ---")
for col in numerical_cols:
    df[col + "_Zscore"] = (df[col] - df[col].mean()) / df[col].std()

print(df[['Age', 'Age_Zscore', 'Fare', 'Fare_Zscore']].head())

# Outlier Detection (IQR)
print("\n--- Outlier Detection (IQR Method) ---")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"\nColumn: {col}")
    print("Lower Bound:", lower)
    print("Upper Bound:", upper)
    print("Outliers Count:", outliers.shape[0])

# Histograms
plt.figure(figsize=(8,5))
plt.hist(df['Age'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Age")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df['Fare'], bins=30, color='orange', edgecolor='black')
plt.title("Histogram of Fare")
plt.show()

# Box Plots
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Age'], color='lightgreen')
plt.title("Box Plot of Age")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['Fare'], color='lightpink')
plt.title("Box Plot of Fare")
plt.show()

# Bar Charts
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df)
plt.title("Bar Chart of Sex")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title("Bar Chart of Passenger Class")
plt.show()

# ======================
# BIVARIATE ANALYSIS
# ======================

# Age vs Fare
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Fare', data=df)
plt.title("Scatter Plot: Age vs Fare")
plt.show()

# Survival vs Sex
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival vs Sex")
plt.show()

# Survival vs Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival vs Passenger Class")
plt.show()

# ======================
# MULTIVARIATE ANALYSIS
# ======================

# Scatter Plot with Hue (Age vs Fare vs Survival)
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title("Age vs Fare vs Survival")
plt.show()

# Pair Plot (Numerical + Survival)
sns.pairplot(df[['Age', 'Fare', 'Survived']], hue='Survived')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


