import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# -----------------------------
# 1. Load Dataset
# -----------------------------
data_path = "data/raw/visa_recommendation_120k_updated.csv"
df = pd.read_csv(data_path)

print("Initial Dataset Shape:", df.shape)
print(df.columns)

# -----------------------------
# 2. Handle Missing Values
# -----------------------------
# Numerical columns -> median
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns -> mode
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled")

# -----------------------------
# 3. Encode Categorical Variables
# -----------------------------
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("Categorical encoding completed")

# -----------------------------
# 4. Save Clean Dataset
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/visa_cleaned_milestone1.csv"
df.to_csv(output_path, index=False)

print("✅ Milestone 1 completed successfully")
print("Final Dataset Shape:", df.shape)
