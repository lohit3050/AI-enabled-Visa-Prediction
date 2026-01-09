import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns -> mode
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values handled")

# -----------------------------
# 3. Generate Processing Time (Target)
# -----------------------------
# Synthetic but realistic logic
df["processing_time_days"] = (
    120
    - df["document_completeness_score"]
    + (df["previous_visa_rejections"] * 10)
    + np.random.randint(5, 25, size=len(df))
)

# Minimum processing time constraint
df["processing_time_days"] = df["processing_time_days"].clip(lower=7)

print("Processing time target generated")

# -----------------------------
# 4. Encode Categorical Variables
# -----------------------------
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Categorical encoding completed")

# -----------------------------
# 5. Define Targets
# -----------------------------
# Classification target
y_classification = df["visa_category (Label)"]

# Regression target
y_regression = df["processing_time_days"]

# Feature set
X = df.drop(["visa_category (Label)", "processing_time_days"], axis=1)

print("Features and targets prepared")

# -----------------------------
# 6. Save Clean Dataset
# -----------------------------
output_path = "data/processed/visa_cleaned_milestone1.csv"
df.to_csv(output_path, index=False)

print("Milestone 1 completed successfully")
print("Final Dataset Shape:", df.shape)
