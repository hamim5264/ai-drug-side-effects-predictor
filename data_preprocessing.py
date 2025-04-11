import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import joblib

# Load main dataset
df_main = pd.read_csv("data/drugs_side_effects_drugs_com.csv")

# Load Bangladeshi drugs dataset
df_bd = pd.read_csv("data/bd_drugs.csv")

# Combine both datasets
df = pd.concat([df_main, df_bd], ignore_index=True)



# Drop rows with missing target values
df.dropna(subset=['side_effects'], inplace=True)

# Drop unnecessary columns
df.drop(columns=[
    'drug_link',
    'medical_condition_url',
    'related_drugs',
    'medical_condition_description'
], inplace=True)

# Fill missing values safely
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(-1)

# Label encode features and target
le_drug = LabelEncoder()
df['drug_name_encoded'] = le_drug.fit_transform(df['drug_name'])

le_condition = LabelEncoder()
df['medical_condition_encoded'] = le_condition.fit_transform(df['medical_condition'])

le_effects = LabelEncoder()
df['side_effects_encoded'] = le_effects.fit_transform(df['side_effects'])

# Keep labels with at least 3 samples
value_counts = df['side_effects_encoded'].value_counts()
valid_labels = value_counts[value_counts > 2].index
df = df[df['side_effects_encoded'].isin(valid_labels)]

# Re-encode the filtered side_effects
le_effects = LabelEncoder()
df['side_effects_encoded'] = le_effects.fit_transform(df['side_effects_encoded'])


# Define features and target
X = df[['drug_name_encoded', 'medical_condition_encoded']]
y = df['side_effects_encoded']

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_idx, test_idx in split.split(X, y):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

# Save encoders
joblib.dump(le_drug, "models/le_drug.pkl")
joblib.dump(le_condition, "models/le_condition.pkl")
joblib.dump(le_effects, "models/le_effects.pkl")

# Save preprocessed data
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Save mapping of label -> original side_effect text
side_effect_map = pd.DataFrame({
    "encoded": df['side_effects_encoded'],
    "text": df['side_effects']
}).drop_duplicates()

side_effect_map.to_csv("data/side_effect_mapping.csv", index=False)


print("✅ Preprocessing complete!")
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)
