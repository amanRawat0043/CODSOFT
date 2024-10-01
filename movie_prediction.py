import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
file_path = r"C:\Users\H\Downloads\IMDb Movies India.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Step 2: Data Preprocessing
# Check the dataset structure
print(df.info())  # See available columns and their data types
print(df.head())  # Inspect the first few rows

# Check if necessary columns are available
expected_columns = ['genre', 'director', 'actors', 'rating']
for col in expected_columns:
    if col not in df.columns:
        print(f"Column '{col}' is missing in the dataset.")
        exit()

# Step 3: Data Cleaning & Handling Missing Values
# Remove rows with missing values in 'genre', 'director', 'actors', or 'rating'
df = df.dropna(subset=expected_columns)

# Step 4: Encoding categorical features
# Label encoding for 'director' and 'actors'
label_enc_director = LabelEncoder()
label_enc_actors = LabelEncoder()

df['director_encoded'] = label_enc_director.fit_transform(df['director'])
df['actors_encoded'] = label_enc_actors.fit_transform(df['actors'])

# One-hot encoding for 'genre' (since genre can have multiple categories)
df = pd.get_dummies(df, columns=['genre'], drop_first=True)

# Step 5: Feature Selection
# Define the features (exclude original 'director', 'actors', and 'rating')
X = df.drop(columns=['rating', 'director', 'actors'])
y = df['rating']  # Target variable

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Selection and Training
# Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Prediction
y_pred = model.predict(X_test)

# Step 9: Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
