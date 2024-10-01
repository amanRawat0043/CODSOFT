import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (update the file path as per your local machine)
file_path = 'Titanic-Dataset.csv'  # Change this to the location of your Titanic dataset
titanic_data = pd.read_csv(file_path)

# Preprocessing: Handle missing values and encode categorical variables
# Fill missing age values with the median
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Fill missing embarked values with the mode (most common embarkation point)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column as it has many missing values
titanic_data.drop(columns=['Cabin'], inplace=True)

# Convert categorical columns 'Sex' and 'Embarked' into numeric values
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'], drop_first=True)

# Drop unnecessary columns for the model
titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

# Define features (X) and target variable (y)
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_rep)
