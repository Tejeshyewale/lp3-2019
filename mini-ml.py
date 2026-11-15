
# Titanic Survival Prediction using Machine Learning

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Select important columns
data = data[['Pclass', 'Sex', 'Age', 'SibSp',
             'Parch', 'Fare', 'Embarked', 'Survived']]

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Convert text into numbers
le = LabelEncoder()

data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])

# Separate input and output
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a new passenger
# Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
# Sex: Female=0, Male=1 (check encoding below)
# Embarked: encoded according to LabelEncoder

new_passenger = pd.DataFrame(
    [[1, 0, 25, 0, 0, 50, 0]],
    columns=X.columns
)

prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("\nPassenger Survived")
else:
    print("\nPassenger Did Not Survive")
