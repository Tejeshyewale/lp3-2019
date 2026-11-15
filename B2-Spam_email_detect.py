
# Email Spam Detection using KNN and SVM

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    'email': [
        'Hello how are you',
        'Meeting is scheduled today',
        'Please submit your assignment',
        'Let us meet tomorrow',
        'Congratulations you won a prize',
        'Win free money now',
        'Claim your free gift',
        'You won a lottery',
        'Get free cash now',
        'Click here to win money'
    ],

    'label': [
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# Convert email text into numerical values
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['email'])

y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

# SVM Model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

# Accuracy
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

# Classification report
print("\nKNN Classification Report:")
print(classification_report(y_test, knn_pred, zero_division=0))

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_pred, zero_division=0))

# Test a new email
new_email = ["Congratulations! You won free money"]

new_data = vectorizer.transform(new_email)

print("\nNew Email Prediction:")
print("KNN:", "Spam" if knn.predict(new_data)[0] == 1 else "Not Spam")
print("SVM:", "Spam" if svm.predict(new_data)[0] == 1 else "Not Spam")
