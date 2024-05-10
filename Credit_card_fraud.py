import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('ml1.csv')

# Feature engineering
X = data[['TransactionAmount', 'Location', 'Time']]  # Using TransactionAmount, Location, Time as features
X = pd.get_dummies(X)  # One-hot encode categorical features
y = data['Fraudulent']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = model.predict(X_test)

# Printing the actual and predicted values for each transaction
for actual, predicted in zip(y_test, y_pred):
    if predicted == 1:
        print("Fraudulent Transaction")
    else:
        print("Real Transaction")