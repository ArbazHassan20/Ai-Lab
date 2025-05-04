import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Sample dummy dataset (replace with actual CSV file)
data = pd.DataFrame({
    'square_footage': [1500, 2000, 1800, 2200],
    'bedrooms': [3, 4, 3, 5],
    'bathrooms': [2, 3, 2, 4],
    'age': [10, 5, 8, 2],
    'location': ['A', 'B', 'A', 'C'],
    'price': [300000, 500000, 400000, 600000]
})

# Handle missing values (if any)
data = data.dropna()

# Encode categorical features
data['location'] = LabelEncoder().fit_transform(data['location'])

# Define features and target
X = data.drop('price', axis=1)
y = data['price']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Predict new house
new_house = [[2000, 3, 2, 5, 1]]  # values must match feature order
predicted_price = model.predict(new_house)
print("Predicted House Price:", predicted_price)
Task 2:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dummy dataset
emails = [
    "Win a million dollars now", "Let's catch up tomorrow", 
    "Click here to claim your prize", "Lunch at 1pm?", 
    "Get free vacation now", "Meeting notes attached"
]
labels = [1, 0, 1, 0, 1, 0]  # 1=spam, 0=not spam

# Convert text to numeric
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
y = labels

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = SVC(kernel='linear')
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
print("Spam Detection Accuracy:", accuracy_score(y_test, y_pred))

# Classify a new email
new_email = ["Congratulations! You won a prize"]
new_vec = vectorizer.transform(new_email)
print("Spam Prediction (1=spam, 0=not):", model.predict(new_vec))

Task3:
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Dummy data
data = pd.DataFrame({
    'total_spending': [1000, 2000, 500, 7000, 300, 8000],
    'age': [25, 40, 20, 35, 19, 45],
    'visits': [5, 10, 2, 15, 1, 20],
    'frequency': [3, 6, 1, 10, 1, 12],
    'value': [0, 1, 0, 1, 0, 1]  # 0 = low value, 1 = high value
})

# Split into X and y
X = data.drop('value', axis=1)
y = data['value']

# Scaling (optional but recommended)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Predict & evaluate
y_pred = model.predict(x_test)
print("Customer Classification Accuracy:", accuracy_score(y_test, y_pred))

# Classify a new customer
new_customer = scaler.transform([[3000, 30, 12, 8]])
print("Customer Value Prediction:", model.predict(new_customer))
