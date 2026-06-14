from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load same data split (must match train.py)
data = fetch_olivetti_faces()
X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Load saved model
model = joblib.load("savedmodel.pth")

# Predict and check accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.4f}")