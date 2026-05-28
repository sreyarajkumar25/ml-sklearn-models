from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Features:
# [Hours Studied, Attendance]
# Attendance:
# Low = 0
# Medium = 1
# High = 2

X = [
    [2, 0],
    [3, 1],
    [5, 2],
    [7, 2],
    [10, 3]
]

# Target:
# Fail = 0
# Pass = 1

y = [1, 1, 1, 1, 1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# New student prediction
new_student = [[5, 1]]

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("PASS")
else:
    print("FAIL")