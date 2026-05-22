"""
Support Vector Machine (SVM) — classification with linear and RBF kernels.
"""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_and_evaluate(X_train, X_test, y_train, y_test, kernel: str) -> None:
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"--- SVM (kernel={kernel}) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print()


def main() -> None:
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("=== Support Vector Machine ===")
    print(f"Target names: {list(data.target_names)}\n")

    train_and_evaluate(X_train, X_test, y_train, y_test, kernel="linear")
    train_and_evaluate(X_train, X_test, y_train, y_test, kernel="rbf")


if __name__ == "__main__":
    main()
