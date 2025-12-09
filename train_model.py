import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATA_FILE = "sign_data.csv"
MODEL_FILE = "sign_model.pkl"

def main():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} samples from {DATA_FILE}")

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    y_pred = model.predict(X_test)
    print("Evaluation on test set:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

if __name__ == "__main__":
    main()
