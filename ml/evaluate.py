import joblib
import pandas as pd
from sklearn.metrics import classification_report
from ml.preprocess import X, y
from sklearn.model_selection import train_test_split

model = joblib.load("model.joblib")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = model.predict(X_val)


report = classification_report(y_val, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
print("\nClassification Report:\n", df_report)


