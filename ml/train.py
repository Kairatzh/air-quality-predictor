from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from ml.preprocess import X_train, y_train, y_test, X_test

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = classification_report(y_true=y_test, y_pred=y_pred)
print(metrics)