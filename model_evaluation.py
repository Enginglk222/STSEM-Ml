from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc_rf = round(accuracy_score(y_test, y_pred),3)
    recall_rf = round(recall_score(y_test, y_pred, average='weighted'),3)
    prec_rf = round(precision_score(y_test, y_pred, average='weighted'),3)
    f1_rf = round(f1_score(y_test, y_pred, average='weighted'),3)

    return f"Accuracy Score: {acc_rf}\nRecall Score: {recall_rf}\nPrecision Score: {prec_rf}\nF-1 Score: {f1_rf}"