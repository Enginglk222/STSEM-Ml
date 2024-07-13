from data_preprocessing import load_data, preprocess_data, split_data
from model_training import train_model, save_model
from model_evaluation import evaluate_model
from explain_model import explain_local, explain_global
import shap

# Load and Preprocess
data = load_data("loan_data_1.csv")
data = preprocess_data(data)

# Split the data
target = 'Loan_Status'
X_train, X_test, y_train, y_test = split_data(data, target)

# Train the model
model = train_model(X_train, y_train)

# Save the model
save_model(model,"model.pkl")

# Evaluate the model
evaluation_result = evaluate_model(model, X_test, y_test)
print(evaluation_result)

# Explain the model
shap.initjs()
explain_local(model, X_train, local_plots='local_plots', index = 0)
explain_global(model, X_train, global_plots='global_plots')