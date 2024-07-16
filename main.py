import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap 
shap.initjs()


# Load and preprocess data
data = pd.read_csv("loan_data_1.csv")
data.drop(columns=['Loan_ID', 'Unnamed: 0'], axis=1, inplace=True)

# Fill missing values
categoricals_nulls = ["Gender", "Dependents", "Education", "Credit_History", "Self_Employed"]
for col in categoricals_nulls:
    vals = data[col].mode().values[0]
    data[col].fillna(vals, inplace=True)

numericals_nulls = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
for col in numericals_nulls:
    vals = data[col].median()
    data[col].fillna(vals, inplace=True)

# Encoding categorical features
categoricals = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
                "Credit_History", "Property_Area", "Loan_Status"]
numericals = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

le = LabelEncoder()
for col in categoricals:
    data[col] = le.fit_transform(data[col])

# Prepare data for modeling
y = data["Loan_Status"]
X = data.drop("Loan_Status", axis=1)

# SMOTE for oversampling
smote = SMOTE(sampling_strategy="all")
X_sm, y_sm = smote.fit_resample(X, y)

data = pd.concat([X_sm, y_sm], axis=1)

# Split the dataset
y = data.Loan_Status
X = data.drop("Loan_Status", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# XAI(SHAP)
observation_index = 102  # Kullanıcının girdiği indeks

# Kullanıcının istediği gözlemi alma
observation = X_train.iloc[observation_index]

# Tahmin yapma
prediction = model.predict([observation])[0]  # Tek gözlemle tahmin

# SHAP değerlerini hesaplama
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(observation)

# Waterfall grafiği
plt.figure()
shap.plots.waterfall(shap.Explanation(values=shap_values[prediction], 
                                          base_values=explainer.expected_value[prediction], 
                                          data=observation, 
                                          feature_names=X_train.columns),max_display=len(X_train.columns), show = False)
plot_path = f'static/waterfall_plot_{observation_index}.png'
plt.tight_layout()
plt.savefig(plot_path)  # Grafiği kaydet
plt.close()


""" (1 kere oluşturulması yeterli)
# Global Interpretation
all_shap_values = explainer.shap_values(X_train)
#Summary_plot
plt.figure()
shap.summary_plot(all_shap_values[0], features=X_train, show = False)
plt.savefig("static/global_shap.png")  # Grafiği kaydet
plt.close()
"""