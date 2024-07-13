import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data.drop(["Unnamed: 0", "Loan_ID"], axis=1, inplace=True)
    # Handling missing data
    categoricals_nulls = ["Gender", "Dependents", "Education", "Credit_History", "Self_Employed"]
    for i in categoricals_nulls:
        vals = data[i].mode().values[0]
        data[i].fillna(vals, inplace=True)
    
    numericals_nulls = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    for i in numericals_nulls:
        vals = data[i].median()
        data[i].fillna(vals, inplace=True)

    categoricals = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
               "Credit_History", "Property_Area", "Loan_Status"]

    numericals = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

    # Encoding values
    le = LabelEncoder()
    for i in categoricals:
        data[i] = le.fit_transform(data[i])
    
    # Data augmentation
    y = data.Loan_Status
    X = data.drop("Loan_Status" , axis = 1)
    smote = SMOTE(sampling_strategy="all")
    X_sm, y_sm = smote.fit_resample(X,y)
    y_sm.value_counts()
    data = pd.concat([X_sm,y_sm], axis = 1)
    
    return data

def split_data(data,target):
    y = data[target]
    X = data.drop(target , axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    return X_train, X_test, y_train, y_test



