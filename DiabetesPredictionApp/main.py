import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.title("Diabetes Prediction")

# classifier_name = st.sidebar.selectbox("Select Classifier", ("Logistic Regression",
#                                                              "GaussianNB", "BernoulliNB", "MultinomialNB",
#                                                              "KNN (KNeighbors)",
#                                                              "Decision Tree",
#                                                              "Random Forest",
#                                                              "Super Vector Machine (SVC)",
#                                                              "XGBoost (XGB)"))

diabetes = pd.read_csv('/Users/kaylakim/PycharmProjects/Diabetes/diabetes.csv')

# getting the input data from the user
# col1, col2, col3 = st.columns(3)
#
# with col1:
#     Pregnancies = st.text_input("Number of Pregnancies")
# with col2:
#     Glucose = st.text_input("Glocose Level")
# with col3:
#     BloodPressure = st.text_input("Blood Pressure value")
#
# with col1:
#     SkinThickness = st.text_input("Skin Thickness value")
# with col2:
#     Insulin = st.text_input("Insulin Level")
# with col3:
#     BMI = st.text_input("BMI value")
#
# with col1:
#     DiabetesPedigreeFunction = st.text_input("'Diabetes Pedigree Function value")
# with col2:
#     Age = st.text_input("Age of the Person")

# code for prediction
X = diabetes.drop(columns='Outcome', axis=1)
y = diabetes.Outcome
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=2)
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
accuracy = round(accuracy_score(y_valid, y_pred), 3)*100
st.write(f"Accuracy= {accuracy}%")

Pregnancies = st.text_input("Number of Pregnancies")
Glucose = st.text_input("Glocose Level")
BloodPressure = st.text_input("Blood Pressure value")
SkinThickness = st.text_input("Skin Thickness value")
Insulin = st.text_input("Insulin Level")
BMI = st.text_input("BMI value")
DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
Age = st.text_input("Age of the Person")

# creating a button for prediction
diagnosis = ''
if st.button("Diabetes Test Result"):
    prediction = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                 DiabetesPedigreeFunction, Age]])
    if (prediction[0] == 0):
        diagnosis = 'The Person is NOT Diabetic.'
    else:
        diagnosis = 'The Person is Diabetic.'
st.success(diagnosis)

