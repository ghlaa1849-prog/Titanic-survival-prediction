# Import the libraries needed for the web app
import joblib
import pandas as pd
import streamlit as st

# Set the page title and icon of the web app
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

# Load the saved model, scaler, and feature columns (saved earlier with joblib)
# cache_resource makes Streamlit load them only once, not on every click
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("feature_columns.pkl")
    return model, scaler, columns

model, scaler, feature_columns = load_artifacts()

# Numerical columns that the scaler was fitted on during training
NUMERICAL = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

# Outlier capping limits (IQR method) used in preprocessing for Age and Fare
AGE_LOWER, AGE_UPPER = 2.5, 54.5
FARE_LOWER, FARE_UPPER = 0.0, 65.6344

# App title and short instruction
st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger information and click Predict.")

# Input form: the user enters the 7 features of the passenger
with st.form("passenger_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        embarked = st.selectbox("Embarked", ["S", "C", "Q"])

    with col2:
       sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0, step=1)
       parch = st.number_input("Parch", min_value=0, max_value=10, value=0, step=1)
       fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0) 
    # Button that triggers the prediction  
    submitted = st.form_submit_button("Predict")

# Run the prediction only after the button is clicked
if submitted:

    # Step 1: apply the same outlier capping used in preprocessing
    age_c = min(max(age, AGE_LOWER), AGE_UPPER)
    fare_c = min(max(fare, FARE_LOWER), FARE_UPPER)

    # Step 2: create one row with exactly the training columns (all zeros at first)
    row = pd.DataFrame([[0] * len(feature_columns)], columns=feature_columns, dtype=float)

   # Step 3: fill in the numerical values entered by the user
    row.loc[0, "Pclass"] = pclass
    row.loc[0, "Age"] = age_c
    row.loc[0, "SibSp"] = sibsp
    row.loc[0, "Parch"] = parch
    row.loc[0, "Fare"] = fare_c

  # Step 4: one-hot encoding (set the matching Sex and Embarked columns to 1)
    row.loc[0, f"Sex_{sex}"] = 1
    row.loc[0, f"Embarked_{embarked}"] = 1

    # Step 5: scale the numerical features using the saved scaler
    row[NUMERICAL] = scaler.transform(row[NUMERICAL])

    # Step 6: predict the class (0 or 1) and the survival probability
    prediction = model.predict(row[feature_columns])[0]
    probability = model.predict_proba(row[feature_columns])[0][1]

    # Step 7: display the result to the user
    st.subheader("Result")
    if prediction == 1:       
        st.success(f"This passenger would likely SURVIVE (probability: {probability:.0%})")
    else:
        st.error(f"This passenger would likely NOT survive (survival probability: {probability:.0%})")

