import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

# Set the title
st.title("Churn Prediction Dashboard")

# Load the model
@st.cache_resource
def load_model():
    model = load('./models/random_forest_model.joblib')
    return model

# Load data for display
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Input data form
def user_input_features():
    st.sidebar.header('User Input Features')

    age = st.sidebar.slider('Age', 18, 100, 30)
    balance = st.sidebar.slider('Balance', 0.0, 250000.0, 0.0)
    credit_score = st.sidebar.slider('Credit Score', 300, 900, 600)
    num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1)
    estimated_salary = st.sidebar.slider('Estimated Salary', 0.0, 200000.0, 50000.0)

    has_cr_card = st.sidebar.selectbox('Has Credit Card', options=[0, 1], index=1)
    is_active_member = st.sidebar.selectbox('Is Active Member', options=[0, 1], index=1)
    tenure = st.sidebar.slider('Tenure', 0, 10, 5)  # Adjust this range according to your data

    data = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Main app logic
model = load_model()

st.subheader('User Input Parameters')
input_data = user_input_features()
st.write(input_data)

# Ensure feature scaling is consistent with the model's training
scaler = StandardScaler()
sample_data = load_data('./data/Churn_Modelling.csv')
features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
sample_data = sample_data[features]
scaler.fit(sample_data)

# Predict
if st.button('Predict'):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')

    st.subheader('Prediction Probability')
    st.write(f'Probability of churn: {prediction_proba[0][1]:.2f}')
    st.write(f'Probability of not churning: {prediction_proba[0][0]:.2f}')

st.subheader('Sample Data')
st.write(load_data('./data/Churn_Modelling.csv').head())
