import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def predict_cancer(model, input_data):
    return model.predict(input_data)

def main():
    st.title("Cancer Detection Chatbot")

    st.write("I will ask you some questions to assess your cancer risk.")
    age = st.number_input('How old are you?', min_value=1, max_value=120, step=1)
    weight = st.number_input('What is your current weight in kg?', min_value=1, max_value=300, step=1)
    smoking_years = st.number_input('What is your smoking history (in years)?', min_value=0, max_value=100, step=1)
    diabetic = st.number_input('Are you diabetic?(1=Yes, 0=No)', min_value=0, max_value=1, step=1)
    cholesterol = st.number_input('What is your cholesterol level (in mg/dL)?', min_value=50, max_value=400, step=1)
    blood_pressure=st.number.input("What is your avg blood pressure level(in numbers)?", min_value=20, max_value=300, step=1)

    # Ensure these column names match your model's features
    features = pd.DataFrame([[age, weight, smoking_years, diabetic, cholesterol]], 
                            columns=['age', 'weight', 'smoking_years', 'diabetic', 'cholesterol', blood_pressure])

    if st.button('Predict'):
        # Load the trained model from a file
        model = joblib.load('model.pkl')
        prediction = predict_cancer(model, features)
        if prediction == 0:
            st.write("The prediction is B (Benign).")
        else:
            st.write("The prediction is M (Malignant).")

if __name__ == '__main__':
    main()
