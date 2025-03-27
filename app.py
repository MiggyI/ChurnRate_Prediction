# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from churn_modeling import train_model, plot_churn_distribution, plot_monthly_charges_distribution
from data_cleaning import get_data_cleaned


def main():
    st.title("Dynamic Churn Prediction App")
    
    st.sidebar.header("Model Settings")
    # Allow the user to adjust the maximum iterations for model training
    max_iter = st.sidebar.slider("Max Iterations", min_value=500, max_value=5000, value=3000, step=100)
    
    # Optionally, show raw cleaned data
    if st.sidebar.checkbox("Show Raw Data"):
        df = get_data_cleaned()
        st.write("Cleaned Data", df.head())
    
    # Load cleaned data
    df = get_data_cleaned()
    
    st.header("Data Visualizations")
    # Display the churn distribution plot
    plot_churn_distribution(df)
    # Display the monthly charges distribution plot
    plot_monthly_charges_distribution(df)
    
    st.header("Model Training and Evaluation")
    # Train the model with the chosen max_iter parameter; returns model, metrics, X_test, and y_test
    model, metrics, X_test, y_test = train_model(max_iter)
    
    st.subheader("Evaluation Metrics")
    st.write("**Accuracy:**", metrics["accuracy"])
    st.write("**ROC AUC Score:**", metrics["roc_auc"])
    
    st.subheader("Classification Report")
    report_str = classification_report(y_test, model.predict(X_test))
    st.text(report_str)
    
    st.subheader("Confusion Matrix")
    st.write(metrics["confusion_matrix"])
    
    st.header("Make a Prediction")
    st.write("Enter customer details below:")
    # Collect user input for two features (you can add more if needed)
    tenure = st.number_input("Tenure (months)", min_value=0, value=12,max_value=120)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0,max_value=200.0)
    
    # Instead of a row of zeros, create a default row using the mean values from X_test
    default_values = X_test.mean().to_dict()
    input_data = pd.DataFrame([default_values])
    
    # Override the default values with the user's inputs for specific features
    if "tenure" in input_data.columns:
        input_data.at[0, "tenure"] = tenure
    if "MonthlyCharges" in input_data.columns:
        input_data.at[0, "MonthlyCharges"] = monthly_charges
    
    # When the user clicks the prediction button, make a prediction using the model
    if st.button("Predict Churn"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0, 1]
        if prediction[0] == 1:
            st.write(f"Prediction: Customer is likely to churn (Probability: {prediction_proba:.2f})")
        else:
            st.write(f"Prediction: Customer is not likely to churn (Probability: {1 - prediction_proba:.2f})")

if __name__ == "__main__":
    main()

#Press Control C to stop the website running on the termnial
#Run Streamlit run app.py to get a website