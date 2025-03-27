import pandas as pd
import streamlit as st

@st.cache_data
def get_data_cleaned():
    df = pd.read_csv("Telco-Customer-Churn.csv")

    #here we changed the TotalCharges datatype
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows where "TotalCharges" is NaN (you can adjust to drop for other columns if needed)
    df.dropna(subset=["TotalCharges"], inplace=True)

    #we dont need the customerID column so we can just drop it.
    df = df.drop("customerID", axis=1)

    #Here we change our churn column to True or False
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    #now lets do it to the rest of our df
    #This changes our categorical columns to True or False
    df_encoded = pd.get_dummies(df, drop_first=True)

    return df_encoded

def main():

    cleaned_df = get_data_cleaned()
    print("Data after encoding (head):")
    print(cleaned_df.head())
    print("Shape:", cleaned_df.shape)

if __name__ == "__main__":
    main()

    


