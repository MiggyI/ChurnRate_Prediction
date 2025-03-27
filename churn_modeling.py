import streamlit as st
import seaborn as sns
#used for plots with less code
import matplotlib.pyplot as plt
#used for displaying plots from seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#used for all the machiene learning and modeling
from data_cleaning import get_data_cleaned
#uses our other file to clean our dataframe. 



# Cache the training process as a resource
@st.cache_resource
def train_model(max_iter):
    # Load cleaned data
    df = get_data_cleaned()
    
    # Split features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    # Split the data into training and testing sets:
    # - test_size=0.20 means 20% of the data is used for testing.
    # - random_state=42 sets a seed for reproducibility (42 is arbitrary; you can use any integer).
    # - stratify=y ensures the same proportion of churn classes in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Build and train the model with the given max_iter parameter
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Collect evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }
    return model, metrics, X_test, y_test

#here are the churn distribution
def plot_churn_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax)
    ax.set_title("Churn Distribution")
    st.pyplot(fig)

#Here we want to show the distrbution of monthly charges
def plot_monthly_charges_distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="MonthlyCharges", kde=True, ax=ax)
    ax.set_title("Distribution of Monthly Charges")
    st.pyplot(fig)
