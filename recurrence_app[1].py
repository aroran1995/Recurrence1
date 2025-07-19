import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# Load and prepare data from Excel
@st.cache_data
def load_data():
    file_path = "Updated File Feb 9 (1).xlsx"
    df = pd.read_excel(file_path, sheet_name='in')
    df = df[df['Volume prior to resolution'] <= df['Volume prior to resolution'].quantile(0.99)]  # remove outliers
    df['ln_volume'] = np.log(df['Volume prior to resolution'] + 1)
    df['ln_time'] = np.log(df['Time elapsed before resolution (number form)'] + 1)
    return df

@st.cache_data
def train_model(df):
    X = df[['ln_volume', 'ln_time']]
    y = df['Recurrent air leak >10min (0=no, 1=yes)']
    model = LogisticRegression().fit(X, y)
    return model

@st.cache_data
def bootstrap_predictions(df, ln_volume, ln_time, n_iterations=1000):
    preds = []
    for _ in range(n_iterations):
        boot_df = resample(df)
        X_boot = boot_df[['ln_volume', 'ln_time']]
        y_boot = boot_df['Recurrent air leak >10min (0=no, 1=yes)']
        model = LogisticRegression().fit(X_boot, y_boot)
        prob = model.predict_proba([[ln_volume, ln_time]])[0][1]
        preds.append(prob)
    return np.percentile(preds, [2.5, 50, 97.5])

# Load data and model
df = load_data()
model = train_model(df)

# --- Streamlit UI ---
st.set_page_config(page_title="Recurrent Air Leak Risk Calculator")
st.title("Recurrent Air Leak Risk Calculator")
st.markdown("""
Enter clinical values below to estimate the risk of a recurrent air leak based on volume drained and time to resolution.
""")

# Inputs
volume_input = st.number_input("Volume Drained (mL):", min_value=0, max_value=500000, value=1000)
time_input = st.number_input("Time Elapsed Before Resolution (minutes):", min_value=0, max_value=10000, value=600)

if volume_input > 0 and time_input > 0:
    ln_volume = np.log(volume_input + 1)
    ln_time = np.log(time_input + 1)
    input_features = np.array([[ln_volume, ln_time]])
    point_estimate = model.predict_proba(input_features)[0][1]

    # Bootstrap confidence interval
    ci_low, median, ci_high = bootstrap_predictions(df, ln_volume, ln_time)

    st.subheader("Predicted Probability of Recurrence")
    st.metric(label="Point Estimate", value=f"{point_estimate:.2%}")
    st.write(f"95% Confidence Interval: **{ci_low:.2%}** to **{ci_high:.2%}**")

    st.caption("Model trained on anonymized clinical data using logistic regression with 1,000 bootstrap iterations.")
else:
    st.warning("Please enter both volume and time greater than zero.")
