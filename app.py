from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_regresi_advertising.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler_advertising.pkl"
DATASET_PATH = BASE_DIR / "data" / "advertising.csv"


@st.cache_data
def load_dataset():
    if not DATASET_PATH.is_file():
        return None
    return pd.read_csv(DATASET_PATH)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
        raise FileNotFoundError("Model files are missing from the model/ directory.")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


st.set_page_config(
    page_title="Advertising sales predictor",
    page_icon=":material/analytics:",
    layout="centered",
)
st.title("Advertising sales predictor")
st.caption("Estimate sales from a TV and Radio advertising budget.")

st.info(
    "This app uses a machine learning model trained on advertising data. "
    "Enter the budget for one campaign period to estimate the expected sales. "
    "Budget values are in US dollars, matching the training dataset."
)

try:
    model, scaler = load_artifacts()
except (FileNotFoundError, OSError, ValueError, ModuleNotFoundError) as error:
    st.error(f"Unable to load the prediction model: {error}")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Campaign budget")
    st.caption("Enter the advertising budget for one campaign period.")
    tv = st.number_input(
        "TV advertising budget (USD)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Budget allocated to TV advertising.",
    )
    radio = st.number_input(
        "Radio advertising budget (USD)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Budget allocated to Radio advertising.",
    )
    submitted = st.form_submit_button(
        "Predict expected sales",
        type="primary",
        width="stretch",
    )

if submitted:
    features = [[tv, radio]]
    try:
        prediction = float(model.predict(scaler.transform(features))[0])
    except (ValueError, TypeError, AttributeError) as error:
        st.error(f"Prediction failed. Check the input values and model format: {error}")
    else:
        st.subheader("Prediction result")
        st.metric(
            label="Estimated sales",
            value=f"${prediction:,.2f}",
            help="Estimated sales output for the given advertising budget.",
        )
        st.caption(
            f"Result based on a budget of **${tv:,.2f}** for TV and **${radio:,.2f}** for Radio."
        )
        st.success("Prediction generated successfully.")

with st.expander("Explore the training dataset"):
    dataset = load_dataset()
    if dataset is None:
        st.warning("The local dataset preview is unavailable.")
    else:
        st.write(
            "This sample shows the TV, Radio, and Newspaper advertising budgets "
            "and the recorded Sales used for model development."
        )
        st.dataframe(dataset.head(10), width="stretch", hide_index=True)
        st.caption(
            f"Showing 10 of {len(dataset):,} records. "
            "Source: Kaggle Advertising Dataset."
        )
        st.link_button(
            "Open the original Kaggle dataset",
            "https://www.kaggle.com/datasets/ashydv/advertising-dataset",
            width="stretch",
        )
