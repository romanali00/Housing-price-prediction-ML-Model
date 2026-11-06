import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page Configuration
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
sns.set_theme(style="whitegrid")

# Load Models & Artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_ridge_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except Exception as e:
        return None, None, None
@st.cache_data
def load_data():
    try:
        # Double '.csv' ki jagah single 'data.csv' use karein
        return pd.read_csv('data.csv')
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

model, scaler, model_columns = load_artifacts()
df = load_data()

# Navigation Sidebar
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Price Prediction", "EDA & Visualizations"])
st.sidebar.divider()

# --- PAGE 1: PRICE PREDICTION ---
if page == "Price Prediction":
    st.title("🏠 House Price Prediction Dashboard")
    st.write("Apne ghar ke features enter karein aur market price estimate karein.")
    st.divider()

    st.sidebar.header("📋 Input House Features")
    living_sqft = st.sidebar.number_input("Living Area (sqft)", min_value=300, max_value=10000, value=2000, step=50)
    lot_sqft = st.sidebar.number_input("Lot Area (sqft)", min_value=500, max_value=50000, value=5000, step=100)
    basement_sqft = st.sidebar.number_input("Basement Area (sqft)", min_value=0, max_value=5000, value=0, step=50)
    
    bedrooms = st.sidebar.slider("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.sidebar.slider("Bathrooms", min_value=1.0, max_value=8.0, value=2.0, step=0.5)
    
    yr_built = st.sidebar.number_input("Year Built", min_value=1900, max_value=2026, value=2000)
    yr_renovated = st.sidebar.number_input("Year Renovated (0 for none)", min_value=0, max_value=2026, value=0)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Selected Features Summary")
        user_input = {
            'Living_square_feet': living_sqft,
            'Lot_square_feet': lot_sqft,
            'Basement_square_feet': basement_sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'Year_Built': yr_built,
            'Year_Renovated': yr_renovated,
            'Sale_year': 2024,
            'Sale_month': 1,
            'Sale_day': 1
        }
        st.dataframe(pd.DataFrame([user_input]), use_container_width=True)

    with col2:
        st.subheader("🎯 Calculate Price")
        if st.button("Predict Price 🚀", use_container_width=True):
            if model is not None and scaler is not None and model_columns is not None:
                try:
                    full_input = pd.DataFrame(0, index=[0], columns=model_columns)
                    for col, val in user_input.items():
                        if col in full_input.columns:
                            full_input[col] = val
                    
                    input_scaled = scaler.transform(full_input)
                    prediction = model.predict(input_scaled)[0]
                    st.success(f"### Estimated Price:\n# **${prediction:,.2f}**")
                except Exception as ex:
                    st.error(f"Prediction Error: {ex}")
            else:
                st.error(".pkl model files load nahi ho sakin.")

# --- PAGE 2: EDA & VISUALIZATIONS ---
elif page == "EDA & Visualizations":
    st.title("📊 Exploratory Data Analysis & Visualizations")
    st.divider()

    if df is not None:
        numeric_cols = list(df.select_dtypes(include=['float64', 'int64', 'int32']).columns)
        numeric_cols = [c for c in numeric_cols if 'unnamed' not in c.lower()]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Feature Distribution")
            selected_dist_col = st.selectbox("Select Column for Distribution:", numeric_cols, index=0)
            
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(df[selected_dist_col].dropna(), kde=True, color='royalblue', ax=ax1)
            ax1.set_xlabel(selected_dist_col)
            ax1.ticklabel_format(style='plain', axis='x')
            st.pyplot(fig1)

        with col2:
            st.subheader("2. Feature Scatter Plot")
            x_var = st.selectbox("Select X-Axis Feature:", numeric_cols, index=min(1, len(numeric_cols)-1))
            y_var = st.selectbox("Select Y-Axis Feature:", numeric_cols, index=0)
            
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x=x_var, y=y_var, color='darkorange', ax=ax2, alpha=0.5)
            ax2.ticklabel_format(style='plain', axis='y')
            st.pyplot(fig2)

        st.divider()

        st.subheader("3. Feature Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        numeric_df = df[numeric_cols]
        if not numeric_df.empty:
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3, cbar=True)
        st.pyplot(fig3)
    else:
        st.error("`cleaned_data.csv` load nahi ho saki. Make sure `cleaned_data.csv` file exact isi folder me save hai.")