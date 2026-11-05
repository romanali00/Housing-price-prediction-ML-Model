# Housing-price-prediction-ML-Model
End-to-end House Price Prediction app deployed with Streamlit, using Ridge Regression, Feature Scaling, and dynamic EDA visualizer tools.
# 🏠 House Price Prediction Dashboard

Interactive Streamlit ML app predicting house prices using Linear Regression, Ridge/RidgeCV, and Lasso/LassoCV with dynamic EDA visualizations.

---

## 🚀 Live Demo & Features

- **Price Prediction**: Enter custom housing features (Living Sqft, Lot Sqft, Bedrooms, Bathrooms, Year Built, etc.) to get an instant model estimate.
- **EDA & Visualizations**: Interactive distribution plots, feature scatter plots, and correlation heatmaps.
- **Machine Learning**: Uses Ridge Regression with normalized feature scaling via `StandardScaler`.

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.x
- **Framework**: Streamlit
- **ML & Data Analysis**: Scikit-Learn, Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib

---

## 📂 Project Structure

```text
├── app.py                     # Main Streamlit Dashboard Script
├── best_ridge_model.pkl       # Trained Ridge Regression Model
├── scaler.pkl                 # StandardScaler Artifact
├── model_columns.pkl          # Feature Column Order Metadata
├── data.csv                   # Cleaned Housing Dataset
├── README.md                  # Project Documentation
└── requirements.txt           # Project Dependencies

⚙️ How to Run Locally

pip install -r requirements.txt
streamlit run app.py

📊 Model Information
Algorithm: Ridge Regression

## 📊 Machine Learning Algorithms Evaluated

- **Linear Regression**: Baseline regression analysis.
- **Ridge & RidgeCV**: Regularized linear regression (L2) with automatic alpha tuning via Cross-Validation (Final Model selected).
- **Lasso & LassoCV**: Feature selection and regularized regression (L1) using Cross-Validation.
