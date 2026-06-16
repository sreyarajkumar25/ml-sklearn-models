import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler

# --- Page Setup ---
st.set_page_config(page_title="ML Model Predictor", layout="wide")
st.title("🎯 Machine Learning Model Predictor")
st.markdown("Enter your input values and get predictions from different ML models!")

# --- Sidebar: Model Selection ---
st.sidebar.header("⚙️ Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ["Linear Regression", "Logistic Regression", "SVM", "Random Forest", "XGBoost"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Enter values and click 'Predict' to see the output")

# --- Function to Load/Train Model ---
@st.cache_resource
def load_or_train_model(model_name):
    """Load existing model or train a new one with sample data."""
    
    # Try to load saved model first (if it exists)
    model_file = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    
    # For demo, we'll train on synthetic data
    # You can replace this with loading your actual trained models
    
    if model_name == "Linear Regression":
        X, y = make_regression(n_samples=500, n_features=2, noise=10, random_state=42)
        model = LinearRegression()
        feature_names = ["Feature 1", "Feature 2"]
        
    elif model_name == "Logistic Regression":
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)
        model = LogisticRegression()
        feature_names = ["Feature 1", "Feature 2"]
        
    elif model_name == "SVM":
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)
        model = SVC(kernel='rbf', random_state=42)
        feature_names = ["Feature 1", "Feature 2"]
        
    elif model_name == "Random Forest":
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        feature_names = ["Feature 1", "Feature 2"]
        
    else:  # XGBoost
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)
        model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        feature_names = ["Feature 1", "Feature 2"]
    
    # Train the model
    model.fit(X, y)
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    return model, scaler, feature_names

# --- Load Model ---
model, scaler, feature_names = load_or_train_model(model_choice)

# --- Main Area: User Input ---
st.header(f"📊 {model_choice}")

# Create input fields based on number of features
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Features")
    
    # Create input boxes for each feature
    input_values = []
    for i, feature in enumerate(feature_names):
        value = st.number_input(
            f"{feature}:", 
            value=0.0, 
            step=0.1,
            key=f"input_{i}"
        )
        input_values.append(value)

# --- Prediction Button ---
st.markdown("---")
predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

# --- Display Results ---
if predict_button:
    # Convert input to numpy array and scale
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Display results
    st.markdown("---")
    st.header("📈 Prediction Result")
    
    # Create columns for better display
    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
    
    with result_col2:
        # Check if it's regression or classification
        if model_choice == "Linear Regression":
            # Regression output - continuous value
            st.metric(
                label="Predicted Value", 
                value=f"{prediction[0]:.2f}",
                delta="Continuous Output"
            )
            st.success(f"✅ The predicted value is **{prediction[0]:.2f}**")
            
        else:
            # Classification output - class label
            st.metric(
                label="Predicted Class", 
                value=f"Class {int(prediction[0])}",
                delta="Classification Output"
            )
            
            # Show probability if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled)[0]
                st.write("**Class Probabilities:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Class 0", f"{proba[0]*100:.1f}%")
                with col_b:
                    st.metric("Class 1", f"{proba[1]*100:.1f}%")
            
            st.success(f"✅ The prediction is **Class {int(prediction[0])}**")
    
    # Show input values for reference
    st.markdown("---")
    with st.expander("📝 View Input Values"):
        input_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": input_values
        })
        st.dataframe(input_df, use_container_width=True)

# --- Footer with Instructions ---
st.markdown("---")
st.caption("💡 Tip: Adjust the input values and click 'Predict' to see how the model responds!")
