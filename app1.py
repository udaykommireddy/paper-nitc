import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from lightgbm import LGBMClassifier

# Streamlit Title
st.title("Antibiotic Usage Prediction (AB_used)")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'AB_used' in df.columns and 'sample-name' in df.columns:
        X = df.drop(['AB_used', 'sample-name'], axis=1)
        y = df['AB_used']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Preprocessing pipeline
        preprocess_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])

        # Define full pipeline with model
        final_pipeline = Pipeline([
            ('preprocessor', preprocess_pipeline),
            ('classifier', LGBMClassifier(random_state=42, n_estimators=200, num_leaves=31, learning_rate=0.1))
        ])

        # Train model
        final_pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = final_pipeline.predict(X_test)
        y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        # Display metrics
        st.subheader("Model Evaluation")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**AUC Score:** {auc:.4f}")
        st.write(f"**Sensitivity (Recall):** {sensitivity:.4f}")
        st.write(f"**Specificity:** {specificity:.4f}")

        # SHAP
        st.subheader("SHAP Explainability")

        # Only compute SHAP on training data (transformed)
        X_train_transformed = preprocess_pipeline.transform(X_train)
        explainer = shap.TreeExplainer(final_pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(X_train_transformed)

        st.write("SHAP Summary Plot:")
        plt.figure()
        shap.summary_plot(shap_values, X_train_transformed, feature_names=X.columns, show=False)
        st.pyplot(plt.gcf())

        # SHAP Dependence plot
        feature_to_plot = st.selectbox("Select feature for SHAP Dependence Plot", X.columns)
        st.write(f"SHAP Dependence Plot for: {feature_to_plot}")
        plt.figure()
        shap.dependence_plot(feature_to_plot, shap_values, X_train_transformed, feature_names=X.columns, show=False)
        st.pyplot(plt.gcf())

    else:
        st.error("Uploaded CSV must contain 'AB_used' and 'sample-name' columns.")
else:
    st.info("Please upload a CSV file.")
