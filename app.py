import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import math
import io

# Set page config
st.set_page_config(
    page_title="Mainframe Price Predictor",
    page_icon="🖥️",
    layout="wide"
)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []

def load_and_clean_data(df):
    """Clean and prepare the dataset"""
    # Make a copy to preserve original
    df_clean = df.copy()
    
    # Handle the target column (Quoted Price)
    if 'Quoted' in df_clean.columns:
        df_clean['Quoted Price'] = df_clean['Quoted'].replace('[\$,]', '', regex=True).str.strip()
        df_clean['Quoted Price'] = pd.to_numeric(df_clean['Quoted Price'], errors='coerce')
        df_clean.drop(columns=['Quoted'], inplace=True)
    
    # Convert numeric columns
    numeric_cols = ['Active MIPS', 'Active MSU', 'MAX CP', 'Memory', 'Crypto', 'ICF', 'IFL', 'ziip', 'Drawers']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def train_model(df):
    """Train the Random Forest model"""
    # Separate features and target
    X = df.drop(columns=['Quoted Price'])
    y = df['Quoted Price']
    
    # Drop rows with NaN in target
    nan_mask = y.notna()
    X = X[nan_mask]
    y = y[nan_mask]
    
    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit preprocessor and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_processed)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, preprocessor, X.columns.tolist(), rmse, r2

# App title and description
st.title("🖥️ Mainframe Quoted Price Prediction App")
st.markdown("Upload your mainframe dataset and predict quoted prices based on system specifications.")

# Sidebar for model training
st.sidebar.header("Model Training")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show data preview
        with st.expander("Data Preview"):
            st.dataframe(df.head())
        
        # Clean data
        df_clean = load_and_clean_data(df)
        
        # Train model button
        if st.sidebar.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    model, preprocessor, feature_names, rmse, r2 = train_model(df_clean)
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.preprocessor = preprocessor
                    st.session_state.feature_names = feature_names
                    st.session_state.model_trained = True
                    
                    st.sidebar.success("Model trained successfully!")
                    st.sidebar.metric("RMSE", f"{rmse:,.2f}")
                    st.sidebar.metric("R² Score", f"{r2:.3f}")
                    
                except Exception as e:
                    st.sidebar.error(f"Error training model: {str(e)}")
        
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")

# Main prediction interface
if st.session_state.model_trained:
    st.header("Make Predictions")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Configuration")
            
            # Default values for common features
            feature_inputs = {}
            
            # Common categorical features
            if 'TYPE' in st.session_state.feature_names:
                feature_inputs['TYPE'] = st.selectbox("Type", ["Net New", "MO", "Other"], key="type")
            
            if 'Business' in st.session_state.feature_names:
                feature_inputs['Business'] = st.selectbox("Business", ["1", "2"], key="business")
            
            # Numeric features
            numeric_features = ['Active MIPS', 'Active MSU', 'MAX CP', 'Memory', 'Crypto', 'ICF', 'IFL', 'ziip', 'Drawers']
            
            for feature in numeric_features:
                if feature in st.session_state.feature_names:
                    if feature in ['Active MIPS', 'Active MSU', 'MAX CP', 'Crypto', 'ICF', 'IFL', 'ziip', 'Drawers']:
                        feature_inputs[feature] = st.number_input(f"{feature}", min_value=0.0, step=1.0, key=feature.lower())
                    else:
                        feature_inputs[feature] = st.number_input(f"{feature}", min_value=0, step=1, key=feature.lower())
        
        with col2:
            st.subheader("Additional Features")
            
            # Handle any remaining features
            remaining_features = [f for f in st.session_state.feature_names if f not in feature_inputs.keys()]
            
            for feature in remaining_features:
                if feature in st.session_state.feature_names:
                    # Try to guess the input type
                    feature_inputs[feature] = st.number_input(f"{feature}", min_value=0.0, step=1.0, key=f"remaining_{feature}")
        
        # Predict button
        predict_button = st.form_submit_button("Predict Quoted Price", type="primary")
        
        if predict_button:
            try:
                # Create input DataFrame with all required features
                input_data = {}
                for feature in st.session_state.feature_names:
                    input_data[feature] = feature_inputs.get(feature, 0)
                
                input_df = pd.DataFrame([input_data])
                
                # Preprocess input
                input_processed = st.session_state.preprocessor.transform(input_df)
                
                # Make prediction
                prediction = st.session_state.model.predict(input_processed)[0]
                
                # Display result
                st.success(f"## Estimated Quoted Price: ${prediction:,.2f}")
                
                # Show feature importance
                feature_importance = st.session_state.model.feature_importances_
                if len(feature_importance) > 0:
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': [f"Feature_{i}" for i in range(len(feature_importance))],
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    st.bar_chart(importance_df.set_index('Feature')['Importance'])
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please ensure all required features are provided correctly.")

else:
    st.info("👆 Please upload a CSV file and train the model using the sidebar to start making predictions.")
    
    # Show example of expected data format
    st.subheader("Expected Data Format")
    example_data = pd.DataFrame({
        'TYPE': ['Net New', 'MO', 'Other'],
        'Business': ['1', '2', '1'],
        'Active MIPS': [100, 150, 200],
        'Active MSU': [50, 75, 100],
        'MAX CP': [4, 6, 8],
        'Memory': [1024, 2048, 4096],
        'Crypto': [1, 2, 0],
        'ICF': [0, 1, 2],
        'IFL': [2, 4, 6],
        'ziip': [1, 2, 3],
        'Drawers': [10, 15, 20],
        'Quoted': ['$100,000', '$150,000', '$200,000']
    })
    
    st.dataframe(example_data)
    st.caption("Note: The 'Quoted' column should contain the target prices you want to predict.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Machine Learning Model: Random Forest Regressor")