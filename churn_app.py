import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Not used directly
# import pickle # Using joblib instead
import xgboost as xgb
import joblib
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image # Keep for potential future logo use
import io
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os # To check if files exist

# Set page configuration
st.set_page_config(
    page_title="Telco Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A; /* Dark Blue */
        text-align: center;
        padding-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB; /* Medium Blue */
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #DBEAFE; /* Light Blue */
        padding-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.4rem; /* Slightly smaller */
        color: #3B82F6; /* Lighter Blue */
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F8FAFC; /* Very Light Gray */
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    .insight-box {
        background-color: #EFF6FF; /* Very Light Blue */
        border-left: 5px solid #3B82F6; /* Lighter Blue */
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        color: #1F2937; /* **Explicitly set dark gray text color** */
    }
    /* Style links inside insight boxes */
    .insight-box a {
        color: #1D4ED8; /* Slightly darker blue for links */
        text-decoration: underline;
    }
    /* Style code/pre inside insight boxes */
     .insight-box pre, .insight-box code {
        background-color: #E5E7EB; /* Light gray background for code */
        color: #111827; /* Darker text for code */
        padding: 0.2em 0.4em;
        margin: 0;
        font-size: 85%;
        border-radius: 3px;
    }
    .prediction-box-churn {
        background-color: #FEE2E2; /* Light Red */
        border: 2px solid #EF4444; /* Red */
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-top: 1rem;
        color: #374151; /* **Set default dark text for content** */
    }
    .prediction-box-stay {
        background-color: #DCFCE7; /* Light Green */
        border: 2px solid #22C55E; /* Green */
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-top: 1rem;
        color: #374151; /* **Set default dark text for content** */
    }
     /* Specific styling for H1 inside prediction boxes */
    .prediction-box-churn h1 {
        color: #DC2626; /* Specific Dark Red for Churn H1 */
        margin-bottom: 0.5rem; /* Adjust spacing */
    }
    .prediction-box-stay h1 {
        color: #16A34A; /* Specific Dark Green for Stay H1 */
        margin-bottom: 0.5rem; /* Adjust spacing */
    }
     /* Ensure H3 and P tags also use the default dark color */
     .prediction-box-churn h3, .prediction-box-churn p,
     .prediction-box-stay h3, .prediction-box-stay p {
         color: inherit; /* Inherit the dark color set on the parent box */
         margin-top: 0.5rem;
         margin-bottom: 0.5rem;
     }
    /* Improve table styling */
    .stDataFrame table { /* Target Streamlit's DataFrame tables */
        width: 100%;
        border-collapse: collapse;
    }
    .stDataFrame th, .stDataFrame td {
        padding: 8px 12px;
        border: 1px solid #e2e8f0; /* Light Gray Border */
        text-align: left;
    }
    .stDataFrame th {
        background-color: #f1f5f9; /* Lighter Gray Header */
        font-weight: bold;
    }
    /* Style Plotly charts */
    .plotly-chart {
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 5px; /* Add slight padding around chart */
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

# Cache data loading to improve performance
@st.cache_data
def load_data(raw_file='churn_raw_data.csv', processed_file='churn_df_su.csv'):
    """Load and prepare the Telco dataset from CSV files"""
    raw_df = pd.DataFrame()
    processed_df = pd.DataFrame()
    data_loaded = False

    try:
        if os.path.exists(raw_file):
            raw_df = pd.read_csv(raw_file)
            print(f"Raw data '{raw_file}' loaded successfully. Shape: {raw_df.shape}")
            data_loaded = True

            # --- Data Cleaning/Consistency for Raw Data ---
            # Standardize Churn column to 0/1 if it exists
            if 'Churn' in raw_df.columns:
                if raw_df['Churn'].dtype == 'object':
                    # Handle potential variations like 'Yes'/'No', '1'/'0'
                    raw_df['Churn'] = raw_df['Churn'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0}).fillna(0).astype(int)
                elif pd.api.types.is_numeric_dtype(raw_df['Churn']):
                    # Ensure it's strictly 0 or 1 if already numeric
                    raw_df['Churn'] = raw_df['Churn'].apply(lambda x: 1 if x == 1 else 0)
                else:
                     # If it's some other unexpected type, try coercing, default to 0
                     raw_df['Churn'] = pd.to_numeric(raw_df['Churn'], errors='coerce').fillna(0).astype(int)
                     st.warning("Raw data 'Churn' column had unexpected format, attempted conversion.", icon="⚠️")

                # Create ChurnFlag if needed (often useful for consistency)
                if 'ChurnFlag' not in raw_df.columns:
                    raw_df['ChurnFlag'] = raw_df['Churn']
            else:
                print("Warning: 'Churn' column not found in raw data.")
                # Only show warning in app if essential functionality depends on it later
                # st.warning("The raw data CSV is missing the 'Churn' column. Some visualizations might be affected.", icon="⚠️")

            # Attempt to convert TotalCharges to numeric, coercing errors (often spaces)
            if 'TotalCharges' in raw_df.columns:
                raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')
                # Fill missing TotalCharges (often where tenure is 0) with 0
                if raw_df['TotalCharges'].isnull().any():
                    raw_df['TotalCharges'] = raw_df['TotalCharges'].fillna(0)
                    print("Filled missing TotalCharges values with 0.")
            else:
                 print("Warning: 'TotalCharges' column not found in raw data.")

            # Ensure tenure is numeric
            if 'tenure' in raw_df.columns:
                raw_df['tenure'] = pd.to_numeric(raw_df['tenure'], errors='coerce').fillna(0)
            else:
                 print("Warning: 'tenure' column not found in raw data.")

            # Ensure MonthlyCharges is numeric
            if 'MonthlyCharges' in raw_df.columns:
                raw_df['MonthlyCharges'] = pd.to_numeric(raw_df['MonthlyCharges'], errors='coerce').fillna(0)
            else:
                 print("Warning: 'MonthlyCharges' column not found in raw data.")

        else:
            st.error(f"Raw data file not found: '{raw_file}'. Please make sure it's in the same directory as the script.", icon="🚨")

        if os.path.exists(processed_file):
            processed_df = pd.read_csv(processed_file)
            print(f"Processed data '{processed_file}' loaded successfully. Shape: {processed_df.shape}")
            data_loaded = True
        else:
            st.error(f"Processed data file not found: '{processed_file}'. This file is needed for model predictions and some visualizations.", icon="🚨")

        if not data_loaded:
            st.error("No data files could be loaded. The application cannot continue.", icon="🚨")
            st.stop() # Stop execution if no data is loaded

        print("Raw DataFrame columns:", raw_df.columns.tolist() if not raw_df.empty else "N/A")
        print("Processed DataFrame columns:", processed_df.columns.tolist() if not processed_df.empty else "N/A")

        return raw_df, processed_df

    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}", icon="🚨")
        # Return empty dataframes to prevent app from crashing completely
        return pd.DataFrame(), pd.DataFrame()

# Cache model loading
@st.cache_resource
def load_models(xgb_file='xgboost_model.joblib', dt_file='decision_tree_model.joblib'):
    """Load pre-trained models"""
    models = {'xgboost': None, 'decision_tree': None}
    model_features = {'xgboost': None, 'decision_tree': None}

    # Load XGBoost model
    try:
        if os.path.exists(xgb_file):
            xgb_model = joblib.load(xgb_file)
            print("XGBoost model loaded successfully.")
            models['xgboost'] = xgb_model
            # Try different ways to get feature names
            if hasattr(xgb_model, 'feature_names_in_'):
                model_features['xgboost'] = xgb_model.feature_names_in_.tolist()
                print("XGBoost model features (from feature_names_in_):", model_features['xgboost'])
            elif hasattr(xgb_model, 'get_booster') and hasattr(xgb_model.get_booster(), 'feature_names') and xgb_model.get_booster().feature_names:
                 model_features['xgboost'] = xgb_model.get_booster().feature_names
                 print("XGBoost model features (from booster):", model_features['xgboost'])
            else:
                 print("Warning: Could not automatically determine XGBoost feature names from the loaded model.")
                 # Attempt to infer from booster attributes if they exist
                 try:
                     if hasattr(xgb_model, '_features_internal'): # Older xgboost versions might store it here
                         model_features['xgboost'] = xgb_model._features_internal
                         print("XGBoost model features (from _features_internal):", model_features['xgboost'])
                     else:
                        st.warning("Could not determine feature names from the XGBoost model file. Predictions might be inaccurate if input data columns don't match the training data.", icon="⚠️")
                 except: # Catch any errors trying backup methods
                    st.warning("Could not determine feature names from the XGBoost model file. Predictions might be inaccurate if input data columns don't match the training data.", icon="⚠️")


        else:
            print(f"XGBoost model file not found: '{xgb_file}'. Prediction functionality will be limited.")
            st.warning(f"XGBoost model file '{xgb_file}' not found. Churn prediction will not be available.", icon="⚠️")

    except Exception as e:
        print(f"Error loading XGBoost model: {str(e)}")
        st.error(f"Error loading the XGBoost model from '{xgb_file}': {str(e)}", icon="🚨")

    # Load Decision Tree model
    try:
        if os.path.exists(dt_file):
            dt_model = joblib.load(dt_file)
            print("Decision Tree model loaded successfully.")
            models['decision_tree'] = dt_model
            if hasattr(dt_model, 'feature_names_in_'):
                model_features['decision_tree'] = dt_model.feature_names_in_.tolist()
                print("Decision Tree model features:", model_features['decision_tree'])
            else:
                print("Warning: Could not automatically determine Decision Tree feature names.")
                #st.warning("Could not determine feature names from the Decision Tree model file. Visualization might use generic names or fail if data columns don't match.", icon="⚠️")
        else:
             print(f"Decision Tree model file not found: '{dt_file}'. Visualization will be limited.")
             st.warning(f"Decision Tree model file '{dt_file}' not found. Decision Tree visualization will not be available.", icon="⚠️")

    except Exception as e:
        print(f"Error loading Decision Tree model: {str(e)}")
        st.error(f"Error loading the Decision Tree model from '{dt_file}': {str(e)}", icon="🚨")

    return models, model_features

def preprocess_input_data(input_df, target_features):
    """
    Preprocesses input DataFrame (e.g., from user form) to match model's expected features.
    Handles one-hot encoding and binary mapping based on target_features.

    Args:
        input_df (pd.DataFrame): DataFrame with raw user inputs (should be 1 row).
        target_features (list): List of feature names the model was trained on.

    Returns:
        pd.DataFrame: Processed DataFrame ready for prediction, or None if error.
    """
    if not target_features:
        st.error("Model feature names are not available. Cannot preprocess input data.", icon="🚨")
        return None
    if input_df.empty:
        st.error("Input data is empty. Cannot preprocess.", icon="🚨")
        return None

    try:
        processed_df = input_df.copy()
        print(f"\n--- Preprocessing Start ---")
        print(f"Original Input Columns: {processed_df.columns.tolist()}")
        print(f"Target Features Expected by Model: {target_features}")

        # --- Step 1: Identify columns needing processing ---
        object_cols_in_input = processed_df.select_dtypes(include=['object', 'category']).columns
        cols_to_encode = []      # For one-hot encoding
        cols_to_map_binary = [] # For direct Yes/No -> 1/0

        print(f"Object columns in input: {object_cols_in_input.tolist()}")

        for col in object_cols_in_input:
            is_one_hot_encoded_in_target = any(f.startswith(col + '_') for f in target_features)
            # Check if the *original* column name exists directly in target_features
            is_direct_in_target = col in target_features

            if is_one_hot_encoded_in_target:
                # If OHE versions exist in target, mark for encoding
                cols_to_encode.append(col)
                print(f"  Marking '{col}' for One-Hot Encoding (found '{col}_...' in target).")
                # Safety check: Ensure the original name isn't *also* in target features if OHE is expected
                if is_direct_in_target:
                     print(f"  WARNING: Original column '{col}' and OHE versions ('{col}_...') both seem expected by model? Prioritizing OHE.")
            elif is_direct_in_target:
                # If original name is directly in target AND it's object type, assume binary mapping
                unique_vals = set(processed_df[col].unique())
                if unique_vals.issubset({'Yes', 'No', None, np.nan}):
                    cols_to_map_binary.append(col)
                    print(f"  Marking '{col}' for Binary Mapping (found '{col}' directly in target, looks like Yes/No).")
                else:
                     print(f"  WARNING: Column '{col}' is object type, expected directly by model, but values ({unique_vals}) aren't Yes/No. Cannot map.")
            else:
                 print(f"  Column '{col}' is object type but neither its direct name nor OHE versions ('{col}_...') found in target features. It will likely be dropped during alignment.")


        # --- Step 2: Apply Binary Mapping ---
        if cols_to_map_binary:
             print(f"Applying binary mapping (Yes=1, No=0) to: {cols_to_map_binary}")
             for col in cols_to_map_binary:
                  processed_df[col] = processed_df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

        # --- Step 3: Apply One-Hot Encoding ---
        if cols_to_encode:
            print(f"Applying One-Hot Encoding to: {cols_to_encode}")
            # Using drop_first=False generally safer unless you know the model used drop_first=True
            processed_df = pd.get_dummies(processed_df, columns=cols_to_encode, drop_first=False, dtype=int) # Use dtype=int
            print(f"Columns after get_dummies: {processed_df.columns.tolist()}")

        # --- Step 4: Feature Alignment (Strict) ---
        print(f"Aligning data to target features strictly.")
        # Create DataFrame with exactly the target features and order, initialized to 0
        aligned_df = pd.DataFrame(0, index=processed_df.index, columns=target_features)

        # Get columns present in BOTH processed data AND target list
        common_cols = list(set(processed_df.columns) & set(target_features))
        print(f"Columns present in both processed data and target list: {common_cols}")

        # Copy data ONLY for common columns
        if common_cols:
            aligned_df[common_cols] = processed_df[common_cols]
        else:
             print("WARNING: No common columns found between processed input and target features after encoding/mapping!")


        # --- Step 5: Final Data Type Check & Conversion ---
        print("Ensuring final data types are numeric...")
        for col in aligned_df.columns: # Iterate through the FINAL aligned columns
            if not pd.api.types.is_numeric_dtype(aligned_df[col]):
                 # If not numeric, attempt conversion (e.g., bools to int)
                 try:
                      # Try converting to float first (more general), then potentially int if no decimals
                      aligned_df[col] = pd.to_numeric(aligned_df[col], errors='coerce').fillna(0)
                      # Optional: Convert to integer if no decimal part remains (cleaner for OHE)
                      # if aligned_df[col].astype(int).eq(aligned_df[col]).all():
                      #      aligned_df[col] = aligned_df[col].astype(int)
                      print(f"  Converted column '{col}' to numeric.")
                 except Exception as final_type_e:
                      print(f"  ERROR: Failed final conversion to numeric for column '{col}'. Type: {aligned_df[col].dtype}. Error: {final_type_e}")
                      st.error(f"Critical error: Could not convert aligned column '{col}' to numeric for the model.", icon="🚨")
                      return None # Fail preprocessing if final conversion fails

        # Final check for any non-numeric types remaining
        final_dtypes = aligned_df.dtypes
        non_numeric_final = final_dtypes[~final_dtypes.apply(pd.api.types.is_numeric_dtype)].index.tolist()

        print("\n--- Preprocessing Output ---")
        print(f"Final Aligned Columns: {aligned_df.columns.tolist()}")
        print("Final Data Types:")
        print(aligned_df.dtypes)

        if non_numeric_final:
             print(f"ERROR: Non-numeric columns remain after all steps: {non_numeric_final}")
             st.error(f"Preprocessing failed. Non-numeric columns remain: {non_numeric_final}. Check model's target features vs. input data.", icon="🚨")
             return None
        else:
             print("Preprocessing successful. All columns are numeric.")
             print("--------------------------\n")
             return aligned_df

    except Exception as e:
        st.error(f"Unexpected error during input data preprocessing: {str(e)}", icon="🚨")
        print(f"Preprocessing error details: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None


def predict_churn(customer_data_processed, model):
    """Predict churn probability for processed customer data"""
    if customer_data_processed is None or model is None:
        st.warning("Missing processed data or model for prediction.", icon="⚠️")
        return None, None
    try:
        # Ensure column order matches model expectation if possible (already handled by preprocess_input_data if target_features were available)
        # But as a failsafe:
        if hasattr(model, 'feature_names_in_'):
             customer_data_processed = customer_data_processed[model.feature_names_in_]
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
             customer_data_processed = customer_data_processed[model.get_booster().feature_names]

        # Make prediction
        churn_prob = model.predict_proba(customer_data_processed)[:, 1][0]
        # Use a standard threshold (can be adjusted based on model tuning/business needs)
        threshold = 0.5
        churn_prediction = 1 if churn_prob >= threshold else 0
        return churn_prediction, churn_prob

    except ValueError as ve:
         st.error(f"Prediction ValueError: {ve}. Often indicates a mismatch between input data features/types and model expectations.", icon="🚨")
         print(f"Prediction ValueError: {ve}")
         print("Data sent to model:")
         print(customer_data_processed)
         print("Data types:")
         print(customer_data_processed.dtypes)
         if hasattr(model, 'feature_names_in_'): print(f"Model expected features: {model.feature_names_in_}")
         return None, None
    except Exception as e:
        st.error(f"Prediction error: {str(e)}. Check logs for details.", icon="🚨")
        print(f"Prediction error details: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for displaying in HTML"""
    try:
        buffer = io.BytesIO()
        # Increase DPI for better quality in web display
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=200)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig) # Close the figure to free memory
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error converting figure to base64: {e}")
        if fig: plt.close(fig) # Ensure figure is closed even if error occurs
        return None

# Use caching for this potentially expensive operation
@st.cache_data
def make_churn_pct_and_count_df(df, max_levels=10):
    """
    Builds a tidy DataFrame of churn percentages AND counts for categorical features
    and binned numerical features (like tenure groups). Handles potential NaNs.
    """
    # Ensure Churn column exists and is numeric (0/1)
    if 'Churn' not in df.columns:
        print("make_churn_pct_and_count_df: 'Churn' column missing.")
        return pd.DataFrame(columns=['feature', 'level', 'churn_pct', 'count', 'churn_count'])

    df_copy = df.copy() # Work on a copy

    # Standardize Churn if not already done
    if not pd.api.types.is_numeric_dtype(df_copy['Churn']):
         try:
             df_copy['Churn'] = pd.to_numeric(df_copy['Churn'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0}), errors='coerce').fillna(0).astype(int)
         except Exception:
              print("make_churn_pct_and_count_df: Could not standardize Churn column.")
              return pd.DataFrame(columns=['feature', 'level', 'churn_pct', 'count', 'churn_count'])

    records = []
    churn_col = 'Churn' # Use the standardized name

    # Select columns except known non-feature/target columns
    cols_to_analyze = [col for col in df_copy.columns if col not in [churn_col, 'customerID', 'ChurnFlag', 'TotalCharges']] # Exclude TotalCharges for simple grouping

    for col in cols_to_analyze:
        # Skip if column is mostly unique values (like IDs or high-cardinality continuous) unless it's a known categorical/binned one
        is_likely_categorical = df_copy[col].dtype == 'object' or df_copy[col].nunique() <= max_levels
        is_known_binned = col in ['TenureGroup'] # Add other known binned columns here if needed

        if is_likely_categorical or is_known_binned:
            try:
                # Handle potential NaNs in grouping column gracefully by treating as a separate category
                # Convert NaNs to a specific string AFTER converting the column to string type
                feature_series = df_copy[col].astype(str).fillna('Missing')

                grouped = df_copy.groupby(feature_series)

                # Calculate counts and percentages
                counts = grouped.size()
                churn_counts = grouped[churn_col].sum()

                # Calculate churn percentages safely (handle division by zero)
                churn_pcts = (churn_counts / counts * 100).fillna(0)


                for lvl in counts.index:
                    records.append({
                        'feature': col,
                        'level': lvl, # Level is now guaranteed to be a string (or 'Missing')
                        'churn_pct': churn_pcts.get(lvl, 0), # Use .get for safety
                        'count': counts.get(lvl, 0),
                        'churn_count': churn_counts.get(lvl, 0)
                    })
            except Exception as e:
                print(f"Error grouping/calculating for feature '{col}': {e}") # Log error but continue
        # else: # Optional: Print skipped columns
            # print(f"Skipping column '{col}' for churn rate analysis (likely high cardinality or continuous).")


    if not records:
        print("make_churn_pct_and_count_df: No records generated.")
        return pd.DataFrame(columns=['feature', 'level', 'churn_pct', 'count', 'churn_count'])

    pct_df = pd.DataFrame.from_records(records)
    pct_df['feature'] = pct_df['feature'].astype(str)
    # 'level' is already string from the processing loop
    # print(f"Generated pct_df with {len(pct_df)} records.") # Debug print
    return pct_df

def plotly_churn_rates(df, feature, title_prefix="Churn Rate by"):
    """Create a Plotly visualization of churn rates by feature level with counts"""
    if feature not in df.columns:
        st.error(f"Feature '{feature}' not found in the dataset.", icon="🚨")
        return go.Figure()

    if 'Churn' not in df.columns:
         st.error(f"Required 'Churn' column not found in data for '{feature}' analysis.", icon="🚨")
         return go.Figure()

    max_lvls = 20 if feature in ['TenureGroup', 'PaymentMethod'] else 10
    feature_df_agg = make_churn_pct_and_count_df(df[[feature, 'Churn']].copy(), max_levels=max_lvls)
    feature_data = feature_df_agg[feature_df_agg['feature'] == feature].copy()

    if feature_data.empty:
        st.warning(f"No aggregated data available to display for feature '{feature}'. Check raw data and levels.", icon="⚠️")
        return go.Figure()

    # --- Sorting Logic --- (Keep as is)
    numeric_sort_success = False
    if feature == 'TenureGroup':
         tenure_order = ['0-12', '13-24', '25-36', '37-48', '49-60', '61+']
         try:
            feature_data['level'] = pd.Categorical(feature_data['level'], categories=tenure_order, ordered=True)
            feature_data = feature_data.sort_values('level')
            numeric_sort_success = True
         except Exception as e:
            print(f"Could not sort TenureGroup category: {e}")
    else:
        try:
            feature_data['level_numeric'] = pd.to_numeric(feature_data['level'])
            feature_data = feature_data.sort_values('level_numeric')
            numeric_sort_success = True
        except ValueError:
             pass
    if not numeric_sort_success:
        feature_data = feature_data.sort_values('churn_pct', ascending=False)
    # --- End Sorting Logic ---

    # --- Create figure using make_subplots ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bars for churn percentage (Primary Y-axis)
    fig.add_trace(
        go.Bar(
            x=feature_data['level'],
            y=feature_data['churn_pct'],
            name='Churn Rate (%)',
            marker_color='#3B82F6',
            text=[f"{pct:.1f}%" for pct in feature_data['churn_pct']],
            textposition='auto',
            hoverinfo='x+y+name',
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=False,
    )

    # Add line for count (Secondary Y-axis)
    fig.add_trace(
        go.Scatter(
            x=feature_data['level'],
            y=feature_data['count'],
            name='Customer Count',
            marker=dict(color='#1E3A8A'),
            mode='lines+markers',
            hoverinfo='x+y+name',
            hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
        ),
        secondary_y=True,
    )

    # --- Corrected fig.update_layout ---
    fig.update_layout(
        title=f'{title_prefix} {feature}',
        xaxis=dict(
             title=feature,
             type='category'
        ),
        yaxis=dict( # Primary Y-axis (Left) - CORRECTED
            title=dict(text='Churn Rate (%)', font=dict(color='#3B82F6')), # Nest font under title
            tickfont=dict(color='#3B82F6'),
            range=[0, max(50, feature_data['churn_pct'].max() * 1.1 if not feature_data['churn_pct'].empty else 50)],
            showgrid=False,
        ),
        yaxis2=dict( # Secondary Y-axis (Right) - CORRECTED
            title=dict(text='Customer Count', font=dict(color='#1E3A8A')), # Nest font under title
            tickfont=dict(color='#1E3A8A'),
            showgrid=True,
            range=[0, max(100, feature_data['count'].max() * 1.1 if not feature_data['count'].empty else 100)]
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='x unified',
        height=450,
        margin=dict(l=60, r=60, t=80, b=50)
    )
    # --- End Corrected fig.update_layout ---

    return fig

# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit app"""

    # Load data and models
    raw_df, processed_df = load_data() # Assign default empty dataframes if loading fails
    models, model_features = load_models()

    # # Check if we have essential data/models
    # Raw data is useful but maybe not strictly essential if processed exists?
    # Processed data is needed for DT features if model doesn't store them.
    # Models are needed for prediction/visualization pages.
    # Add checks within specific pages instead of stopping globally here.

    xgb_model = models['xgboost']
    dt_model = models['decision_tree']
    xgb_features = model_features['xgboost']
    dt_features = model_features['decision_tree']

    # --- Sidebar ---
    st.sidebar.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Navigation</h1>", unsafe_allow_html=True)
    # Add company logo (replace 'logo.png' with your actual logo file if available)
    # logo_path = 'logo.png'
    # if os.path.exists(logo_path):
    #      try:
    #          logo = Image.open(logo_path)
    #          st.sidebar.image(logo, width=150, use_column_width='auto') # Adjust width as needed
    #      except Exception as e:
    #           print(f"Error loading logo: {e}")
    #           st.sidebar.markdown("<div style='text-align: center; padding: 10px 0;'><i>[Your Logo Here]</i></div>", unsafe_allow_html=True)
    # else:
    #     st.sidebar.markdown("<div style='text-align: center; padding: 10px 0;'><i>[Your Logo Here]</i></div>", unsafe_allow_html=True)

    st.sidebar.markdown("---")

    page_options = ["Dashboard"]
    # Add pages conditionally based on available models/data
    if xgb_model and xgb_features:
         page_options.append("Churn Prediction")
    if not raw_df.empty: # Analysis needs raw data
        page_options.append("Churn Analysis")
    if dt_model: # DT Viz needs the model
        page_options.append("Decision Tree Viz")


    if len(page_options) == 1 and page_options[0] == "Dashboard":
         st.warning("Some features (Prediction, Analysis, DT Viz) are unavailable due to missing models or data.", icon="⚠️")

    page = st.sidebar.radio(
        "Select a Page",
        page_options,
        label_visibility="collapsed" # Hide the radio label itself
    )
    st.sidebar.markdown("---")
    st.sidebar.info("This app analyzes customer churn patterns and predicts potential churn using machine learning models.", icon="ℹ️")


    # --- Page Content ---

    # Dashboard Page
    if page == "Dashboard":
        st.markdown("<h1 class='main-header'>Telco Customer Churn Dashboard</h1>", unsafe_allow_html=True)

        if raw_df.empty:
             st.warning("Raw data (`churn_raw_data.csv`) is not available. Dashboard cannot be displayed.", icon="⚠️")
             st.stop() # Stop if no raw data for dashboard

        # Ensure necessary columns exist and are valid before proceeding
        required_dash_cols = {'Churn', 'tenure', 'MonthlyCharges'}
        missing_cols = required_dash_cols - set(raw_df.columns)
        invalid_cols = []
        for col in required_dash_cols.intersection(raw_df.columns):
             if not pd.api.types.is_numeric_dtype(raw_df[col]):
                 invalid_cols.append(col)

        if missing_cols:
             st.warning(f"Dashboard requires columns: {missing_cols}. Some metrics/charts may be missing.", icon="⚠️")
        if invalid_cols:
             st.warning(f"Dashboard columns {invalid_cols} have incorrect data types (should be numeric). Some metrics/charts may be affected.", icon="⚠️")


        dashboard_df = raw_df # Use raw data for dashboard interpretability

        # --- Overview Metrics ---
        st.markdown("<h2 class='sub-header'>Key Performance Indicators</h2>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        total_customers = len(dashboard_df)
        churn_count = 0
        churn_rate = 0.0
        avg_tenure = 0.0
        avg_monthly_charges = 0.0

        # Safely calculate metrics
        if 'Churn' in dashboard_df.columns and pd.api.types.is_numeric_dtype(dashboard_df['Churn']):
            churn_count = int(dashboard_df['Churn'].sum())
            if total_customers > 0:
                churn_rate = (churn_count / total_customers) * 100
        else:
            print("Churn data missing/invalid for KPI calculation.")

        if 'tenure' in dashboard_df.columns and pd.api.types.is_numeric_dtype(dashboard_df['tenure']):
             avg_tenure = dashboard_df['tenure'].mean()
        else:
             print("Tenure data missing/invalid for KPI calculation.")

        if 'MonthlyCharges' in dashboard_df.columns and pd.api.types.is_numeric_dtype(dashboard_df['MonthlyCharges']):
             avg_monthly_charges = dashboard_df['MonthlyCharges'].mean()
        else:
             print("MonthlyCharges data missing/invalid for KPI calculation.")

        # Display metrics
        with col1:
            #st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Customers", f"{total_customers:,}")
            #st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            #st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            # Show N/A if churn data is bad
            churn_display = f"{churn_count:,}" if 'Churn' in dashboard_df.columns and pd.api.types.is_numeric_dtype(dashboard_df['Churn']) else "N/A"
            rate_display = f"{churn_rate:.1f}% of total" if churn_display != "N/A" else ""
            st.metric("Churned Customers", churn_display, rate_display)
            #st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            #st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            tenure_display = f"{avg_tenure:.1f} months" if 'tenure' in dashboard_df.columns and pd.api.types.is_numeric_dtype(dashboard_df['tenure']) else "N/A"
            st.metric("Avg. Tenure", tenure_display)
            #st.markdown("</div>", unsafe_allow_html=True)
        with col4:
            #st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            charge_display = f"${avg_monthly_charges:.2f}" if 'MonthlyCharges' in dashboard_df.columns and pd.api.types.is_numeric_dtype(dashboard_df['MonthlyCharges']) else "N/A"
            st.metric("Avg. Monthly Charges", charge_display)
            #st.markdown("</div>", unsafe_allow_html=True)


        # --- Churn Distribution & Insights ---
        st.markdown("<h2 class='sub-header'>Churn Overview</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1]) # Give more space to the chart

        with col1:
            if 'Churn' in dashboard_df.columns and pd.api.types.is_numeric_dtype(dashboard_df['Churn']):
                # Create pie chart for churn distribution
                churn_dist_data = dashboard_df['Churn'].map({1: 'Churned', 0: 'Retained'}).value_counts().reset_index()
                churn_dist_data.columns = ['Churn Status', 'Count']

                fig_pie = px.pie(
                    churn_dist_data,
                    values='Count',
                    names='Churn Status',
                    title="Customer Churn Distribution",
                    color='Churn Status',
                    color_discrete_map={'Churned': '#EF4444', 'Retained': '#22C55E'}, # Red, Green
                    hole=0.3 # Donut chart style
                )
                fig_pie.update_traces(textposition='outside', textinfo='percent+label', pull=[0.05 if status == 'Churned' else 0 for status in churn_dist_data['Churn Status']]) # Pull churned slice
                fig_pie.update_layout(legend_title_text='Customer Status', height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("Churn data needed for distribution chart.", icon="⚠️")

        with col2:
            #st.markdown("<div class='insight-box' style='height: 400px; overflow-y: auto;'>", unsafe_allow_html=True) # Make box scrollable if content overflows
            st.markdown("<h4>Key Insights</h4>", unsafe_allow_html=True)
            if 'Churn' in dashboard_df.columns and pd.api.types.is_numeric_dtype(dashboard_df['Churn']):
                 st.markdown(f"• **{churn_rate:.1f}%** of customers churned.")
                 st.markdown(f"• **{total_customers - churn_count:,}** customers were retained.")

                 # Calculate and display insights about top churn drivers if columns exist
                 try:
                     if 'Contract' in dashboard_df.columns:
                         churn_by_contract = dashboard_df.groupby('Contract')['Churn'].mean().sort_values(ascending=False) * 100
                         if not churn_by_contract.empty:
                              st.markdown(f"• Highest churn: **{churn_by_contract.index[0]}** contracts ({churn_by_contract.iloc[0]:.1f}%).")

                     if 'InternetService' in dashboard_df.columns:
                         churn_by_internet = dashboard_df.groupby('InternetService')['Churn'].mean().sort_values(ascending=False) * 100
                         if not churn_by_internet.empty:
                             st.markdown(f"• **{churn_by_internet.index[0]}** internet service has the highest churn ({churn_by_internet.iloc[0]:.1f}%).")

                     if 'tenure' in dashboard_df.columns and pd.api.types.is_numeric_dtype(dashboard_df['tenure']):
                          churn_df = dashboard_df[dashboard_df['Churn'] == 1]
                          retain_df = dashboard_df[dashboard_df['Churn'] == 0]
                          churn_by_tenure = churn_df['tenure'].mean() if not churn_df.empty else 0
                          retain_by_tenure = retain_df['tenure'].mean() if not retain_df.empty else 0
                          st.markdown(f"• Avg. tenure for churned: **{churn_by_tenure:.1f} months** (vs. {retain_by_tenure:.1f} for retained).")

                     if 'PaymentMethod' in dashboard_df.columns:
                         churn_by_payment = dashboard_df.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=False) * 100
                         if not churn_by_payment.empty:
                              st.markdown(f"• **{churn_by_payment.index[0]}** payment method sees highest churn ({churn_by_payment.iloc[0]:.1f}%).")

                 except Exception as e:
                     print(f"Dashboard Insight calculation error: {e}") # Log error but don't stop the app
                     st.markdown("• Error calculating some insights.")
            else:
                 st.markdown("• Churn data unavailable for insights.")

            #st.markdown("</div>", unsafe_allow_html=True)


        # --- Key Churn Factors Visualizations ---
        st.markdown("<h2 class='sub-header'>Exploring Key Churn Factors</h2>", unsafe_allow_html=True)

        # Display top factors in a 2x2 grid if columns exist
        col1, col2 = st.columns(2)

        # Helper function to display plot or info message
        def display_plot_or_info(df, column_name, plot_function, title_prefix="Churn Rate by"):
            if column_name in df.columns and 'Churn' in df.columns and pd.api.types.is_numeric_dtype(df['Churn']):
                fig = plot_function(df, column_name, title_prefix)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(f"<div class='metric-card' style='height: 450px; display: flex; align-items: center; justify-content: center;'><p>Data for '{column_name}' analysis not available or Churn data is invalid.</p></div>", unsafe_allow_html=True)


        with col1:
            display_plot_or_info(dashboard_df, 'Contract', plotly_churn_rates)
            display_plot_or_info(dashboard_df, 'InternetService', plotly_churn_rates)

        with col2:
            display_plot_or_info(dashboard_df, 'PaymentMethod', plotly_churn_rates)

            # Tenure grouped analysis
            if 'tenure' in dashboard_df.columns and 'Churn' in dashboard_df.columns and pd.api.types.is_numeric_dtype(dashboard_df['Churn']):
                tenure_df = dashboard_df.copy()
                bins = [-1, 12, 24, 36, 48, 60, np.inf] # Use -1 to include 0
                labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61+']
                tenure_df['TenureGroup'] = pd.cut(
                    tenure_df['tenure'],
                    bins=bins,
                    labels=labels,
                    right=True
                )
                # Check if TenureGroup was created successfully and has data
                if 'TenureGroup' in tenure_df.columns and not tenure_df['TenureGroup'].isnull().all():
                     tenure_fig = plotly_churn_rates(tenure_df, 'TenureGroup', title_prefix="Churn Rate by Tenure Group")
                     st.plotly_chart(tenure_fig, use_container_width=True)
                else:
                     st.markdown("<div class='metric-card' style='height: 450px; display: flex; align-items: center; justify-content: center;'><p>Could not create Tenure Groups for analysis.</p></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='metric-card' style='height: 450px; display: flex; align-items: center; justify-content: center;'><p>Tenure or Churn data not available for grouping.</p></div>", unsafe_allow_html=True)


    # Customer Churn Prediction Page
    elif page == "Churn Prediction":
        st.markdown("<h1 class='main-header'>Customer Churn Prediction</h1>", unsafe_allow_html=True)

        if xgb_model is None or not xgb_features: # Check list has content
             st.error("XGBoost model or its required features are not loaded. Prediction is unavailable.", icon="🚨")
             st.stop()

        st.markdown("""
            <div class='insight-box'>
            <p>Enter customer details below to predict their likelihood of churning.
            The prediction uses the loaded XGBoost model and requires input matching the data it was trained on.</p>
            </div>
        """, unsafe_allow_html=True)

        # --- Input Form ---
        st.markdown("<h2 class='sub-header'>Customer Information</h2>", unsafe_allow_html=True)

        # Helper to get unique values or provide defaults
        def get_options(df, col_name, defaults):
            if not df.empty and col_name in df.columns:
                 options = list(df[col_name].unique())
                 # Ensure standard 'No internet service' / 'No phone service' are present if expected
                 if 'No internet service' in defaults and 'No internet service' not in options: options.append('No internet service')
                 if 'No phone service' in defaults and 'No phone service' not in options: options.append('No phone service')
                 return options
            return defaults

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            # --- Column 1: Profile ---
            with col1:
                st.markdown("<h3 class='section-header'>Customer Profile</h3>", unsafe_allow_html=True)
                gender = st.selectbox("Gender", get_options(raw_df, 'gender', ["Male", "Female"]))
                # SeniorCitizen input is Yes/No, preprocessing handles conversion if model expects 0/1
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", get_options(raw_df, 'Partner', ["No", "Yes"]))
                dependents = st.selectbox("Dependents", get_options(raw_df, 'Dependents', ["No", "Yes"]))
                tenure = st.slider("Tenure (months)", 0, 72, 12, help="How many months has the customer been with the company?")

            # --- Column 2: Services ---
            with col2:
                st.markdown("<h3 class='section-header'>Services Subscribed</h3>", unsafe_allow_html=True)
                phone_service = st.selectbox("Phone Service", get_options(raw_df, 'PhoneService', ["Yes", "No"]))

                lines_defaults = ["No", "Yes", "No phone service"]
                lines_options = get_options(raw_df, 'MultipleLines', lines_defaults)
                if phone_service == "Yes":
                    # Filter out 'No phone service' option if phone service is Yes
                    multiple_lines = st.selectbox("Multiple Lines", [opt for opt in lines_options if opt != "No phone service"])
                else:
                    multiple_lines = "No phone service" # Assign the standard value if no phone service

                internet_defaults = ["DSL", "Fiber optic", "No"]
                internet_service = st.selectbox("Internet Service", get_options(raw_df, 'InternetService', internet_defaults))

                # Conditional Internet Add-ons
                addon_defaults = ["No", "Yes", "No internet service"]
                online_sec_options = get_options(raw_df, 'OnlineSecurity', addon_defaults)
                online_back_options = get_options(raw_df, 'OnlineBackup', addon_defaults)
                dev_prot_options = get_options(raw_df, 'DeviceProtection', addon_defaults)
                tech_supp_options = get_options(raw_df, 'TechSupport', addon_defaults)
                tv_options = get_options(raw_df, 'StreamingTV', addon_defaults)
                movies_options = get_options(raw_df, 'StreamingMovies', addon_defaults)

                if internet_service != "No":
                    online_security = st.selectbox("Online Security", [opt for opt in online_sec_options if opt != "No internet service"])
                    online_backup = st.selectbox("Online Backup", [opt for opt in online_back_options if opt != "No internet service"])
                    device_protection = st.selectbox("Device Protection", [opt for opt in dev_prot_options if opt != "No internet service"])
                    tech_support = st.selectbox("Tech Support", [opt for opt in tech_supp_options if opt != "No internet service"])
                    streaming_tv = st.selectbox("Streaming TV", [opt for opt in tv_options if opt != "No internet service"])
                    streaming_movies = st.selectbox("Streaming Movies", [opt for opt in movies_options if opt != "No internet service"])
                else:
                    # Assign the standard 'No internet service' value for consistency
                    online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No internet service"

            # --- Column 3: Account ---
            with col3:
                st.markdown("<h3 class='section-header'>Account Information</h3>", unsafe_allow_html=True)
                contract = st.selectbox("Contract", get_options(raw_df, 'Contract', ["Month-to-month", "One year", "Two year"]))
                paperless_billing = st.selectbox("Paperless Billing", get_options(raw_df, 'PaperlessBilling', ["Yes", "No"]))
                payment_method = st.selectbox("Payment Method", get_options(raw_df, 'PaymentMethod', ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]))
                # Use number_input for more precise control over charges
                min_charge = 0.0 if raw_df.empty or 'MonthlyCharges' not in raw_df.columns else raw_df['MonthlyCharges'].min()
                max_charge = 200.0 if raw_df.empty or 'MonthlyCharges' not in raw_df.columns else raw_df['MonthlyCharges'].max()
                default_charge = 70.0 if raw_df.empty or 'MonthlyCharges' not in raw_df.columns else raw_df['MonthlyCharges'].median()

                monthly_charges = st.number_input("Monthly Charges ($)", min_value=float(min_charge), max_value=float(max_charge), value=float(default_charge), step=0.05, format="%.2f", help="Customer's current monthly bill amount.")

                # Only calculate/include TotalCharges if the model expects it
                total_charges_input = None
                if 'TotalCharges' in xgb_features:
                     # Provide a calculated estimate or allow user input
                     total_charges_input = tenure * monthly_charges # Simplified estimate
                     st.text_input("Total Charges ($) (auto-calculated)", value=f"{total_charges_input:.2f}", disabled=True, help="Estimated total charges based on tenure and monthly charges. The model uses this if required.")


            # Submit button for the form
            submitted = st.form_submit_button("Predict Churn")

        # --- Prediction Execution and Display ---
        if submitted:
            # Create DataFrame from inputs
            customer_input_dict = {
                'gender': gender,
                'SeniorCitizen': senior_citizen, # Keep as Yes/No here, preprocess handles conversion
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
            }
            # Add TotalCharges only if it's expected by the model AND was calculated
            if 'TotalCharges' in xgb_features and total_charges_input is not None:
                 customer_input_dict['TotalCharges'] = total_charges_input
                 print("Included estimated TotalCharges in input for model.")
            elif 'TotalCharges' in xgb_features:
                 st.warning("Model expects 'TotalCharges' but it could not be estimated from inputs.", icon="⚠️")


            customer_input_data = pd.DataFrame([customer_input_dict])

            st.markdown("<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
            with st.spinner("Analyzing customer profile and predicting churn..."):
                # Preprocess the input data to match model features
                processed_customer_data = preprocess_input_data(customer_input_data, xgb_features)

                if processed_customer_data is not None:
                    # Make prediction
                    churn_prediction, churn_prob = predict_churn(processed_customer_data, xgb_model)

                    if churn_prediction is not None and churn_prob is not None:
                        # Display prediction result
                        if churn_prediction == 1:
                            st.markdown(
                                f"""
                                <div class='prediction-box-churn'>
                                    <h1 style='color: #DC2626; margin-bottom: 5px;'>High Risk of Churn</h1>
                                    <h3 style='margin-top: 5px; margin-bottom: 10px;'>Predicted Churn Probability: {churn_prob:.1%}</h3>
                                    <p>This customer profile suggests a high likelihood of leaving.</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"""
                                <div class='prediction-box-stay'>
                                    <h1 style='color: #16A34A; margin-bottom: 5px;'>Low Risk of Churn</h1>
                                    <h3 style='margin-top: 5px; margin-bottom: 10px;'>Predicted Churn Probability: {churn_prob:.1%}</h3>
                                    <p>This customer profile suggests a low likelihood of leaving.</p>
                                </div>
                                """, unsafe_allow_html=True)

                        # --- Explainability (Simplified - SHAP/LIME would be better) ---
                        st.markdown("<h3 class='section-header'>Key Factors Influencing Prediction (Illustrative)</h3>", unsafe_allow_html=True)
                        factors = []
                        # Simple rule-based illustration - replace with SHAP/LIME in production
                        if contract == "Month-to-month": factors.append("Month-to-month Contract (+ Risk)")
                        if tenure < 6: factors.append("Very Low Tenure (< 6 mo) (+ Risk)")
                        elif tenure > 48: factors.append("High Tenure (> 48 mo) (- Risk)")
                        if internet_service == "Fiber optic" and monthly_charges > 80: factors.append("Fiber Optic & High Charges (+ Risk)")
                        elif internet_service == "DSL": factors.append("DSL Service (Lower Risk than Fiber)")
                        if online_security == "No" and internet_service != "No": factors.append("No Online Security (+ Risk)")
                        if tech_support == "No" and internet_service != "No": factors.append("No Tech Support (+ Risk)")
                        if payment_method == "Electronic check": factors.append("Electronic Check Payment (+ Risk)")
                        if contract == "Two year": factors.append("Two Year Contract (- Risk)")

                        if factors:
                            # Display factors with icons
                            factor_html = "<ul>"
                            for factor in factors[:5]: # Show top 5 illustrative factors
                                 icon = "🔴" if "+" in factor else "🟢" if "-" in factor else "⚪"
                                 factor_html += f"<li style='margin-bottom: 5px;'>{icon} {factor.replace('(+ Risk)', '').replace('(- Risk)', '').strip()}</li>"
                            factor_html += "</ul>"
                            st.markdown(f"<div class='insight-box'>{factor_html}</div>", unsafe_allow_html=True)
                        else:
                             st.markdown("<div class='insight-box'><p>Customer profile appears balanced or lacks strong risk indicators according to simple illustrative rules.</p></div>", unsafe_allow_html=True)
                        st.caption("Note: These factors are illustrative based on common patterns. For precise contribution, model-specific interpretation methods like SHAP are needed.")


                        # --- Recommended Actions ---
                        st.markdown("<h3 class='section-header'>Recommended Actions</h3>", unsafe_allow_html=True)
                        if churn_prediction == 1:
                            st.markdown("""
                                <div class='insight-box' style='border-left-color: #EF4444;'>
                                <h4>Retention Strategy Suggestions:</h4>
                                <ul>
                                    <li><b>Offer Contract Upgrade:</b> Provide a discount/incentive for switching to a 1-year or 2-year contract (if not already on one).</li>
                                    <li><b>Promote Value Add-ons:</b> Offer a trial or discount on Online Security or Tech Support if not subscribed and applicable.</li>
                                    <li><b>Loyalty Discount/Credit:</b> Consider a temporary targeted discount on monthly charges.</li>
                                    <li><b>Proactive Outreach:</b> Schedule a customer service call to understand potential dissatisfaction and offer solutions.</li>
                                    <li><b>Review Payment Options:</b> Suggest switching from Electronic Check to automatic bank/card payments if applicable.</li>
                                </ul>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                                <div class='insight-box' style='border-left-color: #22C55E;'>
                                <h4>Engagement & Growth Suggestions:</h4>
                                <ul>
                                    <li><b>Service Upgrade Review:</b> Check eligibility for relevant service enhancements (e.g., faster internet, premium TV).</li>
                                    <li><b>Bundling Opportunities:</b> Analyze if current services could be bundled for better value or convenience.</li>
                                    <li><b>Referral Program Promotion:</b> Remind customer about referral bonuses if available.</li>
                                    <li><b>Value Confirmation:</b> Periodically highlight usage, reliability, or new features included in their plan.</li>
                                    <li><b>Explore Complementary Services:</b> Suggest other relevant services they aren't currently using.</li>
                                </ul>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        # Error message already shown by predict_churn
                        st.info("Prediction could not be completed. Please check inputs and model compatibility.", icon="ℹ️")
                else:
                    # Error message already shown by preprocess_input_data
                     st.info("Input data could not be processed for prediction. Please check input values.", icon="ℹ️")


    # Churn Analysis Page
    elif page == "Churn Analysis":
        st.markdown("<h1 class='main-header'>Detailed Churn Analysis</h1>", unsafe_allow_html=True)

        if raw_df.empty:
            st.warning("Raw data (`churn_raw_data.csv`) is not available. Analysis cannot be performed.", icon="⚠️")
            st.stop()

        analysis_df = raw_df.copy() # Use a copy for analysis modifications

        # Ensure Churn column is suitable for analysis
        if 'Churn' not in analysis_df.columns or not pd.api.types.is_numeric_dtype(analysis_df['Churn']):
             st.error("A valid 'Churn' column (numeric 0/1) in the raw data is required for this analysis page.", icon="🚨")
             st.stop()


        # --- Tabs for Different Analysis Areas ---
        tab_titles = ["Key Drivers", "Demographics & Tenure", "Contract & Financials", "Segmentation"]
        tabs = st.tabs(tab_titles)

        # --- Tab 1: Key Drivers (Services, Contract) ---
        with tabs[0]:
            st.markdown("<h2 class='sub-header'>Analysis of Key Churn Drivers</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            # Helper defined inside main or globally if needed elsewhere
            def display_analysis_plot(df, column_name, plot_function, title_prefix="Churn Rate by"):
                 """ Helper to display plot or info message for analysis """
                 if column_name in df.columns:
                     fig = plot_function(df, column_name, title_prefix)
                     st.plotly_chart(fig, use_container_width=True)
                 else:
                     st.markdown(f"<div class='metric-card' style='height: 450px; display: flex; align-items: center; justify-content: center;'><p>Data for '{column_name}' analysis not available.</p></div>", unsafe_allow_html=True)

            with col1:
                st.markdown("<h3 class='section-header'>Impact of Contract Type</h3>", unsafe_allow_html=True)
                display_analysis_plot(analysis_df, 'Contract', plotly_churn_rates)

                st.markdown("<h3 class='section-header'>Impact of Internet Service</h3>", unsafe_allow_html=True)
                display_analysis_plot(analysis_df, 'InternetService', plotly_churn_rates)

            with col2:
                st.markdown("<h3 class='section-header'>Impact of Payment Method</h3>", unsafe_allow_html=True)
                display_analysis_plot(analysis_df, 'PaymentMethod', plotly_churn_rates)


                # Add-on Services Analysis (Select Box)
                st.markdown("<h3 class='section-header'>Impact of Add-On Services</h3>", unsafe_allow_html=True)
                addon_services = [
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies'
                ]
                available_addons = [svc for svc in addon_services if svc in analysis_df.columns]

                if available_addons and 'InternetService' in analysis_df.columns:
                    # Filter for customers who have internet service for meaningful comparison
                    internet_customers = analysis_df[analysis_df['InternetService'] != 'No'].copy()
                    if not internet_customers.empty:
                        selected_addon = st.selectbox(
                             "Select Add-on Service to Analyze (for customers with Internet)",
                             available_addons,
                             key="addon_select_analysis"
                         )
                        display_analysis_plot(internet_customers, selected_addon, plotly_churn_rates)
                    else:
                         st.info("No customers with internet service found for add-on analysis.")
                elif not available_addons:
                     st.info("No add-on service columns found in the data (e.g., OnlineSecurity, TechSupport).")
                else: # InternetService column missing
                     st.info("The 'InternetService' column is required to analyze add-on service impact correctly.")


        # --- Tab 2: Demographics & Tenure ---
        with tabs[1]:
            st.markdown("<h2 class='sub-header'>Demographic and Tenure Analysis</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                 # Tenure Distribution by Churn Status
                st.markdown("<h3 class='section-header'>Tenure Distribution by Churn</h3>", unsafe_allow_html=True)
                if 'tenure' in analysis_df.columns:
                    fig_tenure_hist = px.histogram(
                        analysis_df,
                        x='tenure',
                        color=analysis_df['Churn'].map({1: 'Churned', 0: 'Retained'}),
                        barmode='overlay', # Overlay bars for comparison
                        marginal='box', # Add box plot marginal for summary stats
                        color_discrete_map={'Churned': '#EF4444', 'Retained': '#22C55E'},
                        labels={'tenure': 'Tenure (Months)', 'Churn': 'Churn Status'},
                        title='Distribution of Tenure for Churned vs. Retained Customers'
                    )
                    fig_tenure_hist.update_layout(height=400, legend_title_text='Status')
                    st.plotly_chart(fig_tenure_hist, use_container_width=True)
                else:
                    st.info("Tenure data not available for histogram.")

                # Senior Citizen
                st.markdown("<h3 class='section-header'>Impact of Senior Citizen Status</h3>", unsafe_allow_html=True)
                if 'SeniorCitizen' in analysis_df.columns:
                    # Ensure 'SeniorCitizen' is treated as categorical Yes/No for plotting
                    temp_df = analysis_df.copy()
                    # Convert numeric 0/1 to Yes/No if needed for plotting labels
                    if pd.api.types.is_numeric_dtype(temp_df['SeniorCitizen']):
                         temp_df['SeniorCitizen'] = temp_df['SeniorCitizen'].map({1: 'Yes', 0: 'No'}).astype(str)
                    display_analysis_plot(temp_df, 'SeniorCitizen', plotly_churn_rates)
                else:
                     st.info("Senior Citizen data not available.")

            with col2:
                # Gender
                st.markdown("<h3 class='section-header'>Impact of Gender</h3>", unsafe_allow_html=True)
                display_analysis_plot(analysis_df, 'gender', plotly_churn_rates)

                # Dependents
                st.markdown("<h3 class='section-header'>Impact of Having Dependents</h3>", unsafe_allow_html=True)
                display_analysis_plot(analysis_df, 'Dependents', plotly_churn_rates)

                # Partner
                st.markdown("<h3 class='section-header'>Impact of Having a Partner</h3>", unsafe_allow_html=True)
                display_analysis_plot(analysis_df, 'Partner', plotly_churn_rates)


        # --- Tab 3: Contract & Financials ---
        with tabs[2]:
            st.markdown("<h2 class='sub-header'>Contract and Financial Analysis</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                # Monthly Charges Distribution by Churn Status
                st.markdown("<h3 class='section-header'>Monthly Charges Distribution by Churn</h3>", unsafe_allow_html=True)
                if 'MonthlyCharges' in analysis_df.columns:
                    fig_mc_hist = px.histogram(
                        analysis_df,
                        x='MonthlyCharges',
                        color=analysis_df['Churn'].map({1: 'Churned', 0: 'Retained'}),
                        barmode='overlay',
                        marginal='box', # Use box plot marginal
                        color_discrete_map={'Churned': '#EF4444', 'Retained': '#22C55E'},
                        labels={'MonthlyCharges': 'Monthly Charges ($)', 'Churn': 'Churn Status'},
                        title='Distribution of Monthly Charges by Churn Status'
                    )
                    fig_mc_hist.update_layout(height=400, legend_title_text='Status')
                    st.plotly_chart(fig_mc_hist, use_container_width=True)
                else:
                    st.info("Monthly Charges data not available for histogram.")

                # Paperless Billing
                st.markdown("<h3 class='section-header'>Impact of Paperless Billing</h3>", unsafe_allow_html=True)
                display_analysis_plot(analysis_df, 'PaperlessBilling', plotly_churn_rates)

            with col2:
                # Scatter plot: Monthly Charges vs Tenure, colored by Churn
                st.markdown("<h3 class='section-header'>Monthly Charges vs. Tenure</h3>", unsafe_allow_html=True)
                if 'MonthlyCharges' in analysis_df.columns and 'tenure' in analysis_df.columns:
                    hover_cols = ['Contract', 'InternetService']
                    available_hover_cols = [col for col in hover_cols if col in analysis_df.columns]

                    fig_scatter = px.scatter(
                        analysis_df,
                        x='tenure',
                        y='MonthlyCharges',
                        color=analysis_df['Churn'].map({1: 'Churned', 0: 'Retained'}),
                        color_discrete_map={'Churned': '#EF4444', 'Retained': '#22C55E'},
                        title='Monthly Charges vs. Tenure by Churn Status',
                        labels={'tenure': 'Tenure (Months)', 'MonthlyCharges': 'Monthly Charges ($)', 'Churn':'Churn Status'},
                        hover_data=available_hover_cols, # Add available hover info
                        opacity=0.7 # Reduce opacity for dense plots
                    )
                    fig_scatter.update_layout(height=400, legend_title_text='Status')
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    st.markdown("<div class='insight-box'><p>Observe clustering: High churn often occurs with high monthly charges and low tenure, especially for certain contract types (hover over points). Retained customers tend to cluster at higher tenures across various charge levels.</p></div>", unsafe_allow_html=True)
                else:
                    st.info("Monthly Charges or Tenure data not available for scatter plot.")

            # --- Contract Transition Simulation ---
            st.markdown("<h2 class='sub-header'>Contract Transition Simulation</h2>", unsafe_allow_html=True)
            if 'Contract' in analysis_df.columns and 'Churn' in analysis_df.columns:
                st.markdown("""
                    <div class='insight-box'>
                    <p>Simulate the potential impact on the overall churn rate by hypothetically converting a percentage of Month-to-Month customers to longer-term contracts. This helps quantify the value of retention efforts focused on contract upgrades.</p>
                    </div>
                    """, unsafe_allow_html=True)

                try:
                    # Calculate current state safely
                    contract_groups = analysis_df.groupby('Contract')
                    churn_rates = contract_groups['Churn'].mean()
                    counts = contract_groups.size()

                    m2m_churn_rate = churn_rates.get('Month-to-month', 0)
                    m2m_count = counts.get('Month-to-month', 0)
                    one_yr_churn_rate = churn_rates.get('One year', 0)
                    one_yr_count = counts.get('One year', 0)
                    two_yr_churn_rate = churn_rates.get('Two year', 0)
                    two_yr_count = counts.get('Two year', 0)

                    total_customers = len(analysis_df)
                    current_churn_rate = analysis_df['Churn'].mean()

                    if m2m_count == 0:
                         st.info("No Month-to-Month customers found in the data to simulate conversion.")
                    else:
                        # Simulation controls
                        st.markdown("<h3 class='section-header'>Simulation Parameters</h3>", unsafe_allow_html=True)
                        sim_col1, sim_col2 = st.columns(2)
                        with sim_col1:
                            percent_to_1yr = st.slider(
                                "% of M2M Converted to 1-Year", 0, 100, 10,
                                key="sim_1yr",
                                help="Percentage of current Month-to-Month customers assumed to switch to a One Year contract."
                            )
                        with sim_col2:
                            # Max value for 2yr slider depends on the 1yr slider value
                            max_2yr = 100 - percent_to_1yr
                            percent_to_2yr = st.slider(
                                "% of M2M Converted to 2-Year", 0, max_2yr, 5,
                                key="sim_2yr",
                                help="Percentage of current Month-to-Month customers assumed to switch to a Two Year contract."
                            )

                        # Calculate new distribution and churn
                        converted_to_1yr = m2m_count * (percent_to_1yr / 100.0)
                        converted_to_2yr = m2m_count * (percent_to_2yr / 100.0)

                        new_m2m_count = m2m_count - converted_to_1yr - converted_to_2yr
                        new_one_yr_count = one_yr_count + converted_to_1yr
                        new_two_yr_count = two_yr_count + converted_to_2yr

                        # Calculate total churned customers in new scenario
                        new_total_churned = (new_m2m_count * m2m_churn_rate) + \
                                            (new_one_yr_count * one_yr_churn_rate) + \
                                            (new_two_yr_count * two_yr_churn_rate)

                        projected_churn_rate = new_total_churned / total_customers if total_customers > 0 else 0

                        # Display results
                        st.markdown("<h3 class='section-header'>Simulation Results</h3>", unsafe_allow_html=True)
                        res_col1, res_col2, res_col3 = st.columns(3)
                        with res_col1:
                            st.metric("Current Churn Rate", f"{current_churn_rate:.2%}")
                        with res_col2:
                            st.metric("Projected Churn Rate", f"{projected_churn_rate:.2%}",
                                      f"{projected_churn_rate - current_churn_rate:.2%}",
                                      delta_color="inverse") # Lower is better
                        with res_col3:
                            churn_reduction = current_churn_rate - projected_churn_rate
                            customers_saved = churn_reduction * total_customers
                            st.metric("Potential Customers Retained", f"{int(round(customers_saved)):,}") # Round before int

                        # Optional: Add financial impact if MonthlyCharges available
                        if 'MonthlyCharges' in analysis_df.columns and pd.api.types.is_numeric_dtype(analysis_df['MonthlyCharges']):
                            avg_monthly_charge = analysis_df['MonthlyCharges'].mean()
                            estimated_annual_saving = customers_saved * avg_monthly_charge * 12
                            st.success(f"Estimated Annual Revenue Saved: **${estimated_annual_saving:,.2f}** (based on avg. monthly charge)")
                        else:
                             st.info("Monthly Charges data needed to estimate financial impact.")


                except Exception as e:
                    st.error(f"Error during simulation: {e}", icon="🚨")
            else:
                 st.info("Contract and Churn data needed for simulation.")

        # --- Tab 4: Segmentation (Clustering) ---
        with tabs[3]:
            st.markdown("<h2 class='sub-header'>Customer Segmentation via Clustering</h2>", unsafe_allow_html=True)

            st.markdown("""
                    <div class='insight-box'>
                    <p>Use K-Means clustering on numerical features like tenure and charges to identify distinct customer groups (segments). Analyzing the churn rate and characteristics of each segment helps in tailoring marketing and retention strategies more effectively.</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Features for clustering
            numeric_cols_options = ['tenure', 'MonthlyCharges', 'TotalCharges']
            available_numeric = [
                col for col in numeric_cols_options
                if col in analysis_df.columns and pd.api.types.is_numeric_dtype(analysis_df[col])
            ]

            if len(available_numeric) < 2:
                st.warning("At least two numeric features (e.g., tenure, MonthlyCharges) with valid data are needed for clustering analysis.", icon="⚠️")
            else:
                # Cluster configuration
                st.markdown("<h3 class='section-header'>Clustering Configuration</h3>", unsafe_allow_html=True)
                col1, col2 = st.columns([1, 3])
                with col1:
                    features_to_cluster = st.multiselect(
                        "Select Features for Clustering",
                        available_numeric,
                        default=available_numeric[:min(len(available_numeric), 2)], # Default to first 2 available
                        key="cluster_features_select"
                        )
                    n_clusters = st.slider(
                         "Number of Segments (Clusters)",
                         min_value=2, max_value=8, value=3, key="n_clusters_slider",
                         help="How many distinct customer groups to identify."
                         )

                if len(features_to_cluster) < 2:
                     st.warning("Please select at least 2 features for clustering.", icon="⚠️")
                else:
                    with col2:
                        st.markdown("<h3 class='section-header'>Segmentation Results</h3>", unsafe_allow_html=True)
                        try:
                            # Prepare data - Select features and drop rows with NaNs *in those features*
                            cluster_data_raw = analysis_df[features_to_cluster].copy()
                            original_indices = cluster_data_raw.index # Keep track of original indices before dropna
                            cluster_data_clean = cluster_data_raw.dropna()

                            if cluster_data_clean.empty:
                                st.warning("No data remains after removing rows with missing values in selected features. Clustering cannot proceed.", icon="⚠️")
                            else:
                                # Scale data
                                scaler = StandardScaler()
                                scaled_data = scaler.fit_transform(cluster_data_clean)

                                # Perform K-Means
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Set n_init explicitly
                                cluster_labels = kmeans.fit_predict(scaled_data)

                                # Add cluster labels back to the original dataframe (only for rows used in clustering)
                                analysis_df_clustered = analysis_df.loc[cluster_data_clean.index].copy()
                                analysis_df_clustered['Cluster'] = cluster_labels.astype(str) # Treat as categorical string

                                # Check if Churn column is suitable for symbol mapping
                                if 'Churn' not in analysis_df_clustered.columns or not pd.api.types.is_numeric_dtype(
                                        analysis_df_clustered['Churn']):
                                    st.warning(
                                        "Churn column missing or non-numeric in clustered data, cannot use for symbols.",
                                        icon="⚠️")
                                    symbol_col_name = None  # Don't use symbol mapping
                                else:
                                    symbol_col_name = 'Churn'  # Use Churn for symbols
                                    print(f"Using column '{symbol_col_name}' for plot symbols.")  # Debug print

                                # Visualization
                                st.markdown("#### Segment Visualization", unsafe_allow_html=True)
                                hover_cols_cluster = ['Contract', 'InternetService', 'PaymentMethod']
                                available_hover_cluster = [col for col in hover_cols_cluster if col in analysis_df_clustered.columns]

                                if len(features_to_cluster) == 2:
                                    fig_cluster = px.scatter(
                                        analysis_df_clustered,
                                        x=features_to_cluster[0],
                                        y=features_to_cluster[1],
                                        color='Cluster',
                                        symbol='Churn', # Use Churn for symbol
                                        symbol_map={0:'circle', 1:'x'}, # Map 0/1 to symbols (adjust size below)
                                        title=f'Customer Segments based on {", ".join(features_to_cluster)} ({n_clusters} Clusters)',
                                        labels={col: col.replace('Charges', ' Charges ($)') for col in features_to_cluster}, # Add units to labels
                                        hover_data=available_hover_cluster,
                                        color_discrete_sequence=px.colors.qualitative.Pastel # Use a nice color scheme
                                    )
                                    fig_cluster.update_traces(marker=dict(size=8, opacity=0.7))
                                    fig_cluster.update_layout(height=500, legend_title_text='Segment')
                                    st.plotly_chart(fig_cluster, use_container_width=True)

                                elif len(features_to_cluster) == 3:
                                     fig_cluster = px.scatter_3d(
                                        analysis_df_clustered,
                                        x=features_to_cluster[0],
                                        y=features_to_cluster[1],
                                        z=features_to_cluster[2],
                                        color='Cluster',
                                        symbol='Churn',
                                        symbol_map={0:'circle', 1:'cross'},
                                        title=f'Customer Segments based on {", ".join(features_to_cluster)} ({n_clusters} Clusters)',
                                        labels={col: col.replace('Charges', ' Charges ($)') for col in features_to_cluster},
                                        hover_data=available_hover_cluster,
                                        color_discrete_sequence=px.colors.qualitative.Pastel
                                    )
                                     fig_cluster.update_traces(marker=dict(size=5, opacity=0.7))
                                     fig_cluster.update_layout(height=600, legend_title_text='Segment')
                                     st.plotly_chart(fig_cluster, use_container_width=True)
                                else: # If user somehow selected > 3
                                    st.info("Clustering performed, but visualization is only available for 2 or 3 selected features.")


                                # --- Cluster Profiles ---
                                st.markdown("<h3 class='section-header'>Segment Profiles</h3>", unsafe_allow_html=True)

                                # Calculate summary statistics per cluster
                                summary_agg = {
                                    'Churn': 'mean',
                                     # Use a column guaranteed to exist, or index size
                                    features_to_cluster[0]: 'size'
                                }
                                for feat in features_to_cluster:
                                    summary_agg[feat] = 'mean'

                                # Include key categorical features if they exist
                                cat_feats_profile = ['Contract', 'InternetService', 'PaymentMethod']
                                for feat in cat_feats_profile:
                                     if feat in analysis_df_clustered.columns:
                                          # Calculate mode (most frequent value) for categorical features
                                          summary_agg[feat] = lambda x: x.mode()[0] if not x.mode().empty else 'N/A'


                                cluster_summary = analysis_df_clustered.groupby('Cluster').agg(summary_agg).reset_index()

                                # Rename columns for display clarity
                                rename_map = {
                                    features_to_cluster[0] + '_size': 'Segment Size', # Rename the size column
                                    'Churn': 'Avg. Churn Rate',
                                }
                                # Rename mean columns
                                for feat in features_to_cluster:
                                    rename_map[feat] = f'Avg. {feat}'
                                # Rename mode columns
                                for feat in cat_feats_profile:
                                     if feat in cluster_summary.columns: # Check if mode was actually calculated
                                         rename_map[feat] = f'Typical {feat}'

                                # Apply renaming, handling potential KeyError if size wasn't calculated as expected
                                try:
                                     # Correctly identify the size column before renaming
                                     size_col_name = next((col for col in cluster_summary.columns if col.endswith('_size')), None)
                                     if size_col_name:
                                         rename_map[size_col_name] = 'Segment Size' # Set the correct key
                                     cluster_summary = cluster_summary.rename(columns=rename_map)
                                except KeyError as ke:
                                     st.warning(f"Could not rename summary columns correctly: {ke}", icon="⚠️")


                                # Format the summary table for better readability
                                format_dict = {'Avg. Churn Rate': "{:.1%}"}
                                if 'Segment Size' in cluster_summary.columns: format_dict['Segment Size'] = "{:,}"
                                for feat in features_to_cluster:
                                     avg_col_name = f'Avg. {feat}'
                                     if avg_col_name in cluster_summary.columns:
                                         if feat=='tenure': format_dict[avg_col_name] = "{:.1f}"
                                         elif 'Charges' in feat: format_dict[avg_col_name] = "${:.2f}"
                                         else: format_dict[avg_col_name] = "{:.2f}" # Default format

                                styled_summary = cluster_summary.style.format(format_dict)\
                                                                .background_gradient(subset=['Avg. Churn Rate'], cmap='Reds', axis=0)\
                                                                .set_properties(**{'text-align': 'center'}) # Center align content

                                st.dataframe(styled_summary, use_container_width=True)

                                # Add brief text descriptions per cluster
                                st.markdown("#### Segment Descriptions", unsafe_allow_html=True)
                                for index, row in cluster_summary.iterrows():
                                    cluster_id = row['Cluster']
                                    desc = f"<div class='insight-box' style='border-left-color: {px.colors.qualitative.Pastel[int(cluster_id)]};'>" # Use cluster color
                                    desc += f"<b>Segment {cluster_id}</b>"
                                    if 'Segment Size' in row: desc += f" ({row['Segment Size']:,} customers)"
                                    desc += ":<br>"

                                    # Add characteristic descriptions
                                    char_list = []
                                    if 'Avg. Churn Rate' in row: char_list.append(f"Churn Rate: <b>{row['Avg. Churn Rate']:.1%}</b>")
                                    for feat in features_to_cluster:
                                         avg_col = f'Avg. {feat}'
                                         if avg_col in row:
                                              formatted_val = format_dict.get(avg_col, "{}").format(row[avg_col])
                                              char_list.append(f"Avg. {feat}: {formatted_val}")
                                    for feat in cat_feats_profile:
                                         typical_col = f'Typical {feat}'
                                         if typical_col in row and row[typical_col] != 'N/A':
                                              char_list.append(f"Typical {feat}: {row[typical_col]}")

                                    desc += " • " + "<br> • ".join(char_list)
                                    desc += "</div>"
                                    st.markdown(desc, unsafe_allow_html=True)


                        except Exception as e:
                            st.error(f"Error during clustering analysis: {e}", icon="🚨")
                            import traceback
                            st.code(traceback.format_exc()) # Show detailed error in app for debugging


    # Decision Tree Visualization Page
    elif page == "Decision Tree Viz":
        st.markdown("<h1 class='main-header'>Decision Tree Model Visualization</h1>", unsafe_allow_html=True)

        if dt_model is None:
             st.error("Decision Tree model (`decision_tree_model.joblib`) is not loaded. Visualization unavailable.", icon="🚨")
             st.stop()

        st.markdown("""
            <div class='insight-box'>
            <p>This page visualizes the trained Decision Tree model. Decision trees create a flowchart-like structure to predict outcomes (Churned/Retained). Each split point (node) uses a feature to divide the data, aiming to create purer groups at the next level. The paths from the top (root) to the bottom (leaves) represent the decision rules learned by the model.</p>
            </div>
            """, unsafe_allow_html=True)

        # --- Determine Feature Names ---
        viz_feature_names = None
        if dt_features:
            viz_feature_names = dt_features
            print("Using DT features from model object.")
        elif not processed_df.empty:
             # Infer from processed_df if model didn't provide names
             # Ensure 'Churn' or target variable isn't included
             potential_features = [col for col in processed_df.columns if col not in ['Churn', 'ChurnFlag']] # Adapt if target name differs
             if potential_features:
                 viz_feature_names = potential_features
                 #st.warning("Using feature names inferred from the processed data file (`churn_df_su.csv`) as the Decision Tree model file did not contain them. Ensure this data matches the model's training data.", icon="⚠️")
                 print(f"Using DT features inferred from CSV: {viz_feature_names}")
             else:
                  st.error("Could not determine feature names for the Decision Tree plot from model or processed data.", icon="🚨")
        else:
             st.error("Could not determine feature names for the Decision Tree plot. Requires model with features or the processed data file.", icon="🚨")


        if viz_feature_names:
             # --- Tree Plot ---
            st.markdown("<h2 class='sub-header'>Interactive Decision Tree Plot</h2>", unsafe_allow_html=True)
            class_names = ['Retained', 'Churned'] # More descriptive names

            # Tree Depth Slider
            actual_max_depth = dt_model.get_depth()
            # Ensure slider max isn't larger than actual depth, keep limit for readability
            slider_max = min(actual_max_depth, 7)
            slider_default = min(actual_max_depth, 3)

            max_display_depth = st.slider(
                "Select Maximum Tree Depth to Display",
                min_value=1,
                max_value=slider_max,
                value=slider_default,
                help=f"Adjust the complexity of the tree shown (Actual tree depth: {actual_max_depth}). Deeper trees show more detail but can become very large and harder to read."
            )

            try:
                # Plotting the tree using Matplotlib - Use a large figure size dynamically
                # Estimate needed width based on nodes at max depth (roughly 2^depth)
                est_width = max(20, 2.5 * (2**max_display_depth))
                est_height = max(12, 3 * max_display_depth)

                fig, ax = plt.subplots(figsize=(est_width, est_height))
                plot_tree(
                    dt_model,
                    max_depth=max_display_depth,
                    feature_names=viz_feature_names,
                    class_names=class_names,
                    filled=True,
                    rounded=True,
                    fontsize=max(6, 12 - max_display_depth), # Decrease font size slightly for deeper trees
                    ax=ax,
                    proportion=True, # Show proportions instead of counts
                    precision=2 # Decimal places for values/proportions
                )
                ax.set_title(f"Decision Tree for Churn Prediction (Displayed Depth: {max_display_depth})", fontsize=16)
                plt.tight_layout() # Adjust layout

                # Display the plot in Streamlit
                st.pyplot(fig)

                # Offer download button
                try:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight') # Higher DPI for download
                    st.download_button(
                        label="Download Tree Image (PNG)",
                        data=buf.getvalue(),
                        file_name=f"decision_tree_depth_{max_display_depth}.png",
                        mime="image/png"
                    )
                except Exception as dl_e:
                     st.warning(f"Could not create download button for tree image: {dl_e}", icon="⚠️")

                plt.close(fig) # Close figure after plotting and saving to buffer

            except Exception as plot_e:
                st.error(f"Error generating Decision Tree plot: {plot_e}", icon="🚨")
                st.info("Ensure the model is a fitted Decision Tree and feature names list length matches the number of features used by the model.")
                import traceback
                st.code(traceback.format_exc())


            # --- Feature Importance ---
            st.markdown("<h2 class='sub-header'>Feature Importance (Decision Tree)</h2>", unsafe_allow_html=True)
            if hasattr(dt_model, 'feature_importances_'):
                try:
                    if len(viz_feature_names) == len(dt_model.feature_importances_):
                        importance_df = pd.DataFrame({
                            'Feature': viz_feature_names,
                            'Importance': dt_model.feature_importances_
                        }).sort_values(by='Importance', ascending=False)

                        # Filter out features with 0 importance for clarity
                        importance_df = importance_df[importance_df['Importance'] > 0]

                        if not importance_df.empty:
                            # Show top N features or all if fewer than N have importance > 0
                            top_n = min(15, len(importance_df))
                            top_features = importance_df.head(top_n)

                            fig_importance = px.bar(
                                top_features,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title=f'Top {top_n} Most Important Features (Decision Tree)',
                                labels={'Importance': 'Importance Score (Gini Impurity Reduction or similar)'},
                                color='Importance',
                                color_continuous_scale=px.colors.sequential.Blues # Use blue scale
                            )
                            # Order bars by importance and adjust height
                            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}, height=max(400, top_n * 25))
                            st.plotly_chart(fig_importance, use_container_width=True)

                            st.markdown("""
                                <div class='insight-box'>
                                <p>Feature importance scores indicate how much each feature contributes to the model's decisions within this specific Decision Tree. The score typically reflects the total reduction in node impurity (like Gini impurity or entropy) brought about by splits involving that feature across the entire tree. Higher scores mean the feature was more effective in separating churned from retained customers in this model's structure.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                             st.info("No features found with importance greater than 0 in this Decision Tree model.")

                    else:
                         st.warning(f"Mismatch between number of feature names ({len(viz_feature_names)}) and feature importances ({len(dt_model.feature_importances_)}). Cannot display importance plot.", icon="⚠️")

                except Exception as imp_e:
                    st.error(f"Error generating feature importance plot: {imp_e}", icon="🚨")
            else:
                st.info("Feature importance data (`feature_importances_`) not available in the loaded Decision Tree model object.")

        else: # viz_feature_names is None
             st.info("Cannot display Decision Tree visualization or feature importance without valid feature names.")


if __name__ == "__main__":
    main()