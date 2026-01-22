
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
import json
import zipfile
import datetime
import matplotlib.pyplot as plt
import shutil
import tempfile
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # For sample model
from sklearn.datasets import make_classification  # For sample data
from sklearn.preprocessing import LabelEncoder  # For sample data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Ensure matplotlib figures are closed to prevent memory leaks in Streamlit
plt.rcParams.update({'figure.max_open_warning': 0})

# --- Start of 'source.py' content integrated into app.py ---

# Global Variables (from source.py)
TARGET_COLUMN = 'target'
RANDOM_SEED = 42
MODEL_PATH = "credit_model_v1.2.pkl"
DATA_PATH = "credit_data_validation.csv"
# These are used for DiCE, can be updated dynamically via globals() in the app
full_data = None
X = None
y = None
credit_model = None
model_hash_val = None
data_hash_val = None
run_id = None # Added for consistent access in summary generation
positive_class_idx_in_model_output = None # Added for consistent access to positive class index

# Helper function to calculate SHA256 hash
def calculate_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)  # Read in 8KB chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

# Functions (from source.py)

# Custom LabelEncoder wrapper for ColumnTransformer
# ColumnTransformer expects transformers to implement fit_transform and transform
# LabelEncoder by itself works on 1D arrays, so a wrapper is needed for DataFrames.
class LabelEncoderWrapper():
    def fit(self, X, y=None):
        self.encoders = {}
        self.feature_names_in_ = X.columns.tolist() # Store input feature names
        self.mode_values = {} # To store mode for handling unseen values

        for col in self.feature_names_in_:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
            # Store the mode for handling unseen categories during transform, if necessary
            self.mode_values[col] = X[col].mode()[0]
        return self

    def transform(self, X):
        transformed_data = []
        for col in self.feature_names_in_:
            le = self.encoders[col]
            known_classes = set(le.classes_)
            # Map unseen values to mode value from training data to prevent ValueError
            series_to_transform = X[col].apply(lambda x: x if x in known_classes else self.mode_values[col])
            transformed_data.append(le.transform(series_to_transform))
        
        # ColumnTransformer expects a 2D array, so stack the results
        return np.column_stack(transformed_data)

    def get_feature_names_out(self, input_features=None):
         # Return the same feature names as input features, as LabelEncoder doesn't change names
         return self.feature_names_in_ or input_features


def generate_sample_data_and_model(model_path, data_path, target_column, random_seed):
    """Generates a sample dataset and a dummy classification model."""
    st.write("Generating sample data and model...")
    np.random.seed(random_seed)

    # Generate synthetic data
    n_samples = 1000
    n_features = 10
    X_synth, y_synth = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=random_seed
    )
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]
    df_X = pd.DataFrame(X_synth, columns=feature_names)
    df_y = pd.DataFrame(y_synth, columns=[target_column])

    # Add some categorical features and non-numeric for better demo
    df_X['EmploymentType'] = np.random.choice(['Salaried', 'Self-Employed', 'Unemployed'], n_samples)
    df_X['LoanPurpose'] = np.random.choice(['Home', 'Car', 'Education', 'Personal'], n_samples)
    df_X['CreditScore'] = np.random.randint(300, 850, n_samples)
    df_X['Income'] = np.random.randint(30000, 200000, n_samples)
    df_X['LoanAmount'] = np.random.randint(5000, 50000, n_samples)
    df_X['Age'] = np.random.randint(18, 70, n_samples)

    # Create full dataset
    full_data_df = pd.concat([df_X, df_y], axis=1)
    
    # Save data
    full_data_df.to_csv(data_path, index=False)

    # Train a dummy model
    # Define preprocessing steps for the model
    categorical_features = ['EmploymentType', 'LoanPurpose']
    numeric_features = [col for col in df_X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', LabelEncoderWrapper(), categorical_features),
            ('num', 'passthrough', numeric_features)
        ],
        remainder='passthrough' # Keep other columns if any, though not expected here
    )

    # Create a pipeline
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', RandomForestClassifier(random_state=random_seed, n_estimators=50, max_depth=5))])

    # Fit the pipeline
    model_pipeline.fit(df_X, y_synth)
    
    # Save model
    joblib.dump(model_pipeline, model_path)
    st.write(f"Sample data saved to {data_path}")
    st.write(f"Sample model saved to {model_path}")

def load_and_hash_artifacts(model_path, data_path, target_column, random_seed):
    """Loads model and data, calculates their hashes, and prepares data for explanation."""
    model = joblib.load(model_path)
    full_data = pd.read_csv(data_path)

    model_hash = calculate_file_hash(model_path)
    data_hash = calculate_file_hash(data_path)

    # Separate features and target
    if target_column not in full_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    X_raw = full_data.drop(columns=[target_column])
    y = full_data[target_column]

    # Apply preprocessing from the model pipeline to X_raw
    X_processed = model.named_steps['preprocessor'].transform(X_raw)

    # Convert X_processed back to DataFrame with feature names
    # Get feature names from preprocessor (needs careful handling for ColumnTransformer)
    if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
        processed_feature_names = model.named_steps['preprocessor'].get_feature_names_out(X_raw.columns)
    else:
        # Fallback for older sklearn versions or custom setups
        # Manually reconstruct based on the order of transformers in ColumnTransformer
        # Based on the pipeline setup: ('cat', LabelEncoderWrapper(), categorical_features) then ('num', 'passthrough', numeric_features)
        
        categorical_features_input = [col for _, _, cols in model.named_steps['preprocessor'].transformers_ if 'cat' in _ for col in cols]
        numeric_features_input = [col for _, _, cols in model.named_steps['preprocessor'].transformers_ if 'num' in _ for col in cols]
        
        # The ColumnTransformer concatenates outputs in the order specified.
        # So, first 'cat' features' output, then 'num' features' output.
        
        # The LabelEncoderWrapper returns the original feature names.
        processed_feature_names = categorical_features_input + numeric_features_input

    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names, index=X_raw.index)

    return model, full_data, X_processed_df, y, model_hash, data_hash

def generate_global_shap_explanation(model, X_train_exp, feature_names, output_dir, positive_class_idx):
    """Generates global SHAP explanations and saves them."""
    # Use the classifier from the pipeline
    classifier = model.named_steps['classifier']
    explainer = shap.TreeExplainer(classifier) # Assumes Tree-based model for now.
    
    # SHAP values for the processed training data
    shap_values = explainer.shap_values(X_train_exp)

    # For binary classification, shap_values is a list [shap_values_class_0, shap_values_class_1]
    # We typically explain the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[positive_class_idx]

    # Calculate mean absolute SHAP values for global importance
    global_importance = np.abs(shap_values).mean(axis=0)
    global_importance_df = pd.DataFrame({
        'Feature': feature_names, # Use feature names from X_train_exp
        'Mean_Abs_SHAP': global_importance
    }).sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

    # Save to JSON
    output_file_path = os.path.join(output_dir, "global_explanation.json")
    global_importance_df.to_json(output_file_path, orient='records', indent=4)
    
    return global_importance_df, output_file_path

def generate_local_shap_explanations(model, X, instance_ids, explainer_local, output_dir, positive_class_idx):
    """Generates local SHAP explanations (waterfall plots) for selected instances."""
    local_explanations_data = {}
    
    for idx in instance_ids:
        instance_X_processed = X.loc[[idx]] # X is already preprocessed
        shap_explanation_object = explainer_local(instance_X_processed) # This returns an Explanation object

        # For binary classification, shap_explanation_object.values will be (1, n_features, 2)
        # We need to create an Explanation object specific to the positive class for waterfall_plot.
        if len(shap_explanation_object.values.shape) == 3: # (n_instances, n_features, n_classes)
            # Ensure base_values is a scalar for waterfall plot if it was an array for multi-class
            base_val_for_plot = shap_explanation_object.base_values[positive_class_idx].item() if isinstance(shap_explanation_object.base_values, np.ndarray) else shap_explanation_object.base_values.item()

            shap_values_for_plot = shap.Explanation(
                values=shap_explanation_object.values[0, :, positive_class_idx],
                base_values=base_val_for_plot,
                data=shap_explanation_object.data[0],
                feature_names=shap_explanation_object.feature_names
            )
        else:
            # If explainer is already class-specific (e.g., KernelExplainer on predict_proba) or regressor
            shap_values_for_plot = shap_explanation_object[0] # Get explanation for the first instance
            
        # Save plot
        plt.figure()
        shap.waterfall_plot(shap_values_for_plot, max_display=10, show=False)
        plot_path = os.path.join(output_dir, f"local_explanation_instance_{idx}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        # Store data for reporting - extract values from the `shap_values_for_plot` Explanation object
        local_explanations_data[idx] = {
            'instance_features': instance_X_processed.iloc[0].to_dict(),
            'shap_values': shap_values_for_plot.values.tolist(),
            'base_value': shap_values_for_plot.base_values.item(),
            'expected_value': explainer_local.expected_value[positive_class_idx].item() if isinstance(explainer_local.expected_value, np.ndarray) else explainer_local.expected_value.item(),
            'prediction': model.predict_proba(instance_X_processed)[0, positive_class_idx].item(),
            'plot_path': os.path.basename(plot_path)
        }
    
    # Save all local explanations to a single JSON
    output_file_path = os.path.join(output_dir, "local_explanation.json")
    with open(output_file_path, 'w') as f:
        json.dump(local_explanations_data, f, indent=4)

    return local_explanations_data, output_file_path

# To use DiCE, we need to import it.
# If dice_ml is not installed, Streamlit will fail.
# For now, I will use a dummy DiCE result if it's not present.
try:
    import dice_ml
except ImportError:
    st.warning("`dice_ml` not found. Counterfactual generation will use a dummy output.")
    dice_ml = None

def generate_counterfactual_explanation(model, X_processed_data, feature_names, instance_id, desired_class, output_dir):
    """Generates a counterfactual explanation for a denied instance."""
    
    # Access global variables directly
    global full_data, X, y, credit_model, model_hash_val, data_hash_val, RANDOM_SEED, TARGET_COLUMN

    original_instance_raw = full_data.loc[[instance_id]].drop(columns=[TARGET_COLUMN])
    original_instance_dict = original_instance_raw.iloc[0].to_dict()

    if dice_ml is None:
        st.warning("DiCE is not installed. Returning dummy counterfactual.")
        dummy_cf = original_instance_raw.copy()
        # Make some dummy changes
        for col in dummy_cf.columns:
            if dummy_cf[col].dtype in [np.float64, np.int64]:
                if col == 'Income': # Example: increase income
                    dummy_cf[col] = dummy_cf[col] * 1.1
                elif col == 'LoanAmount': # Example: decrease loan amount
                    dummy_cf[col] = dummy_cf[col] * 0.9
                else: # Other numeric features
                    dummy_cf[col] += np.random.uniform(-0.05, 0.05) * dummy_cf[col]
            else: # Categorical features
                if col == 'EmploymentType':
                    dummy_cf[col] = 'Salaried' # Example: change employment type
                else:
                    unique_cat_values = full_data[col].dropna().unique()
                    if len(unique_cat_values) > 1:
                        dummy_cf[col] = np.random.choice([val for val in unique_cat_values if val != dummy_cf[col].iloc[0]])
        
        counterfactual_result = {
            'original_instance': original_instance_dict,
            'original_prediction': model.predict_proba(X_processed_data.loc[[instance_id]])[0, desired_class].item(),
            'counterfactual_df': dummy_cf.to_dict(orient='records'),
            'counterfactual_prediction': 0.85, # Assuming it flips to approved
            'changes_text': "Dummy changes: Increase Income, decrease LoanAmount, change EmploymentType (DiCE not available).",
        }
        output_file_path = os.path.join(output_dir, "counterfactual_example.json")
        with open(output_file_path, 'w') as f:
            json.dump(counterfactual_result, f, indent=4)
        return counterfactual_result

    # If DiCE is available, proceed with actual generation
    
    # Use the raw `full_data` for DiCE's Data object, as it infers feature types better
    dice_data_raw = full_data.copy()
    
    # Identify raw continuous and categorical features
    raw_categorical_features = [col for col in dice_data_raw.columns if dice_data_raw[col].dtype == 'object']
    raw_continuous_features = [col for col in dice_data_raw.columns if dice_data_raw[col].dtype in ['int64', 'float64'] and col != TARGET_COLUMN]

    d = dice_ml.Data(
        dataframe=dice_data_raw,
        continuous_features=raw_continuous_features,
        outcome_name=TARGET_COLUMN
    )

    # Create a robust model wrapper for DiCE to handle preprocessing
    class CustomModelWrapper:
        def __init__(self, pipeline_model):
            self.pipeline = pipeline_model
            self.preprocessor = pipeline_model.named_steps['preprocessor']
            self.classifier = pipeline_model.named_steps['classifier']
            self.feature_names_in_order = X_processed_data.columns.tolist() # Expected feature order after preprocessing

        def predict_proba(self, raw_data_df):
            # DiCE generates raw_data_df. It typically contains all features in their raw form.
            # Ensure raw_data_df has the same columns as the original raw data `full_data`
            # and in the same order, then apply preprocessing.

            expected_raw_cols = globals()['full_data'].drop(columns=[globals()['TARGET_COLUMN']]).columns

            # Use .copy() to avoid SettingWithCopyWarning if raw_data_df is a view
            aligned_raw_df = raw_data_df.copy()
            
            # Add missing columns (if any, though DiCE usually ensures this)
            for col in expected_raw_cols:
                if col not in aligned_raw_df.columns:
                    if col in globals()['full_data'].columns:
                        # Fill with a sensible default, e.g., the mode for categorical, mean for numeric
                        if globals()['full_data'][col].dtype == 'object':
                            aligned_raw_df[col] = globals()['full_data'][col].mode()[0]
                        else:
                            aligned_raw_df[col] = globals()['full_data'][col].mean()
                    else:
                        aligned_raw_df[col] = np.nan # Fallback if column not found in full_data
            
            # Reorder columns to match the training data's raw feature order
            aligned_raw_df = aligned_raw_df[expected_raw_cols]

            # Preprocess the aligned raw data
            X_processed_cf = self.preprocessor.transform(aligned_raw_df)
            
            # Convert back to DataFrame to ensure column names and order for classifier
            X_processed_cf_df = pd.DataFrame(X_processed_cf, columns=self.feature_names_in_order, index=raw_data_df.index)

            return self.classifier.predict_proba(X_processed_cf_df)

    m = dice_ml.Model(model=CustomModelWrapper(model), backend="sklearn", model_type='classifier')
    
    # Instantiate DiCE explainer
    exp = dice_ml.Dice(d, m, method="kdtree") # Other methods: "random", "genetic"

    # Find the query instance from the raw data based on instance_id
    query_instance_raw = dice_data_raw.loc[[instance_id]]
    
    # Generate counterfactuals
    dice_exp = exp.generate_counterfactuals(
        query_instance_raw,
        total_CFs=1,
        desired_class=desired_class,
        permitted_range=None # Allow features to change within reasonable bounds or based on feature constraints
    )
    
    # Extract information
    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
    
    # Remove target column from cf_df for display, if DiCE added it
    if TARGET_COLUMN in cf_df.columns:
        cf_df = cf_df.drop(columns=[TARGET_COLUMN])

    # Generate changes text
    changes_text = "No changes required for counterfactual (original already desired class or no valid CF found)."
    if not cf_df.empty:
        original_row = query_instance_raw.iloc[0]
        cf_row = cf_df.iloc[0]
        
        diff_parts = []
        for col in original_row.index:
            if col == TARGET_COLUMN:
                continue
            orig_val = original_row[col]
            cf_val = cf_row[col]
            
            # Check for NaN and type consistency
            if pd.isna(orig_val) and pd.isna(cf_val):
                continue
            
            # Convert to numeric if possible for comparison, but maintain original type for display
            if pd.api.types.is_numeric_dtype(original_row[col]) and pd.api.types.is_numeric_dtype(cf_row[col]):
                if not np.isclose(orig_val, cf_val):
                    change = cf_val - orig_val
                    if change > 0:
                        diff_parts.append(f"increase {col} by {change:.2f}")
                    else:
                        diff_parts.append(f"decrease {col} by {-change:.2f}")
            else: # Categorical or other non-numeric
                if str(orig_val) != str(cf_val): # Compare as strings for robust categorical comparison
                    diff_parts.append(f"change {col} from '{orig_val}' to '{cf_val}'")
        
        if diff_parts:
            changes_text = "To get approved, consider: " + "; ".join(diff_parts) + "."
        else:
            changes_text = "No discernible changes for counterfactual."

    # Get counterfactual prediction
    # Need to process the counterfactual dataframe through the pipeline
    # Use the CustomModelWrapper to ensure proper preprocessing
    custom_model_wrapper_for_prediction = CustomModelWrapper(model)
    cf_prediction_prob = custom_model_wrapper_for_prediction.predict_proba(cf_df)[0, desired_class].item()


    counterfactual_result = {
        'original_instance': original_instance_dict,
        'original_prediction': model.predict_proba(X_processed_data.loc[[instance_id]])[0, desired_class].item(),
        'counterfactual_df': cf_df.to_dict(orient='records'),
        'counterfactual_prediction': cf_prediction_prob,
        'changes_text': changes_text,
    }

    output_file_path = os.path.join(output_dir, "counterfactual_example.json")
    with open(output_file_path, 'w') as f:
        json.dump(counterfactual_result, f, indent=4)

    return counterfactual_result


def create_config_snapshot(model_hash, data_hash, random_seed, output_dir):
    """Creates a configuration snapshot file."""
    config_data = {
        "model_hash": model_hash,
        "data_hash": data_hash,
        "random_seed": random_seed,
        "timestamp": datetime.datetime.now().isoformat()
    }
    config_file_path = os.path.join(output_dir, "config_snapshot.json")
    with open(config_file_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    return config_file_path

def create_evidence_manifest(output_dir, files_to_bundle):
    """Creates a manifest file with hashes of all bundled files."""
    manifest_data = {
        "files": []
    }
    for filepath in files_to_bundle:
        if os.path.exists(filepath):
            file_hash = calculate_file_hash(filepath)
            manifest_data["files"].append({
                "filename": os.path.basename(filepath),
                "path": os.path.relpath(filepath, output_dir), # Path relative to the bundle root
                "sha256_hash": file_hash
            })
    
    manifest_file_path = os.path.join(output_dir, "evidence_manifest.json")
    with open(manifest_file_path, 'w') as f:
        json.dump(manifest_data, f, indent=4)
    return manifest_file_path

def bundle_artifacts_to_zip(explanation_dir, run_id):
    """Bundles all generated artifacts into a single ZIP archive."""
    zip_filename = f"{run_id}_audit_bundle.zip"
    zip_archive_path = os.path.join("reports", zip_filename) # Store zip in 'reports' parent folder

    # Ensure the parent directory for the zip exists
    os.makedirs(os.path.dirname(zip_archive_path), exist_ok=True)
    
    with zipfile.ZipFile(zip_archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(explanation_dir):
            for file in files:
                filepath = os.path.join(root, file)
                # Arcname is the path inside the zip file
                # We want files inside the explanation_dir to be at the root of the zip
                # e.g., explanation_dir/config_snapshot.json -> config_snapshot.json
                arcname = os.path.relpath(filepath, explanation_dir)
                zipf.write(filepath, arcname)
    return zip_archive_path


def generate_explanation_summary(global_importance_df, local_explanations_data, counterfactual_result, output_dir):
    """Generates a markdown summary of all explanations."""
    summary_path = os.path.join(output_dir, "explanation_summary.md")
    
    with open(summary_path, "w") as f:
        f.write("# Model Validation & Explainability Summary Report\n\n")
        f.write(f"**Run ID:** `{globals()['run_id']}`\n") # Access run_id via globals
        f.write(f"**Date:** `{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n")
        f.write(f"**Model Hash:** `{globals()['model_hash_val']}`\n") # Access via globals
        f.write(f"**Data Hash:** `{globals()['data_hash_val']}`\n\n") # Access via globals
        
        f.write("## 1. Global Feature Importance\n")
        f.write("Overall, these features have the most significant impact on the model's predictions.\n\n")
        if not global_importance_df.empty:
            f.write("### Top 10 Features by Mean Absolute SHAP Value\n")
            f.write(global_importance_df.head(10).to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("No global importance data available.\n\n")

        f.write("## 2. Local Instance Explanations\n")
        f.write("Detailed explanations for selected individual instances.\n\n")
        if local_explanations_data:
            for idx, data in local_explanations_data.items():
                f.write(f"### Instance ID: `{idx}`\n")
                f.write(f"**Predicted Probability (Positive Class):** `{data['prediction']:.4f}`\n")
                f.write(f"**Prediction:** {'Approved' if data['prediction'] >= 0.5 else 'Denied'}\n\n")
                
                f.write("**Key Feature Contributions (SHAP Values):**\n")
                # Create a DataFrame for better display of local SHAP values
                if globals()['X'] is not None:
                    # Access feature names from globals()['X'] (preprocessed features)
                    shap_df = pd.DataFrame({
                        'Feature': globals()['X'].columns,
                        'SHAP Value': data['shap_values']
                    }).sort_values(by='SHAP Value', key=abs, ascending=False)
                    f.write(shap_df.head(5).to_markdown(index=False))
                    f.write("\n\n")
                else:
                    f.write("Feature data (X) not available for detailed SHAP values.\n\n")
                
                f.write(f"*(See `local_explanation_instance_{idx}.png` for full waterfall plot.)*\n\n")
        else:
            f.write("No local explanations generated.\n\n")

        f.write("## 3. Counterfactual Analysis\n")
        f.write("For a denied applicant, what minimal changes would lead to an approval?\n\n")
        if counterfactual_result:
            original = counterfactual_result['original_instance']
            cf_df_raw = pd.DataFrame(counterfactual_result['counterfactual_df']) # This is a list of dicts, convert to df
            if not cf_df_raw.empty:
                cf = cf_df_raw.iloc[0].to_dict()
                f.write(f"### Denied Instance ID: `{st.session_state['denied_instance_for_cf_idx']}`\n")
                f.write(f"**Original Prediction Probability:** `{counterfactual_result['original_prediction']:.4f}`\n")
                f.write(f"**Counterfactual Prediction Probability:** `{counterfactual_result['counterfactual_prediction']:.4f}`\n\n")
                f.write("**Original Instance Details:**\n")
                f.write(pd.DataFrame([original]).to_markdown(index=False))
                f.write("\n\n")
                f.write("**Counterfactual Instance Details (Minimum Changes for Approval):**\n")
                f.write(pd.DataFrame([cf]).to_markdown(index=False))
                f.write("\n\n")
                f.write(f"**Actionable Feedback:** {counterfactual_result['changes_text']}\n\n")
            else:
                f.write("No counterfactual found for the selected instance.\n\n")
        else:
            f.write("No counterfactuals generated.\n\n")

        f.write("---")
        f.write("\n**End of Report**\n")

# --- End of 'source.py' content integrated into app.py ---

st.set_page_config(page_title="QuLab: Lab 5: Interpretability & Explainability Control Workbench", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 5: Interpretability & Explainability Control Workbench")
st.divider()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'
    st.session_state['model'] = None
    st.session_state['full_data'] = None  # Raw data with target column
    st.session_state['X'] = None  # Features only (preprocessed for model)
    st.session_state['y'] = None  # Target only
    st.session_state['model_hash_val'] = None
    st.session_state['data_hash_val'] = None
    st.session_state['explainer_local'] = None  # SHAP TreeExplainer instance
    st.session_state['global_importance_df'] = pd.DataFrame()
    st.session_state['local_explanations_data'] = {}
    st.session_state['counterfactual_result'] = {}
    st.session_state['explanation_summary_content'] = ""
    st.session_state['EXPLANATION_DIR'] = None
    st.session_state['run_id'] = None
    st.session_state['instances_for_local_explanation'] = []
    st.session_state['denied_instance_for_cf_idx'] = None
    st.session_state['config_file'] = None
    st.session_state['output_files_to_bundle'] = []
    st.session_state['manifest_file'] = None
    st.session_state['zip_archive_path'] = None
    st.session_state['uploaded_model_file_obj'] = None
    st.session_state['uploaded_data_file_obj'] = None
    st.session_state['sample_data_loaded'] = False
    st.session_state['model_loaded'] = False
    st.session_state['data_ready'] = False
    st.session_state['global_explanations_generated'] = False
    st.session_state['local_explanations_generated'] = False
    st.session_state['counterfactuals_generated'] = False
    st.session_state['summary_generated'] = False
    st.session_state['artifacts_bundled'] = False
    st.session_state['X_train_exp'] = None
    st.session_state['positive_class_idx_in_model_output'] = None
    st.session_state['target_column'] = TARGET_COLUMN  # from integrated source
    st.session_state['random_seed'] = RANDOM_SEED  # from integrated source
    st.session_state['temp_model_path'] = None
    st.session_state['temp_data_path'] = None
    # Create persistent directories for temporary uploads and reports
    os.makedirs("temp_uploads", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------

pages = [
    "Home",
    "1. Data & Model Loading",
    "2. Global Explanations",
    "3. Local Explanations",
    "4. Counterfactuals",
    "5. Validation Summary",
    "6. Export Artifacts"
]

# Use index to set default selection if needed, but here simple navigation suffices
page_selection = st.sidebar.selectbox("Go to", pages, index=pages.index(st.session_state['current_page']))
st.session_state['current_page'] = page_selection

# -----------------------------------------------------------------------------
# Page: Home
# -----------------------------------------------------------------------------

if st.session_state['current_page'] == "Home":
    st.header("Welcome to the Model Validation & Explainability Workbench")
    st.markdown(f"**User Role:** Anya Sharma, Lead Model Validator")
    st.markdown(f"**Mission:** You are tasked with validating PrimeCredit Bank's new Credit Approval Model (CAM v1.2) before it goes into production. Your goal is to ensure the model is transparent, fair, and compliant with regulatory standards.")
    st.markdown(f"**Workflow:**")
    st.markdown(f"1. **Load Data & Model:** Establish a reproducible baseline by hashing artifacts.\n2. **Global Explanations:** Understand the high-level drivers of model decisions.\n3. **Local Explanations:** Audit specific loan applications (denied vs. approved).\n4. **Counterfactuals:** Generate actionable feedback for denied applicants.\n5. **Validation Summary:** Synthesize findings into a report.\n6. **Export Artifacts:** Create a cryptographically signed audit package.")
    st.info("Navigate to '1. Data & Model Loading' to begin your validation workflow.")

# -----------------------------------------------------------------------------
# Page: 1. Data & Model Loading
# -----------------------------------------------------------------------------

elif st.session_state['current_page'] == "1. Data & Model Loading":
    st.header("1. Data & Model Artifact Loading")
    st.markdown(f"Ensure reproducibility by loading specific versions of the model and dataset. Cryptographic hashes (SHA-256) will be generated to track these artifacts throughout the validation lifecycle.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Custom Upload")
        uploaded_model = st.file_uploader("Upload Model (.pkl)", type=["pkl"], key="custom_model_uploader")
        uploaded_data = st.file_uploader("Upload Feature Data (.csv)", type=["csv"], key="custom_data_uploader")
        
        # Update session state file objects if new files are uploaded
        if uploaded_model:
            st.session_state['uploaded_model_file_obj'] = uploaded_model
        if uploaded_data:
            st.session_state['uploaded_data_file_obj'] = uploaded_data

        if st.button("Load Custom Model & Data", disabled=not (st.session_state['uploaded_model_file_obj'] and st.session_state['uploaded_data_file_obj'])):
            with st.spinner("Loading and hashing custom artifacts..."):
                # Clean up previous temp files if they exist
                if st.session_state['temp_model_path'] and os.path.exists(st.session_state['temp_model_path']):
                    os.remove(st.session_state['temp_model_path'])
                if st.session_state['temp_data_path'] and os.path.exists(st.session_state['temp_data_path']):
                    os.remove(st.session_state['temp_data_path'])

                # Save uploaded files to temp paths with unique names
                temp_model_filename = os.path.join("temp_uploads", f"uploaded_model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f')}.pkl")
                with open(temp_model_filename, "wb") as tmp_model:
                    tmp_model.write(st.session_state['uploaded_model_file_obj'].getbuffer())
                    st.session_state['temp_model_path'] = tmp_model.name
                
                temp_data_filename = os.path.join("temp_uploads", f"uploaded_data_{datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f')}.csv")
                with open(temp_data_filename, "wb") as tmp_data:
                    tmp_data.write(st.session_state['uploaded_data_file_obj'].getbuffer())
                    st.session_state['temp_data_path'] = tmp_data.name

                # Load and Hash
                model, full_data_loaded, X_loaded, y_loaded, model_hash, data_hash = load_and_hash_artifacts(
                    st.session_state['temp_model_path'], 
                    st.session_state['temp_data_path'], 
                    st.session_state['target_column'], 
                    st.session_state['random_seed']
                )
                
                # Update Session State
                st.session_state['model'] = model
                st.session_state['full_data'] = full_data_loaded # Raw data including target and original categoricals
                st.session_state['X'] = X_loaded # Preprocessed features for model
                st.session_state['y'] = y_loaded # Target
                st.session_state['model_hash_val'] = model_hash
                st.session_state['data_hash_val'] = data_hash
                st.session_state['model_loaded'] = True
                st.session_state['data_ready'] = True
                st.session_state['sample_data_loaded'] = False
                
                # Prepare Explainer Background for SHAP
                X_train_exp, _, _, _ = train_test_split(
                    st.session_state['X'], 
                    st.session_state['y'], 
                    test_size=0.2, 
                    random_state=st.session_state['random_seed']
                )
                st.session_state['X_train_exp'] = X_train_exp
                
                # SHAP explainer (for the classifier within the pipeline)
                if isinstance(st.session_state['model'].named_steps['classifier'], (RandomForestClassifier, )) : 
                    st.session_state['explainer_local'] = shap.TreeExplainer(st.session_state['model'].named_steps['classifier'])
                else:
                    # For non-tree models, use KernelExplainer with a sample of preprocessed background data
                    # Using a small background dataset for performance
                    background_data_sample = st.session_state['X_train_exp'].iloc[np.random.choice(st.session_state['X_train_exp'].shape[0], min(100, st.session_state['X_train_exp'].shape[0]), replace=False)]
                    st.session_state['explainer_local'] = shap.KernelExplainer(st.session_state['model'].predict_proba, background_data_sample)

                # Identify Positive Class Index
                classes = list(st.session_state['model'].named_steps['classifier'].classes_) # Get classes from classifier
                if 1 in classes:
                    st.session_state['positive_class_idx_in_model_output'] = classes.index(1)
                elif 'Approved' in classes: # Or whatever string represents the positive class
                    st.session_state['positive_class_idx_in_model_output'] = classes.index('Approved')
                else:
                    st.session_state['positive_class_idx_in_model_output'] = 1 # Default to 1 if not explicitly found
                
                # Reset downstream flags
                st.session_state['global_explanations_generated'] = False
                st.session_state['local_explanations_generated'] = False
                st.session_state['counterfactuals_generated'] = False
                st.session_state['summary_generated'] = False
                st.session_state['artifacts_bundled'] = False
                
                st.success("Custom artifacts loaded successfully!")
                st.rerun()

    with col2:
        st.subheader("Sample Environment")
        st.markdown(f"Load the standard `Credit Approval Model v1.2` and validation dataset provided for this lab.")
        if st.button("Load Sample Credit Model & Data"):
            with st.spinner("Generating and loading sample artifacts..."):
                # Clean up existing sample files if they exist (important for reproducibility between runs)
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                if os.path.exists(DATA_PATH):
                    os.remove(DATA_PATH)

                # Generate Sample Data
                generate_sample_data_and_model(
                    MODEL_PATH, 
                    DATA_PATH, 
                    st.session_state['target_column'], 
                    st.session_state['random_seed']
                )
                
                # Load and Hash
                model, full_data_loaded, X_loaded, y_loaded, model_hash, data_hash = load_and_hash_artifacts(
                    MODEL_PATH, 
                    DATA_PATH, 
                    st.session_state['target_column'], 
                    st.session_state['random_seed']
                )
                
                # Update Session State
                st.session_state['model'] = model
                st.session_state['full_data'] = full_data_loaded # Raw data including target and original categoricals
                st.session_state['X'] = X_loaded # Preprocessed features for model
                st.session_state['y'] = y_loaded # Target
                st.session_state['model_hash_val'] = model_hash
                st.session_state['data_hash_val'] = data_hash
                st.session_state['model_loaded'] = True
                st.session_state['data_ready'] = True
                st.session_state['sample_data_loaded'] = True
                
                # Prepare Explainer Background
                X_train_exp, _, _, _ = train_test_split(
                    st.session_state['X'], 
                    st.session_state['y'], 
                    test_size=0.2, 
                    random_state=st.session_state['random_seed']
                )
                st.session_state['X_train_exp'] = X_train_exp
                
                # SHAP explainer (for the classifier within the pipeline)
                if isinstance(st.session_state['model'].named_steps['classifier'], (RandomForestClassifier, )) :
                    st.session_state['explainer_local'] = shap.TreeExplainer(st.session_state['model'].named_steps['classifier'])
                else:
                    background_data_sample = st.session_state['X_train_exp'].iloc[np.random.choice(st.session_state['X_train_exp'].shape[0], min(100, st.session_state['X_train_exp'].shape[0]), replace=False)]
                    st.session_state['explainer_local'] = shap.KernelExplainer(st.session_state['model'].predict_proba, background_data_sample)
                
                # Identify Positive Class Index
                classes = list(st.session_state['model'].named_steps['classifier'].classes_)
                if 1 in classes:
                    st.session_state['positive_class_idx_in_model_output'] = classes.index(1)
                elif 'Approved' in classes:
                    st.session_state['positive_class_idx_in_model_output'] = classes.index('Approved')
                else:
                    st.session_state['positive_class_idx_in_model_output'] = 1
                
                # Reset downstream flags
                st.session_state['global_explanations_generated'] = False
                st.session_state['local_explanations_generated'] = False
                st.session_state['counterfactuals_generated'] = False
                st.session_state['summary_generated'] = False
                st.session_state['artifacts_bundled'] = False
                
                st.success("Sample artifacts loaded successfully!")
                st.rerun()

    st.divider()
    if st.session_state['data_ready']:
        st.subheader("Loaded Artifact Verification")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown(f"**Model SHA-256 Hash:**")
            st.code(st.session_state['model_hash_val'])
        with col_h2:
            st.markdown(f"**Data SHA-256 Hash:**")
            st.code(st.session_state['data_hash_val'])
        
        st.subheader("Feature Data Preview (First 5 Rows - Preprocessed for Model)")
        st.dataframe(st.session_state['X'].head())
        st.subheader("Raw Data Preview (First 5 Rows - Including Target and Original Categoricals)")
        st.dataframe(st.session_state['full_data'].head())


# -----------------------------------------------------------------------------
# Page: 2. Global Explanations
# -----------------------------------------------------------------------------

elif st.session_state['current_page'] == "2. Global Explanations":
    st.header("2. Global Model Interpretability")
    st.markdown(f"Global explanations provide an aggregate view of how features influence the model's predictions overall. We use SHAP (SHapley Additive exPlanations) values to quantify feature importance.")
    
    st.markdown(r"$$ \phi_0 + \sum_{{i=1}}^{{M}} \phi_i(f, x) = f(x) $$ ")
    st.markdown(r"where $\phi_0$ is the average prediction and $\phi_i(f, x)$ is the contribution of feature $i$.")
    
    if st.button("Generate Global Explanations", disabled=not st.session_state['data_ready']):
        with st.spinner("Calculating global SHAP values..."):
            # Generate unique run ID and directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state['run_id'] = f"validation_run_{timestamp}"
            st.session_state['EXPLANATION_DIR'] = os.path.join("reports", f"session_{timestamp}")
            
            # Clean up old explanation dir if it exists (though timestamp ensures uniqueness)
            if os.path.exists(st.session_state['EXPLANATION_DIR']):
                shutil.rmtree(st.session_state['EXPLANATION_DIR'])
            os.makedirs(st.session_state['EXPLANATION_DIR'], exist_ok=True)

            # Update global variables for functions that might access them
            globals()['positive_class_idx_in_model_output'] = st.session_state['positive_class_idx_in_model_output']

            # Generate Explanations
            global_importance_df, _ = generate_global_shap_explanation(
                st.session_state['model'], # Pass the full pipeline
                st.session_state['X_train_exp'],
                st.session_state['X_train_exp'].columns.tolist(),
                st.session_state['EXPLANATION_DIR'],
                st.session_state['positive_class_idx_in_model_output'] # Pass the index
            )
            
            st.session_state['global_importance_df'] = global_importance_df
            st.session_state['global_explanations_generated'] = True
            
            # Reset downstream
            st.session_state['local_explanations_generated'] = False
            st.session_state['counterfactuals_generated'] = False
            st.session_state['summary_generated'] = False
            st.session_state['artifacts_bundled'] = False
            
            st.rerun()

    if st.session_state['global_explanations_generated']:
        st.subheader("Global Feature Importance")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"**Top Features by Mean |SHAP| Value**")
            st.dataframe(st.session_state['global_importance_df'].head(10))
        
        with col2:
            st.markdown(f"**SHAP Summary Plot**")
            
            if st.session_state['explainer_local'] is None: # Should have been initialized in page 1
                if isinstance(st.session_state['model'].named_steps['classifier'], (RandomForestClassifier, )) :
                    explainer = shap.TreeExplainer(st.session_state['model'].named_steps['classifier'])
                else:
                    background_data_sample = st.session_state['X_train_exp'].iloc[np.random.choice(st.session_state['X_train_exp'].shape[0], min(100, st.session_state['X_train_exp'].shape[0]), replace=False)]
                    explainer = shap.KernelExplainer(st.session_state['model'].predict_proba, background_data_sample)
            else:
                explainer = st.session_state['explainer_local']
            
            shap_values = explainer.shap_values(st.session_state['X_train_exp'])
            
            # Handle binary classification case for SHAP values shape
            if isinstance(shap_values, list):
                # For binary classification, shap_values is a list of arrays [class0, class1]. Use class1.
                shap_values_to_plot = shap_values[st.session_state['positive_class_idx_in_model_output']]
            else:
                shap_values_to_plot = shap_values

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_to_plot, st.session_state['X_train_exp'], show=False)
            st.pyplot(plt.gcf())
            plt.close() # Close figure to free memory

# -----------------------------------------------------------------------------
# Page: 3. Local Explanations
# -----------------------------------------------------------------------------

elif st.session_state['current_page'] == "3. Local Explanations":
    st.header("3. Local Instance Explanations")
    st.markdown(r"For a specific instance $x$, the SHAP values $\phi_i(f, x)$ quantify how much each feature $i$ contributes to the prediction $f(x)$ compared to the average prediction $\phi_0$. A positive SHAP value for a feature means it pushed the prediction higher (towards approval), while a negative value pushed it lower (towards denial).")

    if not st.session_state['data_ready']:
        st.warning("Please load data and model first.")
    else:
        # Default Selection Logic
        default_indices = []
        if st.session_state['model'] is not None and st.session_state['X'] is not None:
            # Calculate probabilities to find interesting cases
            probs = st.session_state['model'].predict_proba(st.session_state['X'])[:, st.session_state['positive_class_idx_in_model_output']]
            
            # Find one denied (prob < 0.5), one approved (prob >= 0.5), one borderline (prob close to 0.5)
            # Use original data index
            denied_candidates = st.session_state['X'].index[probs < 0.5].tolist()
            approved_candidates = st.session_state['X'].index[probs >= 0.5].tolist()
            
            # Find borderline by sorting absolute difference from 0.5
            borderline_indices = st.session_state['X'].index[np.argsort(np.abs(probs - 0.5))].tolist()
            
            if denied_candidates:
                default_indices.append(denied_candidates[0])
            if approved_candidates:
                if approved_candidates[0] not in default_indices: # Avoid duplicate if only one instance
                    default_indices.append(approved_candidates[0])
            if borderline_indices:
                for b_idx in borderline_indices:
                    if b_idx not in default_indices:
                        default_indices.append(b_idx)
                        break
            
            # Ensure default_indices does not exceed 3
            default_indices = default_indices[:3]
                        
        selected_instances = st.multiselect(
            "Select up to 3 instance IDs to explain",
            options=st.session_state['X'].index.tolist(),
            default=default_indices,
            max_selections=3
        )
        st.session_state['instances_for_local_explanation'] = selected_instances

        if st.button("Generate Local Explanations", disabled=len(selected_instances) == 0):
            with st.spinner("Generating local SHAP waterfall plots..."):
                # Ensure EXPLANATION_DIR is set, especially if skipping Global Explanations
                if st.session_state['EXPLANATION_DIR'] is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state['run_id'] = f"validation_run_{timestamp}"
                    st.session_state['EXPLANATION_DIR'] = os.path.join("reports", f"session_{timestamp}")
                    os.makedirs(st.session_state['EXPLANATION_DIR'], exist_ok=True)

                # Ensure explainer_local is initialized
                if st.session_state['explainer_local'] is None:
                    if isinstance(st.session_state['model'].named_steps['classifier'], (RandomForestClassifier, )):
                        st.session_state['explainer_local'] = shap.TreeExplainer(st.session_state['model'].named_steps['classifier'])
                    else:
                        background_data_sample = st.session_state['X_train_exp'].iloc[np.random.choice(st.session_state['X_train_exp'].shape[0], min(100, st.session_state['X_train_exp'].shape[0]), replace=False)]
                        st.session_state['explainer_local'] = shap.KernelExplainer(st.session_state['model'].predict_proba, background_data_sample)

                # Update global variables for functions that might access them
                globals()['positive_class_idx_in_model_output'] = st.session_state['positive_class_idx_in_model_output']

                local_data, _ = generate_local_shap_explanations(
                    st.session_state['model'], # Pass the full pipeline
                    st.session_state['X'], # X is already preprocessed
                    st.session_state['instances_for_local_explanation'],
                    st.session_state['explainer_local'],
                    st.session_state['EXPLANATION_DIR'],
                    st.session_state['positive_class_idx_in_model_output'] # Pass the index
                )
                st.session_state['local_explanations_data'] = local_data
                st.session_state['local_explanations_generated'] = True
                
                # Reset downstream
                st.session_state['counterfactuals_generated'] = False
                st.session_state['summary_generated'] = False
                st.session_state['artifacts_bundled'] = False
                
                st.rerun()

        if st.session_state['local_explanations_generated']:
            st.divider()
            for idx in st.session_state['instances_for_local_explanation']:
                st.subheader(f"Analysis for Instance ID: {idx}")
                col_info, col_plot = st.columns([1, 2])
                
                instance_X_processed = st.session_state['X'].loc[[idx]] # X is already preprocessed
                
                # Handle case where explainer might not have been initialized or model not loaded
                if st.session_state['explainer_local'] is None:
                    try:
                        if isinstance(st.session_state['model'].named_steps['classifier'], (RandomForestClassifier, )) :
                            st.session_state['explainer_local'] = shap.TreeExplainer(st.session_state['model'].named_steps['classifier'])
                        else:
                            background_data_sample = st.session_state['X_train_exp'].iloc[np.random.choice(st.session_state['X_train_exp'].shape[0], min(100, st.session_state['X_train_exp'].shape[0]), replace=False)]
                            st.session_state['explainer_local'] = shap.KernelExplainer(st.session_state['model'].predict_proba, background_data_sample)

                    except Exception as e:
                        st.error(f"Error initializing SHAP Explainer: {e}. Please ensure model is loaded correctly.")
                        continue

                shap_values_single = st.session_state['explainer_local'](instance_X_processed)
                
                # Create a proper Explanation object for waterfall plot
                if len(shap_values_single.values.shape) == 3:
                     positive_class_idx_in_model_output = st.session_state['positive_class_idx_in_model_output']
                     base_val_for_plot_display = shap_values_single.base_values[positive_class_idx_in_model_output].item() if isinstance(shap_values_single.base_values, np.ndarray) else shap_values_single.base_values.item()
                     shap_explanation_for_waterfall = shap.Explanation(
                         values=shap_values_single.values[0, :, positive_class_idx_in_model_output],
                         base_values=base_val_for_plot_display,
                         data=shap_values_single.data[0],
                         feature_names=shap_values_single.feature_names
                     )
                else:
                     shap_explanation_for_waterfall = shap_values_single[0] # For single-output, Explanation[0] gives Explanation for that instance

                with col_info:
                    # Prediction
                    prob = st.session_state['model'].predict_proba(instance_X_processed)[0, st.session_state['positive_class_idx_in_model_output']]
                    status = "Approved" if prob >= 0.5 else "Denied"
                    st.metric("Prediction Probability", f"{prob:.4f}", delta=status)
                    
                    st.markdown(f"**Feature Values (Preprocessed for Model):**")
                    st.dataframe(instance_X_processed.T)
                    
                    # Also show raw feature values from full_data
                    st.markdown(f"**Raw Feature Values:**")
                    st.dataframe(st.session_state['full_data'].loc[[idx]].drop(columns=[st.session_state['target_column']]).T)


                with col_plot:
                    st.markdown(f"**Contribution Waterfall Plot**")
                    plt.figure(figsize=(10, 6))
                    shap.waterfall_plot(shap_explanation_for_waterfall, max_display=10, show=False)
                    st.pyplot(plt.gcf())
                    plt.close() # Close figure to free memory
                st.divider()

# -----------------------------------------------------------------------------
# Page: 4. Counterfactuals
# -----------------------------------------------------------------------------

elif st.session_state['current_page'] == "4. Counterfactuals":
    st.header("4. Counterfactual Analysis")
    st.markdown(f"Counterfactuals help us understand what minimal changes would flip a decision from 'Denied' to 'Approved'.")
    
    st.markdown(r"$$ \min_{{x'}} \text{{distance}}(x, x') \quad \text{{s.t.}} \quad f(x') = y' \quad \text{{and}} \quad x' \in \mathcal{{X}} $$ ")
    st.markdown(r"where $\text{{distance}}(x, x')$ measures the effort to change features.")

    if not st.session_state['data_ready']:
        st.warning("Please load data and model first.")
    else:
        # Identify Denied Instances
        probs = st.session_state['model'].predict_proba(st.session_state['X'])[:, st.session_state['positive_class_idx_in_model_output']]
        denied_indices = st.session_state['X'].index[probs < 0.5].tolist()
        
        if not denied_indices:
            st.warning("No denied instances found in the dataset to generate counterfactuals for. All instances are predicted as Approved.")
            st.session_state['denied_instance_for_cf_idx'] = None
        else:
            if st.session_state['denied_instance_for_cf_idx'] is None or st.session_state['denied_instance_for_cf_idx'] not in denied_indices:
                # Set a default if none selected or the previously selected is no longer denied
                st.session_state['denied_instance_for_cf_idx'] = denied_indices[0]

            selected_denied_id = st.selectbox(
                "Select a denied instance ID to analyze",
                options=denied_indices,
                index=denied_indices.index(st.session_state['denied_instance_for_cf_idx']) if st.session_state['denied_instance_for_cf_idx'] in denied_indices else 0
            )
            st.session_state['denied_instance_for_cf_idx'] = selected_denied_id

            if st.button("Generate Counterfactual Example"):
                with st.spinner("Computing counterfactuals using DiCE..."):
                    # Ensure EXPLANATION_DIR is set, especially if skipping Global/Local Explanations
                    if st.session_state['EXPLANATION_DIR'] is None:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.session_state['run_id'] = f"validation_run_{timestamp}"
                        st.session_state['EXPLANATION_DIR'] = os.path.join("reports", f"session_{timestamp}")
                        os.makedirs(st.session_state['EXPLANATION_DIR'], exist_ok=True)
                        
                    # EXPLICITLY UPDATE GLOBAL VARIABLES (now integrated functions can access them)
                    globals()['full_data'] = st.session_state['full_data']
                    globals()['X'] = st.session_state['X']
                    globals()['y'] = st.session_state['y']
                    globals()['credit_model'] = st.session_state['model'] # Renamed for clarity in integrated functions
                    globals()['model_hash_val'] = st.session_state['model_hash_val']
                    globals()['data_hash_val'] = st.session_state['data_hash_val']
                    globals()['RANDOM_SEED'] = st.session_state['random_seed']
                    globals()['TARGET_COLUMN'] = st.session_state['target_column']
                    globals()['positive_class_idx_in_model_output'] = st.session_state['positive_class_idx_in_model_output']
                    
                    if st.session_state['sample_data_loaded']:
                         globals()['MODEL_PATH'] = MODEL_PATH
                         globals()['DATA_PATH'] = DATA_PATH
                    else:
                         globals()['MODEL_PATH'] = st.session_state['temp_model_path']
                         globals()['DATA_PATH'] = st.session_state['temp_data_path']

                    counterfactual_result = generate_counterfactual_explanation(
                        st.session_state['model'], # Pass the full pipeline
                        st.session_state['X'], # X (preprocessed) is passed, but full_data (raw) is accessed via globals in the function
                        st.session_state['X'].columns.tolist(),
                        st.session_state['denied_instance_for_cf_idx'],
                        st.session_state['positive_class_idx_in_model_output'], # Desired class (e.g., 1 for Approved)
                        st.session_state['EXPLANATION_DIR']
                    )
                    st.session_state['counterfactual_result'] = counterfactual_result
                    st.session_state['counterfactuals_generated'] = True
                    
                    # Reset downstream
                    st.session_state['summary_generated'] = False
                    st.session_state['artifacts_bundled'] = False
                    
                    st.rerun()

            if st.session_state['counterfactuals_generated']:
                st.subheader("Counterfactual Result")
                cf_data = st.session_state['counterfactual_result']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Original Instance (Denied - Raw Features)**")
                    # Display original instance using raw features from full_data
                    original_raw_df = st.session_state['full_data'].loc[[st.session_state['denied_instance_for_cf_idx']]].drop(columns=[st.session_state['target_column']])
                    st.dataframe(original_raw_df.T)
                with col2:
                    st.markdown(f"**Counterfactual Instance (Approved - Raw Features)**")
                    # DiCE returns raw features, so display directly
                    if isinstance(cf_data['counterfactual_df'], list):
                        st.dataframe(pd.DataFrame(cf_data['counterfactual_df']).T)
                    else: # If it's a DataFrame already
                        st.dataframe(cf_data['counterfactual_df'].T)

                if cf_data.get('changes_text'):
                    st.success(f"Actionable Feedback: {cf_data['changes_text']}")
                else:
                    st.warning("No actionable feedback generated for this instance.")


# -----------------------------------------------------------------------------
# Page: 5. Validation Summary
# -----------------------------------------------------------------------------

elif st.session_state['current_page'] == "5. Validation Summary":
    st.header("5. Validation & Interpretability Report")
    st.markdown(f"Synthesize the findings from Global, Local, and Counterfactual analyses into a cohesive validation summary.")

    if st.button("Generate Explanation Summary", disabled=not (st.session_state['global_explanations_generated'] and st.session_state['local_explanations_generated'])):
        with st.spinner("Generating report..."):
            # Ensure EXPLANATION_DIR is set, especially if skipping Global/Local Explanations
            if st.session_state['EXPLANATION_DIR'] is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state['run_id'] = f"validation_run_{timestamp}"
                st.session_state['EXPLANATION_DIR'] = os.path.join("reports", f"session_{timestamp}")
                os.makedirs(st.session_state['EXPLANATION_DIR'], exist_ok=True)
                
            # EXPLICITLY UPDATE GLOBAL VARIABLES (now integrated) for `generate_explanation_summary`
            globals()['model_hash_val'] = st.session_state['model_hash_val']
            globals()['data_hash_val'] = st.session_state['data_hash_val']
            globals()['TARGET_COLUMN'] = st.session_state['target_column']
            globals()['X'] = st.session_state['X'] # Required for local shap df creation in summary
            globals()['run_id'] = st.session_state['run_id'] # Also used by summary

            generate_explanation_summary(
                st.session_state['global_importance_df'],
                st.session_state['local_explanations_data'],
                st.session_state['counterfactual_result'],
                st.session_state['EXPLANATION_DIR']
            )
            
            # Read the generated markdown file
            summary_path = os.path.join(st.session_state['EXPLANATION_DIR'], "explanation_summary.md")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    st.session_state['explanation_summary_content'] = f.read()
                st.session_state['summary_generated'] = True
                st.session_state['artifacts_bundled'] = False
                st.rerun()
            else:
                st.error("Summary file was not found. Please ensure explanations were generated successfully.")
                st.session_state['summary_generated'] = False


    if st.session_state['summary_generated']:
        st.markdown("---")
        st.markdown(st.session_state['explanation_summary_content'])

# -----------------------------------------------------------------------------
# Page: 6. Export Artifacts
# -----------------------------------------------------------------------------

elif st.session_state['current_page'] == "6. Export Artifacts":
    st.header("6. Export Validation Artifacts")
    st.markdown(f"Bundle all evidence, configuration snapshots, and explanation reports into a signed, reproducible ZIP archive for audit purposes.")
    
    st.markdown(r"$$ \text{{SHA-256}}(\text{{file\_content}}) = \text{{hexadecimal\_hash\_string}} $$ ")
    st.markdown(r"where $\text{{SHA-256}}$ is the cryptographic hash function ensuring data integrity.")

    if st.button("Export Audit-Ready Bundle (.zip)", disabled=not st.session_state['summary_generated']):
        with st.spinner("Bundling artifacts..."):
            # Ensure EXPLANATION_DIR and run_id are set
            if st.session_state['EXPLANATION_DIR'] is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state['run_id'] = f"validation_run_{timestamp}"
                st.session_state['EXPLANATION_DIR'] = os.path.join("reports", f"session_{timestamp}")
                os.makedirs(st.session_state['EXPLANATION_DIR'], exist_ok=True)

            # EXPLICITLY UPDATE GLOBAL VARIABLES (now integrated)
            globals()['model_hash_val'] = st.session_state['model_hash_val']
            globals()['data_hash_val'] = st.session_state['data_hash_val']
            globals()['RANDOM_SEED'] = st.session_state['random_seed']
            globals()['TARGET_COLUMN'] = st.session_state['target_column']
            
            if st.session_state['sample_data_loaded']:
                 globals()['MODEL_PATH'] = MODEL_PATH
                 globals()['DATA_PATH'] = DATA_PATH
            else:
                 globals()['MODEL_PATH'] = st.session_state['temp_model_path']
                 globals()['DATA_PATH'] = st.session_state['temp_data_path']

            # Create Config Snapshot
            config_file = create_config_snapshot(
                st.session_state['model_hash_val'],
                st.session_state['data_hash_val'],
                st.session_state['random_seed'],
                st.session_state['EXPLANATION_DIR']
            )
            st.session_state['config_file'] = config_file
            
            # Define files to bundle - check if they actually exist
            # Paths inside EXPLANATION_DIR are specific filenames
            files_to_check = [
                os.path.join(st.session_state['EXPLANATION_DIR'], "global_explanation.json"),
                os.path.join(st.session_state['EXPLANATION_DIR'], "local_explanation.json"),
                os.path.join(st.session_state['EXPLANATION_DIR'], "counterfactual_example.json"),
                os.path.join(st.session_state['EXPLANATION_DIR'], "explanation_summary.md"),
                config_file
            ]
            
            # Add any generated local explanation plot files (PNGs)
            for idx in st.session_state['instances_for_local_explanation']:
                plot_path = os.path.join(st.session_state['EXPLANATION_DIR'], f"local_explanation_instance_{idx}.png")
                files_to_check.append(plot_path)

            st.session_state['output_files_to_bundle'] = [f for f in files_to_check if os.path.exists(f)]

            if not st.session_state['output_files_to_bundle']:
                st.error("No explanation files found to bundle. Please ensure all previous steps were completed.")
                st.session_state['artifacts_bundled'] = False
                st.stop()
            
            # Create Manifest
            manifest_file = create_evidence_manifest(
                st.session_state['EXPLANATION_DIR'],
                st.session_state['output_files_to_bundle']
            )
            st.session_state['manifest_file'] = manifest_file
            st.session_state['output_files_to_bundle'].append(manifest_file)
            
            # Zip
            zip_archive_path = bundle_artifacts_to_zip(
                st.session_state['EXPLANATION_DIR'],
                st.session_state['run_id']
            )
            st.session_state['zip_archive_path'] = zip_archive_path
            st.session_state['artifacts_bundled'] = True
            st.rerun()

    if st.session_state['artifacts_bundled']:
        st.success("Artifacts bundled successfully!")
        st.markdown(f"**Evidence Manifest:** `{os.path.basename(st.session_state['manifest_file'])}` created.")
        st.markdown(f"**Archive:** `{os.path.basename(st.session_state['zip_archive_path'])}` ready for download.")
        
        with open(st.session_state['zip_archive_path'], "rb") as f:
            st.download_button(
                label="Download Audit Bundle (.zip)",
                data=f,
                file_name=os.path.basename(st.session_state['zip_archive_path']),
                mime="application/zip"
            )

