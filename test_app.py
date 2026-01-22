
import pytest
from streamlit.testing.v1 import AppTest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
import joblib
import hashlib
import json
import datetime
import zipfile

# --- Dummy Source.py content (for local testing environment setup) ---
# In a real scenario, this would be in a separate file named 'source.py'

TARGET_COLUMN = 'target'
RANDOM_SEED = 42
MODEL_PATH = 'dummy_model.pkl'
DATA_PATH = 'dummy_data.csv'

# Dummy global variables that will be updated by the app
full_data = None
X = None
y = None
credit_model = None
model_hash_val = None
data_hash_val = None

class DummyModel:
    def __init__(self):
        self.classes_ = [0, 1]

    def predict_proba(self, X_input):
        if isinstance(X_input, pd.DataFrame) and not X_input.empty:
            num_rows = len(X_input)
        else:
            num_rows = 1 
        
        if X_input is not None and 'feature1' in X_input.columns:
            probs_class1 = np.where(X_input['feature1'] > 0.5, 0.8, 0.2)
            probs_class0 = 1 - probs_class1
            return np.array(list(zip(probs_class0, probs_class1)))
        else:
            return np.array([[0.8, 0.2]] * num_rows)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class MockShapExplanation:
    def __init__(self, values, base_values, data, feature_names):
        self.values = values
        self.base_values = base_values
        self.data = data
        self.feature_names = feature_names

    @property
    def output_names(self):
        return ["class_0", "class_1"]

class MockShapTreeExplainer:
    def __init__(self, model):
        self.model = model
        self.expected_output_dims = 2
    
    def __call__(self, X):
        values = np.random.rand(X.shape[0], X.shape[1], self.expected_output_dims)
        base_values = np.random.rand(self.expected_output_dims)
        return MockShapExplanation(values, base_values, X.values, X.columns.tolist())

    def shap_values(self, X):
        return [np.random.rand(X.shape[0], X.shape[1]), np.random.rand(X.shape[0], X.shape[1])]

def load_and_hash_artifacts(model_path, data_path, target_column, random_seed):
    dummy_data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        target_column: np.random.randint(0, 2, 100)
    })
    
    dummy_data.loc[0:49, 'feature1'] = np.linspace(0.1, 0.4, 50) 
    dummy_data.loc[50:99, 'feature1'] = np.linspace(0.6, 0.9, 50) 

    dummy_model = DummyModel()
    
    global full_data, X, y, credit_model, model_hash_val, data_hash_val
    full_data = dummy_data
    X = dummy_data.drop(columns=[target_column])
    y = dummy_data[target_column]
    credit_model = dummy_model
    model_hash_val = hashlib.sha256(b"dummy_model_content").hexdigest()
    data_hash_val = hashlib.sha256(b"dummy_data_content").hexdigest()

    return dummy_model, dummy_data, X, y, model_hash_val, data_hash_val

def generate_sample_data_and_model(model_path, data_path, target_column, random_seed):
    return load_and_hash_artifacts(model_path, data_path, target_column, random_seed)

def generate_global_shap_explanation(model, X_train_exp, feature_names, explanation_dir):
    dummy_importance_df = pd.DataFrame({
        'Feature': ['feature1', 'feature2', 'feature3'],
        'Mean |SHAP| Value': [0.5, 0.3, 0.2]
    })
    os.makedirs(explanation_dir, exist_ok=True)
    with open(os.path.join(explanation_dir, "global_explanation.json"), "w") as f:
        json.dump(dummy_importance_df.to_dict(), f)
    
    return dummy_importance_df, "dummy_plot_path.png"

def generate_local_shap_explanations(model, X, instance_ids, explainer_local, explanation_dir):
    dummy_local_data = {}
    for idx in instance_ids:
        instance_X = X.loc[[idx]]
        shap_feature1 = 0.3 if instance_X['feature1'].iloc[0] > 0.5 else -0.3
        shap_feature2 = np.random.uniform(-0.1, 0.1)
        shap_feature3 = np.random.uniform(-0.1, 0.1)
        
        dummy_local_data[idx] = {
            'shap_values': [shap_feature1, shap_feature2, shap_feature3], 
            'feature_values': instance_X.iloc[0].to_dict()
        }
    
    os.makedirs(explanation_dir, exist_ok=True)
    with open(os.path.join(explanation_dir, "local_explanation.json"), "w") as f:
        json.dump(dummy_local_data, f, cls=NumpyEncoder)
        
    return dummy_local_data, "dummy_local_plot.png"

def generate_counterfactual_explanation(model, X, feature_names, denied_instance_idx, target_class, explanation_dir):
    original_instance = X.loc[[denied_instance_idx]].to_dict('records')[0]
    
    cf_feature1 = original_instance['feature1'] + 0.2
    if cf_feature1 > 1.0: cf_feature1 = 1.0 
    
    counterfactual_df = pd.DataFrame({
        'feature1': [cf_feature1],
        'feature2': [original_instance['feature2']],
        'feature3': [original_instance['feature3']]
    }, index=[denied_instance_idx])
    
    changes_text = f"Increase 'feature1' from {original_instance['feature1']:.2f} to {cf_feature1:.2f}."
    
    os.makedirs(explanation_dir, exist_ok=True)
    with open(os.path.join(explanation_dir, "counterfactual_example.json"), "w") as f:
        json.dump({
            'original_instance': original_instance,
            'counterfactual_df': counterfactual_df.to_dict(), # Convert DataFrame to dict
            'changes_text': changes_text
        }, f, cls=NumpyEncoder)

    return {
        'original_instance': original_instance,
        'counterfactual_df': counterfactual_df,
        'changes_text': changes_text
    }

def generate_explanation_summary(global_importance_df, local_explanations_data, counterfactual_result, explanation_dir):
    summary_content = f"""# Validation Summary
    
    **Model Hash:** {model_hash_val}
    **Data Hash:** {data_hash_val}
    **Target Column:** {TARGET_COLUMN}

    ## Global Insights
    Based on the global SHAP explanations, the most important features are:
    1. {global_importance_df['Feature'].iloc[0]} (Mean |SHAP| Value: {global_importance_df['Mean |SHAP| Value'].iloc[0]:.2f})
    2. {global_importance_df['Feature'].iloc[1]} (Mean |SHAP| Value: {global_importance_df['Mean |SHAP| Value'].iloc[1]:.2f})

    ## Local Explanations
    Examined {len(local_explanations_data)} instances. For instance ID {list(local_explanations_data.keys())[0]}, feature '{list(local_explanations_data[list(local_explanations_data.keys())[0]]['feature_values'].keys())[0]}' had a significant impact.

    ## Counterfactual Analysis
    For a denied instance, a counterfactual was generated by making the following changes:
    {counterfactual_result['changes_text']}
    """
    
    os.makedirs(explanation_dir, exist_ok=True)
    summary_path = os.path.join(explanation_dir, "explanation_summary.md")
    with open(summary_path, "w") as f:
        f.write(summary_content)
    return summary_path

def create_config_snapshot(model_hash, data_hash, random_seed, explanation_dir):
    config_content = {
        "model_hash": model_hash,
        "data_hash": data_hash,
        "random_seed": random_seed
    }
    config_path = os.path.join(explanation_dir, "config_snapshot.json")
    with open(config_path, "w") as f:
        json.dump(config_content, f)
    return config_path

def create_evidence_manifest(explanation_dir, files_to_bundle):
    manifest_content = {
        "run_id": "dummy_run_id",
        "timestamp": "2023-01-01T12:00:00", 
        "files": {os.path.basename(f): hashlib.sha256(b"dummy_file_content").hexdigest() for f in files_to_bundle}
    }
    manifest_path = os.path.join(explanation_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_content, f)
    return manifest_path

def bundle_artifacts_to_zip(explanation_dir, run_id):
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    zip_path = os.path.join(reports_dir, f"{run_id}.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.writestr("dummy_file_in_zip.txt", "This is a dummy file.")
    return zip_path

import sys
sys.modules['source'] = sys.modules[__name__]

# --- End Dummy Source.py content ---

@pytest.fixture(autouse=True)
def mock_plotting_libraries():
    with patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.gcf') as mock_gcf, \
         patch('matplotlib.pyplot.show'), \
         patch('matplotlib.pyplot.clf'), \
         patch('matplotlib.pyplot.close'), \
         patch('streamlit.pyplot') as mock_st_pyplot, \
         patch('shap.summary_plot'), \
         patch('shap.waterfall_plot'), \
         patch('shap.TreeExplainer', new=MockShapTreeExplainer):
        
        mock_figure_instance = MagicMock()
        mock_figure.return_value = mock_figure_instance
        mock_gcf.return_value = mock_figure_instance

        yield

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    os.makedirs("temp_uploads", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    yield

    if os.path.exists("temp_uploads"):
        shutil.rmtree("temp_uploads")
    if os.path.exists("reports"):
        shutil.rmtree("reports")

def test_home_page_initial_content():
    at = AppTest.from_file("app.py").run()
    assert at.markdown[0].value == f"**User Role:** Anya Sharma, Lead Model Validator"
    assert "Welcome to the Model Validation & Explainability Workbench" in at.header[0].value
    assert at.info[0].value == "Navigate to '1. Data & Model Loading' to begin your validation workflow."

def test_load_sample_data_and_model():
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("1. Data & Model Loading").run()
    
    # Assert buttons are initially enabled/disabled correctly
    assert at.button[0].disabled == True # Load Custom Model & Data
    assert at.button[1].disabled == False # Load Sample Credit Model & Data

    at.button[1].click().run() # Click "Load Sample Credit Model & Data"

    assert at.success[0].value == "Sample artifacts loaded successfully!"
    assert at.session_state['model_loaded'] == True
    assert at.session_state['data_ready'] == True
    assert at.session_state['sample_data_loaded'] == True
    assert at.session_state['model_hash_val'] is not None
    assert at.session_state['data_hash_val'] is not None
    assert "Model SHA-256 Hash" in at.subheader[2].value
    assert "Data SHA-256 Hash" in at.subheader[3].value
    assert at.dataframe[0].value is not None # Feature Data Preview

def test_global_explanations_workflow():
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("1. Data & Model Loading").run()
    at.button[1].click().run() # Load Sample Data

    at.sidebar.selectbox[0].set_value("2. Global Explanations").run()
    
    assert at.button[0].disabled == False # "Generate Global Explanations" button should be enabled

    at.button[0].click().run()

    assert at.session_state['global_explanations_generated'] == True
    assert "Global Feature Importance" in at.subheader[0].value
    assert at.dataframe[0].value is not None # Global importance dataframe
    assert "SHAP Summary Plot" in at.markdown[2].value # Check for plot header
    # The actual plot rendering is mocked, so we check for the text/element existence

def test_local_explanations_workflow():
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("1. Data & Model Loading").run()
    at.button[1].click().run() # Load Sample Data

    at.sidebar.selectbox[0].set_value("2. Global Explanations").run()
    at.button[0].click().run() # Generate Global Explanations (prerequisite for some downstream flags)

    at.sidebar.selectbox[0].set_value("3. Local Explanations").run()

    # The multiselect should have default values, so button should be enabled
    assert at.button[0].disabled == False # "Generate Local Explanations"

    # Select some instances for explanation
    # AppTest automatically selects defaults based on app logic, so just clicking the button is enough.
    at.button[0].click().run()

    assert at.session_state['local_explanations_generated'] == True
    assert len(at.session_state['instances_for_local_explanation']) > 0
    assert "Analysis for Instance ID" in at.subheader[0].value # Check if local explanations are displayed
    assert at.metric[0].value is not None # Prediction Probability
    assert at.dataframe[0].value is not None # Feature Values
    # Plot is mocked, so just check existence of related markdown
    assert "Contribution Waterfall Plot" in at.markdown[2].value


def test_counterfactuals_workflow():
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("1. Data & Model Loading").run()
    at.button[1].click().run() # Load Sample Data

    at.sidebar.selectbox[0].set_value("2. Global Explanations").run()
    at.button[0].click().run() 

    at.sidebar.selectbox[0].set_value("3. Local Explanations").run()
    at.button[0].click().run() 

    at.sidebar.selectbox[0].set_value("4. Counterfactuals").run()

    # Selectbox for denied instance should be present and enabled if denied instances exist
    assert at.selectbox[0].disabled == False

    # The selectbox should have a default selected denied instance
    assert at.button[0].disabled == False # "Generate Counterfactual Example"

    at.button[0].click().run()

    assert at.session_state['counterfactuals_generated'] == True
    assert "Counterfactual Result" in at.subheader[0].value
    assert "Original Instance (Denied)" in at.markdown[1].value
    assert "Counterfactual Instance (Approved)" in at.markdown[2].value
    assert at.dataframe[0].value is not None # Original instance dataframe
    assert at.dataframe[1].value is not None # Counterfactual instance dataframe
    assert "Actionable Feedback:" in at.success[0].value


def test_validation_summary_workflow():
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("1. Data & Model Loading").run()
    at.button[1].click().run() # Load Sample Data

    at.sidebar.selectbox[0].set_value("2. Global Explanations").run()
    at.button[0].click().run() 

    at.sidebar.selectbox[0].set_value("3. Local Explanations").run()
    at.button[0].click().run() 

    at.sidebar.selectbox[0].set_value("4. Counterfactuals").run()
    if at.selectbox[0] and at.button[0].disabled == False: # Ensure a denied instance is selected before generating CF
        at.button[0].click().run() # Generate Counterfactual Example
    else:
        pytest.skip("Skipping counterfactuals and summary tests as no denied instances found or button disabled.")


    at.sidebar.selectbox[0].set_value("5. Validation Summary").run()
    
    assert at.button[0].disabled == False # "Generate Explanation Summary" button should be enabled

    at.button[0].click().run()

    assert at.session_state['summary_generated'] == True
    assert "Validation Summary" in at.markdown[1].value # Check for summary content

def test_export_artifacts_workflow():
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("1. Data & Model Loading").run()
    at.button[1].click().run() # Load Sample Data

    at.sidebar.selectbox[0].set_value("2. Global Explanations").run()
    at.button[0].click().run() 

    at.sidebar.selectbox[0].set_value("3. Local Explanations").run()
    at.button[0].click().run() 

    at.sidebar.selectbox[0].set_value("4. Counterfactuals").run()
    if at.selectbox[0] and at.button[0].disabled == False:
        at.button[0].click().run() 
    else:
        pytest.skip("Skipping counterfactuals and export tests as no denied instances found or button disabled.")

    at.sidebar.selectbox[0].set_value("5. Validation Summary").run()
    at.button[0].click().run() 

    at.sidebar.selectbox[0].set_value("6. Export Artifacts").run()

    assert at.button[0].disabled == False # "Export Audit-Ready Bundle (.zip)" button should be enabled

    at.button[0].click().run()

    assert at.session_state['artifacts_bundled'] == True
    assert at.success[0].value == "Artifacts bundled successfully!"
    assert "Evidence Manifest:" in at.markdown[1].value
    assert "Archive:" in at.markdown[2].value
    assert at.download_button[0].label == "Download Audit Bundle (.zip)"

def test_custom_upload_data_and_model():
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("1. Data & Model Loading").run()

    # Check initial state of custom upload button
    assert at.button[0].disabled == True

    # Simulate file uploads by providing dummy file objects
    # For file_uploader, you need to provide a BytesIO object with content
    dummy_model_content = joblib.dumps(DummyModel())
    dummy_data_content = pd.DataFrame({
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'target': np.random.randint(0, 2, 10)
    }).to_csv(index=False).encode('utf-8')

    model_file = MagicMock()
    model_file.name = "custom_model.pkl"
    model_file.getbuffer.return_value = dummy_model_content

    data_file = MagicMock()
    data_file.name = "custom_data.csv"
    data_file.getbuffer.return_value = dummy_data_content

    at.file_uploader[0].set_value(model_file)
    at.file_uploader[1].set_value(data_file).run()

    # Now the button should be enabled
    assert at.button[0].disabled == False 

    at.button[0].click().run() # Click "Load Custom Model & Data"

    assert at.success[0].value == "Custom artifacts loaded successfully!"
    assert at.session_state['model_loaded'] == True
    assert at.session_state['data_ready'] == True
    assert at.session_state['sample_data_loaded'] == False
    assert at.session_state['model_hash_val'] is not None
    assert at.session_state['data_hash_val'] is not None
    assert "Model SHA-256 Hash" in at.subheader[2].value
    assert "Data SHA-256 Hash" in at.subheader[3].value
    assert at.dataframe[0].value is not None
