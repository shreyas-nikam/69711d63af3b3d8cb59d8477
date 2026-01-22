
import pytest
from streamlit.testing.v1 import AppTest
import pandas as pd
import numpy as np
import os
import io
import pickle
import zipfile
from unittest.mock import patch, MagicMock

# --- Mocks for source.py functions and external libraries ---
# These mocks simulate the behavior of external dependencies and functions from source.py.
# In a real test setup, ensure 'source.py' is importable in your test environment,
# or mock the 'source' module itself if it's not present.

MOCKED_RANDOM_SEED = 42
MOCKED_TARGET_COLUMN = 'loan_approved'

class MockModel:
    """A mock model to simulate predict_proba and predict methods."""
    def predict_proba(self, X):
        # Simulate binary classification probabilities based on 'credit_score'
        if 'credit_score' in X.columns:
            probs = np.array([[0.1, 0.9] if score > 600 else [0.9, 0.1] for score in X['credit_score']])
        else: # Default if credit_score isn't in test X
            probs = np.array([[0.5, 0.5]] * len(X))
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    @property
    def classes_(self):
        return np.array([0, 1])

def mock_load_and_hash_artifacts(model_path, data_path, target_column, random_seed):
    """Mocks the loading and hashing of model and data."""
    mock_model = MockModel()
    
    # Create sample data for testing
    num_samples = 10
    mock_data = pd.DataFrame({
        'feature1': np.random.rand(num_samples),
        'credit_score': np.random.randint(300, 850, num_samples),
        'income': np.random.randint(30000, 150000, num_samples),
        'debt_to_income': np.random.rand(num_samples) * 0.5,
        MOCKED_TARGET_COLUMN: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # Mixed approvals/denials
    })
    X = mock_data.drop(columns=[target_column])
    y = mock_data[target_column]
    model_hash = "mock_model_hash_123"
    data_hash = "mock_data_hash_456"
    return mock_model, mock_data, X, y, model_hash, data_hash

def mock_generate_sample_data_and_model(model_filename, data_filename, target_column, random_seed):
    """Mocks the generation of sample data and model files."""
    # Simulate file creation by touching files or writing dummy content
    with open(model_filename, 'wb') as f:
        pickle.dump(MockModel(), f) # write a dummy pickle
    pd.DataFrame({'a': [1]}).to_csv(data_filename, index=False) # write a dummy csv

def mock_generate_global_shap_explanation(model, X_train_exp, feature_names, explanation_dir):
    """Mocks the generation of global SHAP explanations."""
    mock_importance_df = pd.DataFrame({
        'Feature': ['credit_score', 'income', 'feature1', 'debt_to_income'],
        'Importance': [0.5, 0.3, 0.1, 0.05]
    }).sort_values('Importance', ascending=False)
    
    # Create a mock shap.Explanation object
    mock_shap_values = MagicMock()
    if not X_train_exp.empty:
        mock_shap_values.values = np.random.rand(len(X_train_exp), len(feature_names))
        mock_shap_values.base_values = np.array([0.5] * len(X_train_exp))
        mock_shap_values.data = X_train_exp.values
    else:
        mock_shap_values.values = np.array([])
        mock_shap_values.base_values = np.array([0.5])
        mock_shap_values.data = np.array([])

    return mock_importance_df, [None, mock_shap_values] # TreeExplainer returns [expected_value, shap_values]

def mock_generate_local_shap_explanations(model, X, instances, explainer, explanation_dir):
    """Mocks the generation of local SHAP explanations."""
    local_explanations = {}
    shap_explanations_list = []
    for idx in instances:
        instance_data = X.loc[idx].to_dict()
        pred_prob = model.predict_proba(pd.DataFrame([instance_data]))[0, 1]
        local_explanations[f"instance_{idx}"] = {
            "instance_id": idx,
            "original_features": instance_data,
            "model_prediction": pred_prob,
            "shap_values": {f: np.random.rand() for f in X.columns},
            "explanation_summary": "Mock local explanation for instance " + str(idx)
        }
        
        mock_exp = MagicMock()
        mock_exp.values = np.random.rand(len(X.columns))
        mock_exp.base_values = 0.5
        mock_exp.data = X.loc[idx].values
        mock_exp.feature_names = X.columns.tolist()
        shap_explanations_list.append(mock_exp)
    return local_explanations, shap_explanations_list

def mock_generate_counterfactual_explanation(model, X, feature_names, instance_idx, desired_class, explanation_dir):
    """Mocks the generation of counterfactual explanations."""
    original_instance_df = pd.DataFrame([X.loc[instance_idx]])
    original_instance = original_instance_df.iloc[0].to_dict()
    original_pred_prob = model.predict_proba(original_instance_df)[0, desired_class]
    
    cf_instance = original_instance.copy()
    # Simulate a change that would flip the prediction
    if 'credit_score' in cf_instance:
        cf_instance['credit_score'] = 750 # Assume this is enough to flip
    
    cf_pred_prob = model.predict_proba(pd.DataFrame([cf_instance]))[0, desired_class]
    
    features_changed = {}
    if original_instance.get('credit_score') != cf_instance.get('credit_score'):
        features_changed['credit_score'] = {
            'original': original_instance.get('credit_score'),
            'counterfactual': cf_instance.get('credit_score')
        }

    return {
        'original_instance': original_instance,
        'original_prediction_prob_desired_class': original_pred_prob,
        'counterfactual_instance': cf_instance,
        'counterfactual_prediction_prob_desired_class': cf_pred_prob,
        'features_changed': features_changed
    }

def mock_generate_explanation_summary(global_importance_df, local_explanations_data, counterfactual_result, explanation_dir):
    """Mocks the generation of the validation summary."""
    return "## Mock Validation Summary\nThis is a mock summary report from the test."

def mock_create_config_snapshot(model_hash, data_hash, random_seed, explanation_dir):
    """Mocks the creation of the config snapshot file."""
    # Ensure explanation_dir exists for the mock file creation
    os.makedirs(explanation_dir, exist_ok=True)
    file_path = os.path.join(explanation_dir, 'config_snapshot.json')
    with open(file_path, 'w') as f:
        f.write(f'{{"model_hash": "{model_hash}", "data_hash": "{data_hash}", "random_seed": {random_seed}}}')
    return file_path

def mock_create_evidence_manifest(explanation_dir, output_files_to_bundle):
    """Mocks the creation of the evidence manifest file."""
    # Ensure explanation_dir exists for the mock file creation
    os.makedirs(explanation_dir, exist_ok=True)
    file_path = os.path.join(explanation_dir, 'evidence_manifest.json')
    manifest_content = {"manifest": {os.path.basename(f): "mock_hash" for f in output_files_to_bundle}}
    with open(file_path, 'w') as f:
        import json
        json.dump(manifest_content, f)
    return file_path

def mock_bundle_artifacts_to_zip(explanation_dir, run_id):
    """Mocks the bundling of artifacts into a ZIP file."""
    reports_dir = os.path.join('reports')
    os.makedirs(reports_dir, exist_ok=True)
    zip_path = os.path.join(reports_dir, f'session_05_validation_run_{run_id}.zip')
    
    # Simulate creating an empty zip file for download
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('dummy_artifact.txt', 'This is a dummy artifact for the zip.')
    return zip_path

# Pytest fixture to patch all necessary dependencies before each test
@pytest.fixture(autouse=True)
def mock_all_dependencies():
    # Use context managers for patching to ensure proper cleanup
    with patch('source.RANDOM_SEED', MOCKED_RANDOM_SEED), \
         patch('source.TARGET_COLUMN', MOCKED_TARGET_COLUMN), \
         patch('source.load_and_hash_artifacts', new=mock_load_and_hash_artifacts), \
         patch('source.generate_sample_data_and_model', new=mock_generate_sample_data_and_model), \
         patch('source.generate_global_shap_explanation', new=mock_generate_global_shap_explanation), \
         patch('source.generate_local_shap_explanations', new=mock_generate_local_shap_explanations), \
         patch('source.generate_counterfactual_explanation', new=mock_generate_counterfactual_explanation), \
         patch('source.generate_explanation_summary', new=mock_generate_explanation_summary), \
         patch('source.create_config_snapshot', new=mock_create_config_snapshot), \
         patch('source.create_evidence_manifest', new=mock_create_evidence_manifest), \
         patch('source.bundle_artifacts_to_zip', new=mock_bundle_artifacts_to_zip), \
         patch('os.makedirs', MagicMock()), \
         patch('os.remove', MagicMock()), \
         patch('os.path.exists', return_value=True), \
         patch('shap.TreeExplainer', MagicMock(return_value=MagicMock())), \
         patch('shap.summary_plot', MagicMock()), \
         patch('shap.waterfall_plot', MagicMock()), \
         patch('matplotlib.pyplot.subplots', MagicMock(return_value=(MagicMock(), MagicMock()))), \
         patch('matplotlib.pyplot.close', MagicMock()):
        yield

# Ensure the app.py is in the current directory or specify the path correctly
# For AppTest, the app.py file must be accessible from where the test is run.
APP_FILE = "app.py"

# --- Test Functions ---

def test_home_page():
    at = AppTest.from_file(APP_FILE).run()
    assert at.title[0].value == "QuLab: Lab 5: Interpretability & Explainability Control Workbench"
    assert at.markdown[0].value == "Model Explanation & Explainability Control Workbench"
    assert at.info[0].value == "Navigate to '1. Upload & Configure' to start your validation process."
    assert at.session_state["current_page"] == "Home"
    assert "Model Loaded: ❌" in at.info[1].value
    assert "Data Loaded: ❌" in at.info[2].value

def test_upload_configure_page_load_sample_data():
    at = AppTest.from_file(APP_FILE).run()
    
    # Navigate to "1. Upload & Configure"
    at.sidebar.selectbox[0].set_value("1. Upload & Configure").run()
    assert at.session_state["current_page"] == "1. Upload & Configure"

    # Click 'Load Sample Data' button
    at.button[1].click().run()

    assert at.success[0].value == "Sample model and data loaded successfully!"
    assert at.session_state["model_loaded"] is True
    assert at.session_state["data_loaded"] is True
    assert at.session_state["model_hash"] == "mock_model_hash_123"
    assert at.session_state["data_hash"] == "mock_data_hash_456"
    assert "First 5 rows of feature data:" in at.markdown[-2].value
    assert isinstance(at.session_state["X"], pd.DataFrame)
    assert not at.session_state["X"].empty

def test_upload_configure_page_upload_files():
    at = AppTest.from_file(APP_FILE).run()
    at.sidebar.selectbox[0].set_value("1. Upload & Configure").run()

    # Mock file content for upload
    mock_model_content = b"dummy_model_content_for_pkl"
    mock_data_content = b"feature1,credit_score,income,debt_to_income,loan_approved\n0.1,700,50000,0.2,1\n0.2,500,30000,0.4,0"

    # Simulate file uploads using io.BytesIO
    # file_uploader.set_uploaded_file expects a file-like object or bytes,
    # and also a name and type.
    at.file_uploader[0].set_uploaded_file(name="test_model.pkl", data=io.BytesIO(mock_model_content), type="pkl").run()
    at.file_uploader[1].set_uploaded_file(name="test_data.csv", data=io.BytesIO(mock_data_content), type="csv").run()

    # Click 'Load Uploaded Files' button
    at.button[0].click().run()

    assert at.success[0].value == "Model and data loaded successfully from uploaded files!"
    assert at.session_state["model_loaded"] is True
    assert at.session_state["data_loaded"] is True
    assert at.session_state["model_hash"] == "mock_model_hash_123"
    assert at.session_state["data_hash"] == "mock_data_hash_456"
    assert at.session_state["loaded_model_filename"] == "test_model.pkl"
    assert at.session_state["loaded_data_filename"] == "test_data.csv"


def test_global_explanations_page():
    at = AppTest.from_file(APP_FILE).run()
    
    # Load sample data first to enable global explanations
    at.sidebar.selectbox[0].set_value("1. Upload & Configure").run()
    at.button[1].click().run() # Load Sample Data

    # Navigate to "2. Global Explanations"
    at.sidebar.selectbox[0].set_value("2. Global Explanations").run()
    assert at.session_state["current_page"] == "2. Global Explanations"

    # Click 'Generate Global Explanations' button
    at.button[0].click().run()

    assert at.success[0].value == "Global explanations generated!"
    assert not at.session_state["global_importance_df"].empty
    assert isinstance(at.session_state["global_shap_values"], list)
    assert at.dataframe[0].value.columns.tolist() == ['Feature', 'Importance']
    
    # Verify content of the summary markdown
    assert "From the summary plot and the `global_importance_df`, I can clearly see which features, such as `credit_score` and `income`, are most influential" in at.markdown[-1].value


def test_local_explanations_page():
    at = AppTest.from_file(APP_FILE).run()
    
    # Load sample data first
    at.sidebar.selectbox[0].set_value("1. Upload & Configure").run()
    at.button[1].click().run() # Load Sample Data

    # Navigate to "3. Local Explanations"
    at.sidebar.selectbox[0].set_value("3. Local Explanations").run()
    assert at.session_state["current_page"] == "3. Local Explanations"

    # The default selection logic aims for 2-3 instances.
    initial_selected_indices = at.multiselect[0].value
    assert len(initial_selected_indices) > 0

    # We can refine the selection if needed, but for now, rely on default or set a specific one
    at.multiselect[0].set_value([initial_selected_indices[0]]).run() # Select only the first default

    # Click 'Generate Local Explanations' button
    at.button[0].click().run()

    assert at.success[0].value == "Local explanations generated!"
    assert at.session_state["local_explanations_data"] is not None
    assert len(at.session_state["local_explanations_data"]) == 1 # Because we selected only one
    assert len(at.session_state["shap_explanations_list_for_plots"]) == 1
    
    # Verify expander and JSON content presence
    assert at.expander[0].label.startswith(f"Instance ID: {initial_selected_indices[0]}")
    assert "model_prediction" in at.json[0].value

    # Verify content of the summary markdown
    assert "The local SHAP explanations provide critical insights into individual loan decisions." in at.markdown[-1].value


def test_counterfactuals_page():
    at = AppTest.from_file(APP_FILE).run()
    
    # Load sample data first
    at.sidebar.selectbox[0].set_value("1. Upload & Configure").run()
    at.button[1].click().run() # Load Sample Data

    # Navigate to "4. Counterfactuals"
    at.sidebar.selectbox[0].set_value("4. Counterfactuals").run()
    assert at.session_state["current_page"] == "4. Counterfactuals"

    # Our mock data has indices 0-9, and y has [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    # So denied instances are at indices 1, 3, 5, 7, 9. The selectbox default should pick one of these.
    # Check default selected value
    selected_denied_instance_idx = at.selectbox[0].value
    assert selected_denied_instance_idx in [1, 3, 5, 7, 9]

    # Click 'Generate Counterfactual Example' button
    at.button[0].click().run()

    assert at.success[0].value == "Counterfactual example generated!"
    assert at.session_state["counterfactual_result"] is not None
    assert "original_instance" in at.session_state["counterfactual_result"]
    assert "counterfactual_instance" in at.session_state["counterfactual_result"]
    assert "features_changed" in at.session_state["counterfactual_result"]

    # Verify display elements
    assert at.markdown[2].value == "##### Original Instance:"
    assert "credit_score" in at.json[0].value
    assert at.markdown[3].value == "##### Counterfactual Instance:"
    assert "credit_score" in at.json[1].value
    assert at.markdown[4].value == "##### Features Changed to Flip Prediction:"
    assert "credit_score" in at.json[2].value # Check for key in the dictionary display

    # Verify content of the summary markdown
    assert "The counterfactual analysis provides invaluable 'what-if' scenarios for PrimeCredit Bank." in at.markdown[-1].value


def test_validation_summary_page():
    at = AppTest.from_file(APP_FILE).run()
    
    # Perform actions to populate session state for summary generation
    _setup_app_for_summary_or_export(at)

    # Navigate to "5. Validation Summary"
    at.sidebar.selectbox[0].set_value("5. Validation Summary").run()
    assert at.session_state["current_page"] == "5. Validation Summary"

    # Click 'Generate Validation Summary' button
    at.button[0].click().run()

    assert at.success[0].value == "Validation summary generated!"
    assert at.session_state["explanation_summary_md"] == "## Mock Validation Summary\nThis is a mock summary report from the test."
    assert at.markdown[2].value == at.session_state["explanation_summary_md"]

    # Verify content of the final markdown
    assert "The `explanation_summary.md` document captures Anya's comprehensive analysis." in at.markdown[-1].value


def test_export_artifacts_page():
    at = AppTest.from_file(APP_FILE).run()
    
    # Perform actions to populate session state for artifact bundling
    _setup_app_for_summary_or_export(at)

    # Navigate to "6. Export Artifacts"
    at.sidebar.selectbox[0].set_value("6. Export Artifacts").run()
    assert at.session_state["current_page"] == "6. Export Artifacts"

    # Click 'Generate & Bundle All Audit Artifacts' button
    at.button[0].click().run()

    assert at.success[0].value.startswith("All audit-ready artifacts bundled into:")
    assert at.session_state["zip_archive_path"] is not None
    
    # Verify the download button is present
    assert at.download_button[0].label == "Download Audit-Ready Artifact Bundle"
    assert os.path.basename(at.download_button[0].file_name).startswith("session_05_validation_run_")
    assert at.download_button[0].file_name.endswith(".zip")
    assert at.download_button[0].mime == "application/zip"

    # Verify content of the final markdown
    assert "This single, self-contained archive is PrimeCredit Bank's **audit-ready artifact bundle**." in at.markdown[-1].value

def _setup_app_for_summary_or_export(at: AppTest):
    """Helper function to set up the app state for summary and export tests."""
    # Load sample data
    at.sidebar.selectbox[0].set_value("1. Upload & Configure").run()
    at.button[1].click().run() # Load Sample Data
    
    # Generate Global Explanations
    at.sidebar.selectbox[0].set_value("2. Global Explanations").run()
    at.button[0].click().run()

    # Generate Local Explanations (selecting one instance)
    at.sidebar.selectbox[0].set_value("3. Local Explanations").run()
    initial_selected_indices = at.multiselect[0].value
    if initial_selected_indices:
        at.multiselect[0].set_value([initial_selected_indices[0]]).run()
    at.button[0].click().run()

    # Generate Counterfactual Example (using a default denied instance)
    at.sidebar.selectbox[0].set_value("4. Counterfactuals").run()
    # The app should automatically select a denied instance if available, otherwise disabled button
    if at.button[0].disabled is False: # Only click if enabled
        at.button[0].click().run()
    
    # Ensure all explanations are generated for the summary/export pages
    # The actual content is mocked, so just triggering the actions is enough.
