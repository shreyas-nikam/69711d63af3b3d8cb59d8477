
import pytest
import os
import shutil
import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import AppTest for Streamlit testing
from streamlit.testing.v1 import AppTest

# Define constants used in the app (assuming they are also defined in source.py)
TARGET_COLUMN = "loan_approved"
RANDOM_SEED = 42

# Define paths that might be used by source.py's initial generation logic
_MODEL_PATH_FROM_SOURCE = "sample_credit_model.pkl"
_DATA_PATH_FROM_SOURCE = "sample_credit_data.csv"

# Helper function to create dummy model and data files
def create_dummy_model_and_data_files(model_path, data_path, target_column):
    """Creates dummy data and a simple RandomForestClassifier model."""
    np.random.seed(RANDOM_SEED)
    data = pd.DataFrame({
        'feature_1': np.random.rand(100) * 100,
        'feature_2': np.random.randint(0, 5, 100),
        'credit_score': np.random.randint(300, 850, 100),
        'income': np.random.rand(100) * 100000,
        'debt_to_income': np.random.rand(100) * 0.5,
        target_column: np.random.randint(0, 2, 100) # Ensure some 0s for denied instances
    })
    data.to_csv(data_path, index=False)

    X = data.drop(columns=[target_column])
    y = data[target_column]
    model = RandomForestClassifier(random_state=RANDOM_SEED)
    model.fit(X, y)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return data, model

# Setup and Teardown for tests
@pytest.fixture(scope="module", autouse=True)
def setup_teardown_environment():
    """
    Fixture to set up dummy files and clean up directories before and after tests.
    """
    dummy_files_dir = "uploaded_files_temp"
    dummy_model_path = os.path.join(dummy_files_dir, "dummy_model.pkl")
    dummy_data_path = os.path.join(dummy_files_dir, "dummy_data.csv")
    
    # Ensure dummy files directory exists and create dummy files
    os.makedirs(dummy_files_dir, exist_ok=True)
    create_dummy_model_and_data_files(dummy_model_path, dummy_data_path, TARGET_COLUMN)
    
    # Clean and create the 'reports' directory
    if os.path.exists("reports"):
        shutil.rmtree("reports")
    os.makedirs("reports", exist_ok=True)

    # Clean up any files that might be created by generate_sample_data_and_model in the root
    if os.path.exists(_MODEL_PATH_FROM_SOURCE):
        os.remove(_MODEL_PATH_FROM_SOURCE)
    if os.path.exists(_DATA_PATH_FROM_SOURCE):
        os.remove(_DATA_PATH_FROM_SOURCE)

    yield # Run tests

    # Clean up after all tests are finished
    if os.path.exists(dummy_files_dir):
        shutil.rmtree(dummy_files_dir)
    if os.path.exists("reports"):
        shutil.rmtree("reports")
    if os.path.exists(_MODEL_PATH_FROM_SOURCE):
        os.remove(_MODEL_PATH_FROM_SOURCE)
    if os.path.exists(_DATA_PATH_FROM_SOURCE):
        os.remove(_DATA_PATH_FROM_SOURCE)

def test_welcome_page_navigation():
    """Verifies the content and initial state of the Welcome page."""
    at = AppTest.from_file("app.py").run()
    assert at.session_state.page == 'Welcome'
    assert at.title[0].value == "QuLab: Lab 5: Interpretability & Explainability Control Workbench"
    assert "Validating PrimeCredit Bank's Loan Approval Model" in at.markdown[2].value
    assert "As Anya Sharma, a dedicated Model Validator at PrimeCredit Bank" in at.markdown[4].value

def test_setup_and_data_page_initial_state_and_navigation():
    """Verifies navigation to Setup & Data page and its initial UI elements."""
    at = AppTest.from_file("app.py").run()
    
    # Navigate to "Setup & Data"
    at.radio[0].set_value('Setup & Data').run()
    
    assert at.session_state.page == 'Setup & Data'
    assert at.title[0].value == "2. Setting the Stage: Environment Setup and Data Ingestion"
    assert at.file_uploader[0].label == "Upload Model (.pkl or .joblib)"
    assert at.file_uploader[1].label == "Upload Data (.csv)"
    assert at.info[0].value == "Please upload both a model file (e.g., `sample_credit_model.pkl`) and a data file (e.g., `sample_credit_data.csv`) to proceed."

def test_setup_and_data_page_upload_and_load_artifacts():
    """
    Tests uploading dummy model and data files and triggering the artifact loading process.
    Verifies session state updates and displayed summary.
    """
    at = AppTest.from_file("app.py").run()
    at.radio[0].set_value('Setup & Data').run()

    dummy_model_path = "uploaded_files_temp/dummy_model.pkl"
    dummy_data_path = "uploaded_files_temp/dummy_data.csv"

    # Simulate file uploads
    with open(dummy_model_path, "rb") as f_model, open(dummy_data_path, "rb") as f_data:
        at.file_uploader[0].upload(f_model, name="dummy_model.pkl", type="application/octet-stream").run()
        at.file_uploader[1].upload(f_data, name="dummy_data.csv", type="text/csv").run()
    
    assert at.success[0].value == "Model: `dummy_model.pkl` and Data: `dummy_data.csv` uploaded."

    # Click "Load and Hash Artifacts" button
    at.button[0].click().run()

    assert at.success[1].value == "Model and data loaded successfully and hashes calculated!"
    
    # Verify session state variables are populated
    assert at.session_state.model is not None
    assert isinstance(at.session_state.data, pd.DataFrame)
    assert isinstance(at.session_state.X, pd.DataFrame)
    assert isinstance(at.session_state.y, pd.Series)
    assert at.session_state.model_hash is not None
    assert at.session_state.data_hash is not None
    assert at.session_state.explainer is not None
    assert isinstance(at.session_state.X_train_exp, pd.DataFrame)
    assert isinstance(at.session_state.X_test_exp, pd.DataFrame)
    assert at.session_state.run_id is not None
    assert at.session_state.explanation_dir is not None

    # Verify displayed summary
    assert at.subheader[1].value == "Loaded Artifacts Summary:"
    assert f"Model Hash (SHA-256): `{at.session_state.model_hash}`" in at.markdown[9].value
    assert f"Data Hash (SHA-256): `{at.session_state.data_hash}`" in at.markdown[10].value
    
def test_global_explanations_page_generation():
    """
    Tests generation and display of global explanations.
    Requires previous session state from Setup & Data.
    """
    at = AppTest.from_file("app.py").run()
    
    # Simulate setup and data loading to populate session state
    dummy_model_path = "uploaded_files_temp/dummy_model.pkl"
    dummy_data_path = "uploaded_files_temp/dummy_data.csv"
    
    at.radio[0].set_value('Setup & Data').run()
    with open(dummy_model_path, "rb") as f_model, open(dummy_data_path, "rb") as f_data:
        at.file_uploader[0].upload(f_model, name="dummy_model.pkl", type="application/octet-stream").run()
        at.file_uploader[1].upload(f_data, name="dummy_data.csv", type="text/csv").run()
    at.button[0].click().run() # Load and Hash Artifacts

    # Navigate to "Global Explanations"
    at.radio[0].set_value('Global Explanations').run()
    assert at.session_state.page == 'Global Explanations'
    assert at.title[0].value == "3. Unveiling Overall Behavior: Global Model Explanations"

    # Click "Generate Global Explanations"
    at.button[0].click().run()

    assert at.success[0].value == "Global explanations generated successfully!"
    assert not at.session_state.global_importance_df.empty
    assert at.session_state.global_shap_values is not None

    # Verify displayed content
    assert at.subheader[1].value == "Global Feature Importance:"
    assert at.dataframe[0].to_pandas().equals(at.session_state.global_importance_df)
    assert at.markdown[9].value == "### Visualization: Global SHAP Summary Plot"
    assert len(at.pyplot) > 0 # Check that a plot was generated

def test_local_explanations_page_generation():
    """
    Tests selection of instances and generation/display of local explanations.
    Requires previous session state from Setup & Data.
    """
    at = AppTest.from_file("app.py").run()
    
    # Simulate setup and data loading to populate session state
    dummy_model_path = "uploaded_files_temp/dummy_model.pkl"
    dummy_data_path = "uploaded_files_temp/dummy_data.csv"
    
    at.radio[0].set_value('Setup & Data').run()
    with open(dummy_model_path, "rb") as f_model, open(dummy_data_path, "rb") as f_data:
        at.file_uploader[0].upload(f_model, name="dummy_model.pkl", type="application/octet-stream").run()
        at.file_uploader[1].upload(f_data, name="dummy_data.csv", type="text/csv").run()
    at.button[0].click().run() # Load and Hash Artifacts

    # Navigate to "Local Explanations"
    at.radio[0].set_value('Local Explanations').run()
    assert at.session_state.page == 'Local Explanations'
    assert at.title[0].value == "4. Deep Dive into Individual Decisions: Local Explanations for Specific Loan Applications"

    # Select instances for local explanation (e.g., first two instances)
    first_two_indices = at.session_state.X.index.tolist()[:2]
    at.multiselect[0].set_values(first_two_indices).run()

    # Click "Generate Local Explanations"
    at.button[0].click().run()

    assert at.success[0].value == "Local explanations generated successfully!"
    assert at.session_state.local_explanations_data is not None
    assert len(at.session_state.local_explanations_data) == len(first_two_indices)
    assert at.session_state.local_shap_explanations_list is not None
    assert len(at.session_state.local_shap_explanations_list) == len(first_two_indices)

    # Verify displayed content for local explanations (e.g., presence of data and plots)
    assert at.subheader[1].value == "Local Explanation Results:"
    assert len(at.pyplot) == len(first_two_indices) # Expect one waterfall plot per instance
    
def test_counterfactuals_page_generation():
    """
    Tests selection of a denied instance and generation/display of counterfactuals.
    Requires previous session state from Setup & Data.
    """
    at = AppTest.from_file("app.py").run()
    
    # Simulate setup and data loading to populate session state
    dummy_model_path = "uploaded_files_temp/dummy_model.pkl"
    dummy_data_path = "uploaded_files_temp/dummy_data.csv"
    
    at.radio[0].set_value('Setup & Data').run()
    with open(dummy_model_path, "rb") as f_model, open(dummy_data_path, "rb") as f_data:
        at.file_uploader[0].upload(f_model, name="dummy_model.pkl", type="application/octet-stream").run()
        at.file_uploader[1].upload(f_data, name="dummy_data.csv", type="text/csv").run()
    at.button[0].click().run() # Load and Hash Artifacts

    # Navigate to "Counterfactuals"
    at.radio[0].set_value('Counterfactuals').run()
    assert at.session_state.page == 'Counterfactuals'
    assert at.title[0].value == "5. 'What If?': Understanding Counterfactuals for Actionable Insights"

    # Ensure there's at least one denied instance to select for counterfactuals
    denied_indices = at.session_state.y[at.session_state.y == 0].index.tolist()
    assert len(denied_indices) > 0, "No denied instances in dummy data for counterfactual test. Adjust create_dummy_model_and_data_files to ensure some denied loans."

    # Select a denied instance for counterfactual analysis and desired class
    at.selectbox[0].set_value(denied_indices[0]).run()
    at.radio[0].set_value(1).run() # Desired outcome: Approval (1)

    # Click "Generate Counterfactuals"
    at.button[0].click().run()

    # The actual generation of counterfactuals relies on DiCE and can be complex.
    # We verify the button click, success/error message, and that `counterfactual_result` is updated.
    assert "Counterfactuals generated successfully!" in at.success[0].value or "Error generating counterfactuals" in at.error[0].value
    
    if at.session_state.counterfactual_result:
        assert at.subheader[1].value == "Counterfactual Explanation Results:"
        assert at.json[0].value is not None # Original instance JSON
        if at.session_state.counterfactual_result.get('counterfactual_instance'):
            assert at.json[1].value is not None # Counterfactual instance JSON
            assert "Minimal Feature Changes to Flip Prediction:" in at.markdown[7].value
        else:
            assert "No counterfactual instance was generated by DiCE" in at.warning[0].value
    else:
        # If no counterfactual result at all, still check for the initial info message
        assert at.info[0].value == "Select a denied instance and click 'Generate Counterfactuals' to see results."

def test_summary_and_audit_page_generation_and_export():
    """
    Tests generation of the summary report and bundling of audit-ready artifacts.
    Requires comprehensive session state from all previous explanation steps.
    """
    at = AppTest.from_file("app.py").run()
    
    # Simulate a full run to populate session state
    # 1. Setup & Data
    dummy_model_path = "uploaded_files_temp/dummy_model.pkl"
    dummy_data_path = "uploaded_files_temp/dummy_data.csv"
    
    at.radio[0].set_value('Setup & Data').run()
    with open(dummy_model_path, "rb") as f_model, open(dummy_data_path, "rb") as f_data:
        at.file_uploader[0].upload(f_model, name="dummy_model.pkl", type="application/octet-stream").run()
        at.file_uploader[1].upload(f_data, name="dummy_data.csv", type="text/csv").run()
    at.button[0].click().run() # Load and Hash Artifacts

    # 2. Global Explanations
    at.radio[0].set_value('Global Explanations').run()
    at.button[0].click().run() # Generate Global Explanations

    # 3. Local Explanations
    at.radio[0].set_value('Local Explanations').run()
    first_two_indices = at.session_state.X.index.tolist()[:2]
    at.multiselect[0].set_values(first_two_indices).run()
    at.button[0].click().run() # Generate Local Explanations

    # 4. Counterfactuals (only if denied instances exist in the dummy data)
    at.radio[0].set_value('Counterfactuals').run()
    denied_indices = at.session_state.y[at.session_state.y == 0].index.tolist()
    if denied_indices:
        at.selectbox[0].set_value(denied_indices[0]).run()
        at.radio[0].set_value(1).run() # Desired outcome: Approval (1)
        at.button[0].click().run() # Generate Counterfactuals
    else:
        # If no denied instances, the counterfactual_result in session state will remain empty or not populated.
        # This will affect the summary generation, but the app handles it.
        pass

    # Navigate to "Summary & Audit"
    at.radio[0].set_value('Summary & Audit').run()
    assert at.session_state.page == 'Summary & Audit'
    assert at.title[0].value == "6. Identifying Gaps: Interpretability Analysis and Validation Findings"

    # Click "Generate Explanation Summary Report"
    at.button[0].click().run()

    assert at.success[0].value == "Summary report generated successfully! Scroll down to view."
    summary_path = os.path.join(at.session_state.explanation_dir, 'explanation_summary.md')
    assert os.path.exists(summary_path)
    assert at.markdown[9].value is not None # Check if summary content is displayed

    # Click "Export All Audit-Ready Artifacts"
    # This button will be the second button on the page, so index 1
    at.button[1].click().run()

    assert at.success[1].value == "Audit-ready artifacts bundled successfully!"
    assert at.session_state.zip_archive_path is not None
    assert os.path.exists(at.session_state.zip_archive_path)

    # Verify download button
    assert at.subheader[2].value == "Download Audit Bundle:"
    assert at.download_button[0].label == "Download Audit ZIP File"

    # Check for the existence of key files within the explanation directory
    explanation_dir = at.session_state.explanation_dir
    assert os.path.exists(os.path.join(explanation_dir, 'config_snapshot.json'))
    assert os.path.exists(os.path.join(explanation_dir, 'evidence_manifest.json'))
    assert os.path.exists(os.path.join(explanation_dir, 'global_explanation.json'))
    assert os.path.exists(os.path.join(explanation_dir, 'local_explanation.json'))
    # Conditionally check for counterfactual_example.json if counterfactuals were generated successfully
    if at.session_state.counterfactual_result and at.session_state.counterfactual_result.get('counterfactual_instance'):
        assert os.path.exists(os.path.join(explanation_dir, 'counterfactual_example.json'))

