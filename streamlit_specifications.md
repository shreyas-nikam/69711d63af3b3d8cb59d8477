
# Streamlit Application Specification: Model Validation & Explainability Approval Workbench

## 1. Application Overview

### Purpose

The "Model Validation & Explainability Approval Workbench" serves as a critical tool for Model Validators (like Anya Sharma at PrimeCredit Bank) to rigorously assess the transparency, fairness, and compliance of machine learning models. It operationalizes model interpretability as a formal enterprise control, enabling the generation, analysis, and documentation of audit-ready explanation artifacts. The application ensures that model decisions can be explained to auditors, regulators, and internal stakeholders in a reproducible and defensible way.

### High-Level Story Flow

1.  **Welcome & Context**: The user (Anya Sharma) starts at the home page, understanding her mission to validate the Credit Approval Model (CAM v1.2).
2.  **Model & Data Ingestion**: Anya uploads a trained ML model (e.g., `sample_credit_model.pkl`) and its corresponding feature dataset (e.g., `sample_credit_data.csv`), or loads them via a convenient sample button. The system automatically calculates cryptographic hashes for both artifacts to ensure traceability.
3.  **Global Explanations**: Anya proceeds to generate global feature importance using SHAP, gaining an aggregate view of which factors generally drive the model's decisions. This helps her verify if the model's general logic aligns with PrimeCredit's lending policies.
4.  **Local Explanations**: For a deeper dive, Anya selects specific loan applications (e.g., a denied, an approved, and a borderline case) from the dataset. She then generates local SHAP explanations to understand the contribution of each feature to these individual predictions.
5.  **Counterfactuals**: To provide actionable insights, Anya investigates counterfactual examples for a denied applicant using DiCE. This reveals the minimal changes to an applicant's features that would have resulted in a loan approval, crucial for customer feedback and policy refinement.
6.  **Validation Findings**: Anya synthesizes all generated explanations, identifies any interpretability gaps (e.g., opaque feature interactions), and formulates a clear recommendation for the model's deployment or further refinement.
7.  **Audit-Ready Export**: In the final step, Anya securely bundles all generated explanations, a configuration snapshot, and an evidence manifest (containing SHA-256 hashes of each file) into a timestamped ZIP archive, ensuring full auditability and reproducibility of her validation work.

---

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import datetime
import io
import matplotlib.pyplot as plt
import shap # Required for shap.Explanation object in generate_local_shap_explanations and explainer init
from sklearn.model_selection import train_test_split # Used for X_train_exp / X_test_exp

# Import all functions and necessary global variables from the source.py file
# Note: Global variables like model_hash_val, data_hash_val from source.py
# will reflect the state when source.py is initially parsed.
# For dynamic data from user uploads, st.session_state will be the source of truth,
# especially for audit-ready files like config_snapshot.json.
from source import (
    RANDOM_SEED,
    TARGET_COLUMN,
    calculate_file_hash,
    load_and_hash_artifacts,
    generate_sample_data_and_model,
    generate_global_shap_explanation,
    generate_local_shap_explanations,
    generate_counterfactual_explanation,
    generate_explanation_summary,
    create_config_snapshot,
    create_evidence_manifest,
    bundle_artifacts_to_zip,
    # The following globals from source.py are imported but will represent
    # values from the *initial* execution of source.py when imported.
    # For dynamic data, st.session_state variables will be used instead.
    model_hash_val, # Static hash from source.py's initial load
    data_hash_val,  # Static hash from source.py's initial load
)
```

### `st.session_state` Design

The following `st.session_state` keys will be used to manage the application's state across interactions and page navigation, providing dynamic storage for loaded models, data, and generated explanations:

*   **`current_page`** (str): Stores the name of the currently active page for multi-page simulation.
    *   **Initialized**: `'Home'`
    *   **Updated**: By the user's selection in `st.sidebar.selectbox`.
    *   **Read**: To conditionally render the appropriate page content.
*   **`model_loaded`** (bool): Flag indicating whether an ML model has been successfully loaded.
    *   **Initialized**: `False`
    *   **Updated**: `True` after `load_and_hash_artifacts` (or sample data loading) successfully processes a model.
    *   **Read**: To enable/disable subsequent UI elements that require a loaded model.
*   **`data_loaded`** (bool): Flag indicating whether a feature dataset has been successfully loaded.
    *   **Initialized**: `False`
    *   **Updated**: `True` after `load_and_hash_artifacts` (or sample data loading) successfully processes data.
    *   **Read**: To enable/disable subsequent UI elements that require loaded data.
*   **`model`** (object): The deserialized machine learning model object.
    *   **Initialized**: `None`
    *   **Updated**: With the model object returned by `load_and_hash_artifacts`.
    *   **Read**: Passed as the `model` argument to all explanation generation functions.
*   **`data`** (pd.DataFrame): The full loaded dataset, including features and the target column.
    *   **Initialized**: `pd.DataFrame()`
    *   **Updated**: With the full data DataFrame returned by `load_and_hash_artifacts`.
    *   **Read**: Used for display, and as the `dataframe` argument for `dice_ml.Data` internally during counterfactual generation.
*   **`X`** (pd.DataFrame): Features extracted from the loaded data, excluding the target column.
    *   **Initialized**: `pd.DataFrame()`
    *   **Updated**: With the features DataFrame returned by `load_and_hash_artifacts`.
    *   **Read**: Passed as `X_data` to explanation functions, used for instance selection.
*   **`y`** (pd.Series): The target variable extracted from the loaded data.
    *   **Initialized**: `pd.Series()`
    *   **Updated**: With the target Series returned by `load_and_hash_artifacts`.
    *   **Read**: Used to identify 'denied' or 'approved' instances for local and counterfactual explanation selection.
*   **`model_hash`** (str): The SHA-256 hash of the currently loaded model file.
    *   **Initialized**: `None`
    *   **Updated**: With the hash returned by `load_and_hash_artifacts`.
    *   **Read**: Displayed in the UI and included in `config_snapshot.json` for audit.
*   **`data_hash`** (str): The SHA-256 hash of the currently loaded data file.
    *   **Initialized**: `None`
    *   **Updated**: With the hash returned by `load_and_hash_artifacts`.
    *   **Read**: Displayed in the UI and included in `config_snapshot.json` for audit.
*   **`feature_names`** (list): A list of column names representing the features in `st.session_state.X`.
    *   **Initialized**: `[]`
    *   **Updated**: Populated from `st.session_state.X.columns.tolist()` after data loading.
    *   **Read**: Passed as `feature_names` argument to explanation functions.
*   **`X_train_exp`** (pd.DataFrame): A subset of `X` used as background data for global SHAP explanations (e.g., a training split).
    *   **Initialized**: `pd.DataFrame()`
    *   **Updated**: Derived from `st.session_state.X` using `train_test_split` during initial data loading.
    *   **Read**: Passed as `X_data` to `generate_global_shap_explanation`.
*   **`explainer_global`** (shap.TreeExplainer object): A SHAP explainer instance used for global explanations.
    *   **Initialized**: `None`
    *   **Updated**: Created within `generate_global_shap_explanation` (or just before) using `st.session_state.model`.
    *   **Read**: Used to calculate SHAP values for global and (potentially) local explanations.
*   **`global_importance_df`** (pd.DataFrame): A DataFrame containing global feature importance scores, sorted by importance.
    *   **Initialized**: `pd.DataFrame()`
    *   **Updated**: Returned by `generate_global_shap_explanation`.
    *   **Read**: Displayed in the "Global Explanations" page and passed to `generate_explanation_summary`.
*   **`global_shap_values`** (np.ndarray or list of np.ndarray): Raw SHAP values generated for global explanation (e.g., for class 1 in classification).
    *   **Initialized**: `None`
    *   **Updated**: Returned by `generate_global_shap_explanation`.
    *   **Read**: Used as input for `shap.summary_plot`.
*   **`instances_for_local_explanation`** (list): A list of integer indices (from `st.session_state.X`) selected by the user for local explanation.
    *   **Initialized**: `[]`
    *   **Updated**: By user selection via `st.multiselect` or by pre-calculated defaults upon data load.
    *   **Read**: Passed as `instances_to_explain` to `generate_local_shap_explanations`.
*   **`local_explanations_data`** (dict): A dictionary storing detailed local explanation results for selected instances.
    *   **Initialized**: `{}`
    *   **Updated**: Returned by `generate_local_shap_explanations`.
    *   **Read**: Displayed in the "Local Explanations" page and passed to `generate_explanation_summary`.
*   **`shap_explanations_list_for_plots`** (list of shap.Explanation): A list of `shap.Explanation` objects, each corresponding to a selected instance for local explanation, used for waterfall plots.
    *   **Initialized**: `[]`
    *   **Updated**: Returned by `generate_local_shap_explanations`.
    *   **Read**: Used to generate and display local SHAP waterfall plots.
*   **`denied_instance_for_cf_idx`** (int): The index of a specific denied instance chosen by the user for counterfactual explanation.
    *   **Initialized**: `None`
    *   **Updated**: By user selection via `st.selectbox` or by a pre-calculated default (first denied instance found).
    *   **Read**: Passed as `instance_idx` to `generate_counterfactual_explanation`.
*   **`counterfactual_result`** (dict): A dictionary containing the counterfactual explanation data (original instance, counterfactual instance, feature changes).
    *   **Initialized**: `{}`
    *   **Updated**: Returned by `generate_counterfactual_explanation`.
    *   **Read**: Displayed in the "Counterfactuals" page and passed to `generate_explanation_summary`.
*   **`explanation_summary_md`** (str): A markdown string containing the generated model validation summary report.
    *   **Initialized**: `''`
    *   **Updated**: Returned by `generate_explanation_summary`.
    *   **Read**: Displayed in the "Validation Summary" page.
*   **`run_id`** (str): A unique timestamp-based identifier for the current validation run.
    *   **Initialized**: `datetime.datetime.now().strftime('%Y%m%d_%H%M%S')` on app startup.
    *   **Updated**: Used to create a unique directory for artifacts (`explanation_dir`).
    *   **Read**: Used in the filename of the final bundled ZIP archive.
*   **`explanation_dir`** (str): The full path to the directory where all explanation artifacts for the current run are saved.
    *   **Initialized**: `f'reports/session_05_validation_run_{st.session_state.run_id}'` on app startup.
    *   **Updated**: `os.makedirs` ensures this directory exists.
    *   **Read**: Passed as `explanation_dir` to all explanation generation and bundling functions.
*   **`output_files_to_bundle`** (list): A list of absolute file paths to all generated explanation artifacts that need to be included in the final ZIP bundle.
    *   **Initialized**: `[]`
    *   **Updated**: Populated as explanation files (`.json`, `.md`) are created, and by `create_config_snapshot`, `create_evidence_manifest`.
    *   **Read**: Passed to `create_evidence_manifest` and `bundle_artifacts_to_zip`.
*   **`zip_archive_path`** (str): The full path to the generated audit-ready ZIP archive.
    *   **Initialized**: `None`
    *   **Updated**: Returned by `bundle_artifacts_to_zip`.
    *   **Read**: Used by the `st.download_button`.
*   **`loaded_model_filename`** (str): The filename of the model that was last loaded (either uploaded or sample).
    *   **Initialized**: `None`
    *   **Updated**: Set after successful model loading.
    *   **Read**: Displayed in "Upload & Configure" and in the sidebar summary.
*   **`loaded_data_filename`** (str): The filename of the data that was last loaded (either uploaded or sample).
    *   **Initialized**: `None`
    *   **Updated**: Set after successful data loading.
    *   **Read**: Displayed in "Upload & Configure" and in the sidebar summary.
*   **`temp_model_path`** (str): A temporary file path used to save an uploaded model before processing.
    *   **Initialized**: `None`
    *   **Updated**: Set when a user uploads a model file.
    *   **Read**: Passed to `load_and_hash_artifacts`. Cleaned up after use.
*   **`temp_data_path`** (str): A temporary file path used to save an uploaded dataset before processing.
    *   **Initialized**: `None`
    *   **Updated**: Set when a user uploads a data file.
    *   **Read**: Passed to `load_and_hash_artifacts`. Cleaned up after use.
*   **`shap_explainer_for_local`** (shap.TreeExplainer): A dedicated SHAP explainer instance for local and counterfactual explanations, created once the model is loaded.
    *   **Initialized**: `None`
    *   **Updated**: Initialized with `shap.TreeExplainer(st.session_state.model)` when `model_loaded` becomes `True`.
    *   **Read**: Passed to `generate_local_shap_explanations` and implicitly used by `dice_ml` for counterfactuals.

### UI Interactions and Function Calls

#### **Global Configuration and Session State Initialization (executed on script run)**

```python
# Initialize session state variables with defaults
if 'RANDOM_SEED' not in st.session_state:
    st.session_state['RANDOM_SEED'] = RANDOM_SEED # From source.py global
if 'TARGET_COLUMN' not in st.session_state:
    st.session_state['TARGET_COLUMN'] = TARGET_COLUMN # From source.py global

# Initialize other session state variables
for key, default_value in {
    'current_page': 'Home',
    'model_loaded': False,
    'data_loaded': False,
    'model': None,
    'data': pd.DataFrame(),
    'X': pd.DataFrame(),
    'y': pd.Series(),
    'model_hash': None,
    'data_hash': None,
    'feature_names': [],
    'X_train_exp': pd.DataFrame(),
    'global_importance_df': pd.DataFrame(),
    'global_shap_values': None,
    'instances_for_local_explanation': [],
    'local_explanations_data': {},
    'shap_explanations_list_for_plots': [],
    'denied_instance_for_cf_idx': None,
    'counterfactual_result': {},
    'explanation_summary_md': '',
    'run_id': None,
    'explanation_dir': None,
    'output_files_to_bundle': [],
    'zip_archive_path': None,
    'loaded_model_filename': None,
    'loaded_data_filename': None,
    'temp_model_path': None,
    'temp_data_path': None,
    'shap_explainer_for_local': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Generate a unique run_id and explanation_dir if not yet set for the session
if st.session_state.run_id is None:
    st.session_state.run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    st.session_state.explanation_dir = os.path.join('reports', f'session_05_validation_run_{st.session_state.run_id}')
    os.makedirs(st.session_state.explanation_dir, exist_ok=True)
```

#### **Sidebar Navigation**

The Streamlit sidebar will host the navigation controls and display a summary of the current loading status.

```python
with st.sidebar:
    st.title("PrimeCredit Bank 🏦")
    st.markdown("### Model Validation Workbench")
    # Navigation selectbox updates st.session_state.current_page
    st.session_state.current_page = st.selectbox(
        "Navigate",
        ["Home", "1. Upload & Configure", "2. Global Explanations", "3. Local Explanations",
         "4. Counterfactuals", "5. Validation Summary", "6. Export Artifacts"]
    )
    st.markdown("---")
    # Display current status using session state
    st.info(f"Model Loaded: {'✅' if st.session_state.model_loaded else '❌'}")
    st.info(f"Data Loaded: {'✅' if st.session_state.data_loaded else '❌'}")
    if st.session_state.model_loaded:
        st.caption(f"Model: {st.session_state.loaded_model_filename} ({st.session_state.model_hash[:8]}...)")
    if st.session_state.data_loaded:
        st.caption(f"Data: {st.session_state.loaded_data_filename} ({st.session_state.data_hash[:8]}...)")
```

#### **Page 1: Home**

This page provides an introduction and sets the context for the user persona (Anya Sharma).

```python
if st.session_state.current_page == "Home":
    st.title("Model Explanation & Explainability Control Workbench")
    st.markdown(f"## Validating PrimeCredit Bank's Loan Approval Model")
    st.markdown(f"") # Placeholder for consistent spacing/breaks
    st.markdown(f"As **Anya Sharma**, a dedicated Model Validator at PrimeCredit Bank, my primary responsibility is to ensure that all machine learning models used in critical business processes are transparent, fair, and compliant with internal governance and external regulatory standards.")
    st.markdown(f"Today, my focus is on a newly developed **Credit Approval Model (CAM v1.2)**, which will determine loan eligibility for our customers. Before this model can be deployed, I must rigorously assess its interpretability and explainability.")
    st.markdown(f"My goal is to thoroughly vet this model, identifying any interpretability gaps that could lead to biased decisions, regulatory scrutiny, or a lack of trust from stakeholders. I need to demonstrate that the model's decisions are defensible and understandable, not just to me, but also to internal auditors and future regulators. This application serves as my workbench to generate, analyze, and document the required explanations as audit-ready artifacts.")
    st.markdown(f"")
    st.info("Navigate to '1. Upload & Configure' to start your validation process.")
```

#### **Page 2: 1. Upload & Configure**

This page allows the user to upload their model and data files or load sample files. It triggers the loading and hashing process from `source.py`.

```python
elif st.session_state.current_page == "1. Upload & Configure":
    st.title("2. Setting the Stage: Environment Setup and Data Ingestion")
    st.markdown(f"My first step is to prepare my environment and load the necessary model and data for validation. Reproducibility is paramount in model validation; therefore, I will fix a random seed and compute SHA-256 hashes for both the model and dataset to ensure traceability and detect any unauthorized changes.")
    st.markdown(f"")

    st.subheader("Load Model and Data")
    uploaded_model_file = st.file_uploader("Upload Trained ML Model (.pkl or .joblib)", type=["pkl", "joblib"])
    uploaded_data_file = st.file_uploader("Upload Feature Dataset (.csv)", type=["csv"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Uploaded Files", type="primary", disabled=(uploaded_model_file is None or uploaded_data_file is None)):
            # Save uploaded files temporarily
            st.session_state.temp_model_path = "uploaded_model.pkl"
            st.session_state.temp_data_path = "uploaded_data.csv"
            with open(st.session_state.temp_model_path, "wb") as f:
                f.write(uploaded_model_file.getbuffer())
            with open(st.session_state.temp_data_path, "wb") as f:
                f.write(uploaded_data_file.getbuffer())
            
            st.session_state.loaded_model_filename = uploaded_model_file.name
            st.session_state.loaded_data_filename = uploaded_data_file.name
            
            # Call source.py function to load and hash
            try:
                model, full_data, X, y, model_hash_val, data_hash_val = load_and_hash_artifacts(
                    st.session_state.temp_model_path, st.session_state.temp_data_path, st.session_state.TARGET_COLUMN, st.session_state.RANDOM_SEED
                )
                # Update session state with loaded artifacts
                st.session_state.model = model
                st.session_state.data = full_data
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.model_hash = model_hash_val
                st.session_state.data_hash = data_hash_val
                st.session_state.feature_names = X.columns.tolist()
                st.session_state.model_loaded = True
                st.session_state.data_loaded = True

                # Split data for explanation purposes (as per source.py logic for X_train_exp)
                X_train_exp, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=st.session_state.RANDOM_SEED)
                st.session_state.X_train_exp = X_train_exp

                # Initialize SHAP explainer for local explanations
                st.session_state.shap_explainer_for_local = shap.TreeExplainer(model)

                st.success("Model and data loaded successfully from uploaded files!")
            except Exception as e:
                st.error(f"Error loading uploaded files: {e}")
                st.session_state.model_loaded = False
                st.session_state.data_loaded = False
            finally:
                # Clean up temporary files
                if os.path.exists(st.session_state.temp_model_path): os.remove(st.session_state.temp_model_path)
                if os.path.exists(st.session_state.temp_data_path): os.remove(st.session_state.temp_data_path)

    with col2:
        if st.button("Load Sample Data", disabled=(st.session_state.model_loaded and st.session_state.data_loaded)):
            # Ensure sample files exist or generate them if missing
            # The generate_sample_data_and_model function is called, which creates 'sample_credit_model.pkl' and 'sample_credit_data.csv'
            generate_sample_data_and_model('sample_credit_model.pkl', 'sample_credit_data.csv', st.session_state.TARGET_COLUMN, st.session_state.RANDOM_SEED)

            st.session_state.temp_model_path = 'sample_credit_model.pkl'
            st.session_state.temp_data_path = 'sample_credit_data.csv'
            st.session_state.loaded_model_filename = 'sample_credit_model.pkl'
            st.session_state.loaded_data_filename = 'sample_credit_data.csv'

            # Call source.py function to load and hash the (now guaranteed to exist) sample files
            try:
                model, full_data, X, y, model_hash_val, data_hash_val = load_and_hash_artifacts(
                    st.session_state.temp_model_path, st.session_state.temp_data_path, st.session_state.TARGET_COLUMN, st.session_state.RANDOM_SEED
                )
                # Update session state with loaded artifacts
                st.session_state.model = model
                st.session_state.data = full_data
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.model_hash = model_hash_val
                st.session_state.data_hash = data_hash_val
                st.session_state.feature_names = X.columns.tolist()
                st.session_state.model_loaded = True
                st.session_state.data_loaded = True

                # Split data for explanation purposes
                X_train_exp, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=st.session_state.RANDOM_SEED)
                st.session_state.X_train_exp = X_train_exp

                # Initialize SHAP explainer
                st.session_state.shap_explainer_for_local = shap.TreeExplainer(model)
                
                st.success("Sample model and data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading sample files: {e}")
                st.session_state.model_loaded = False
                st.session_state.data_loaded = False

    if st.session_state.model_loaded and st.session_state.data_loaded:
        st.subheader("Loaded Artifact Details:")
        st.markdown(f"- **Model File:** `{st.session_state.loaded_model_filename}`")
        st.markdown(f"- **Model Hash (SHA-256):** `{st.session_state.model_hash}`")
        st.markdown(f"- **Data File:** `{st.session_state.loaded_data_filename}`")
        st.markdown(f"- **Data Hash (SHA-256):** `{st.session_state.data_hash}`")
        st.markdown(f"- **Random Seed Used:** `{st.session_state.RANDOM_SEED}`")
        st.markdown(f"- **Model type identified:** `{type(st.session_state.model)}`")
        st.markdown(f"- **Data features:** `{', '.join(st.session_state.feature_names)}`")

        st.markdown(f"First 5 rows of feature data:")
        st.dataframe(st.session_state.X.head())

        st.markdown(f"")
        st.markdown(f"The initial setup is complete. I've successfully loaded the Credit Approval Model and the associated feature dataset. Crucially, I've generated cryptographic hashes for both artifacts: `{st.session_state.model_hash}` for `{st.session_state.loaded_model_filename}` and `{st.session_state.data_hash}` for `{st.session_state.loaded_data_filename}`. These hashes are vital for maintaining an immutable audit trail; any future change to either the model or the dataset would result in a different hash, immediately signaling a potential issue to an auditor. This step aligns with PrimeCredit Bank's stringent requirements for data and model integrity. The data has also been pre-processed, separating features from the target variable, making it ready for explanation generation.")
```

#### **Page 3: 2. Global Explanations**

This page focuses on generating and displaying global feature importance using SHAP.

```python
elif st.session_state.current_page == "2. Global Explanations":
    st.title("3. Unveiling Overall Behavior: Global Model Explanations")
    st.markdown(f"As a Model Validator, I first need to grasp the overall behavior of the CAM v1.2. Which factors generally drive its decisions for approving or denying loans? Global explanations provide an aggregate view of feature importance, revealing which features have the most impact across all predictions. This helps me verify if the model's general logic aligns with PrimeCredit's lending policies and expert domain knowledge. For tree-based models like our `RandomForestClassifier`, SHAP (SHapley Additive exPlanations) values are an excellent choice for this. The SHAP value $\phi_i$ for a feature $i$ represents the average marginal contribution of that feature value to the prediction across all possible coalitions of features.")
    st.markdown(r"The fundamental idea behind SHAP values is to attribute the prediction of an instance $x$ to its features by considering the contribution of each feature to moving the prediction from the base value (average prediction) to the current prediction. The sum of the SHAP values for all features and the base value equals the model's output for that instance:")
    st.markdown(r"$$ \phi_0 + \sum_{{i=1}}^{{M}} \phi_i(f, x) = f(x) $$")
    st.markdown(r"where $\phi_0$ is the expected model output (the base value), $M$ is the number of features, $\phi_i(f, x)$ is the SHAP value for feature $i$ for instance $x$, and $f(x)$ is the model's prediction for instance $x$.")
    st.markdown(f"")

    if st.button("Generate Global Explanations", disabled=not st.session_state.model_loaded):
        with st.spinner("Generating global SHAP explanations... This may take a moment."):
            try:
                # Call source.py function to generate global SHAP explanations
                global_importance_df, global_shap_values_raw = generate_global_shap_explanation(
                    st.session_state.model, 
                    st.session_state.X_train_exp, 
                    st.session_state.feature_names, 
                    st.session_state.explanation_dir
                )
                # Update session state with results
                st.session_state.global_importance_df = global_importance_df
                st.session_state.global_shap_values = global_shap_values_raw 

                st.success("Global explanations generated!")
            except Exception as e:
                st.error(f"Error generating global explanations: {e}")

    if not st.session_state.global_importance_df.empty:
        st.subheader("Global Feature Importance Ranking")
        st.dataframe(st.session_state.global_importance_df)

        st.subheader("SHAP Global Summary Plot")
        # Ensure shap_values_for_plot is correctly extracted for display
        if isinstance(st.session_state.global_shap_values, list):
            shap_values_for_plot = st.session_state.global_shap_values[1] # Assuming class 1 is the positive outcome
        else:
            shap_values_for_plot = st.session_state.global_shap_values

        if shap_values_for_plot is not None and not st.session_state.X_train_exp.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Use a sample of X_train_exp for plotting if it's very large, as per notebook logic
            sample_data_for_plot = st.session_state.X_train_exp.sample(
                min(1000, st.session_state.X_train_exp.shape[0]), 
                random_state=st.session_state.RANDOM_SEED
            ) if st.session_state.X_train_exp.shape[0] > 1000 else st.session_state.X_train_exp
            
            shap.summary_plot(
                shap_values_for_plot, 
                sample_data_for_plot, 
                plot_type="bar", 
                show=False
            )
            st.pyplot(fig) # Display the matplotlib figure
            plt.close(fig) # Close the figure to free up memory
        else:
            st.warning("Global SHAP values or data for plotting not available.")

        st.markdown(f"")
        st.markdown(f"The global SHAP explanation reveals the overall drivers of the Credit Approval Model. From the summary plot and the `global_importance_df`, I can clearly see which features, such as `credit_score` and `income`, are most influential in the model's decisions regarding loan approval. This high-level overview confirms that the model is largely relying on expected financial health indicators, which aligns with PrimeCredit Bank's established lending criteria. This gives me initial confidence that the model's general behavior is sensible and explainable to senior stakeholders. However, global explanations only tell part of the story; I need to investigate specific individual decisions to ensure consistency and fairness.")
```

#### **Page 4: 3. Local Explanations**

This page allows the user to select specific instances and generate detailed local SHAP explanations for them.

```python
elif st.session_state.current_page == "3. Local Explanations":
    st.title("4. Deep Dive into Individual Decisions: Local Explanations for Specific Loan Applications")
    st.markdown(f"While global explanations are useful, they don't explain why a *specific* loan applicant was approved or denied. As a Model Validator, I frequently encounter requests to understand individual decisions, especially for denied applications or those with unusual profiles. For PrimeCredit Bank, it's crucial to provide clear, defensible reasons to customers for loan denials or approvals. I will select a few representative cases from our `sample_credit_data` to generate local explanations using SHAP. This allows me to examine the contribution of each feature to that particular prediction.")
    st.markdown(r"For a specific instance $x$, the SHAP values $\phi_i(f, x)$ quantify how much each feature $i$ contributes to the prediction $f(x)$ compared to the average prediction $\phi_0$. A positive SHAP value for a feature means it pushed the prediction higher (towards approval), while a negative value pushed it lower (towards denial).")
    st.markdown(f"")

    if st.session_state.data_loaded:
        st.subheader("Select Instances for Local Explanation")
        st.dataframe(st.session_state.X.head()) # Show head of data for context to help user select

        # Pre-calculate suggested instances for user selection, mirroring source.py logic
        selected_indices_for_local_default = []
        if not st.session_state.X.empty and not st.session_state.y.empty:
            denied_indices = st.session_state.y[st.session_state.y == 0].index
            approved_indices = st.session_state.y[st.session_state.y == 1].index

            suggested_unique_indices = set()
            
            if not denied_indices.empty:
                suggested_unique_indices.add(denied_indices[0])
            elif not st.session_state.X.empty:
                suggested_unique_indices.add(st.session_state.X.index[0])

            if not approved_indices.empty:
                for idx in approved_indices:
                    if idx not in suggested_unique_indices:
                        suggested_unique_indices.add(idx)
                        break
            if len(suggested_unique_indices) < 2 and not st.session_state.X.empty:
                for idx in st.session_state.X.index:
                    if idx not in suggested_unique_indices:
                        suggested_unique_indices.add(idx)
                        break

            probabilities_for_borderline = np.zeros(len(st.session_state.X))
            if len(st.session_state.X) > 0 and st.session_state.TARGET_COLUMN in st.session_state.model.classes_:
                positive_class_idx_for_proba_main_block = np.where(st.session_state.model.classes_ == 1)[0][0]
                model_predict_proba_output = st.session_state.model.predict_proba(st.session_state.X)
                if model_predict_proba_output.shape[1] > positive_class_idx_for_proba_main_block:
                    probabilities_for_borderline = model_predict_proba_output[:, positive_class_idx_for_proba_main_block]
            
            borderline_idx = None
            if len(probabilities_for_borderline) > 0:
                borderline_idx_pos_in_X = np.argmin(np.abs(probabilities_for_borderline - 0.5))
                borderline_idx = st.session_state.X.index[borderline_idx_pos_in_X]

            if borderline_idx is not None:
                suggested_unique_indices.add(borderline_idx)
            
            selected_indices_for_local_default = sorted(list(suggested_unique_indices))[:3]
            if not selected_indices_for_local_default and not st.session_state.X.empty:
                selected_indices_for_local_default = [st.session_state.X.index[0]]

        selected_indices_for_local = st.multiselect(
            "Select Instance IDs from the full dataset (e.g., 0, 1, 2, ...):",
            options=st.session_state.X.index.tolist(),
            default=selected_indices_for_local_default,
            help="Select up to 3 instance IDs for detailed local explanations."
        )
        st.session_state.instances_for_local_explanation = selected_indices_for_local

        if st.button("Generate Local Explanations", disabled=not st.session_state.model_loaded or not st.session_state.instances_for_local_explanation):
            with st.spinner("Generating local SHAP explanations..."):
                try:
                    # Use the stored explainer from session state
                    if st.session_state.shap_explainer_for_local is None:
                        st.session_state.shap_explainer_for_local = shap.TreeExplainer(st.session_state.model)

                    local_explanations_data, shap_explanations_list = generate_local_shap_explanations(
                        st.session_state.model, 
                        st.session_state.X, 
                        st.session_state.instances_for_local_explanation, 
                        st.session_state.shap_explainer_for_local, 
                        st.session_state.explanation_dir
                    )
                    # Update session state with results
                    st.session_state.local_explanations_data = local_explanations_data
                    st.session_state.shap_explanations_list_for_plots = shap_explanations_list
                    st.success("Local explanations generated!")
                except Exception as e:
                    st.error(f"Error generating local explanations: {e}")

    if st.session_state.local_explanations_data:
        st.subheader("Local Explanation Details:")
        for i, (inst_key, explanation_data) in enumerate(st.session_state.local_explanations_data.items()):
            instance_id = int(inst_key.split('_')[1])
            with st.expander(f"Instance ID: {instance_id} (Predicted Approval Probability: {explanation_data['model_prediction']:.4f})"):
                st.json(explanation_data)
                
                # Display waterfall plot
                if st.session_state.shap_explanations_list_for_plots:
                    try:
                        # Find the corresponding SHAP explanation object based on data identity
                        current_shap_exp = next((exp for exp in st.session_state.shap_explanations_list_for_plots if np.array_equal(exp.data, st.session_state.X.loc[instance_id].values)), None)
                        if current_shap_exp:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            shap.waterfall_plot(current_shap_exp, max_display=10, show=False)
                            st.pyplot(fig)
                            plt.close(fig) # Close the figure to free up memory
                        else:
                            st.warning(f"Could not find SHAP waterfall plot for instance ID {instance_id}. (There might be a mismatch in data used for explanation generation and plotting list).")
                    except Exception as e:
                        st.error(f"Error displaying waterfall plot for instance ID {instance_id}: {e}")

        st.markdown(f"")
        st.markdown(f"The local SHAP explanations provide critical insights into individual loan decisions. For instance, analyzing the waterfall plot for a *denied* application, I can clearly see that a low `credit_score` and high `debt_to_income` ratio were the primary negative contributors, pushing the loan approval probability below the threshold. Conversely, for an *approved* application, a high `credit_score` and `income` might be the dominant positive factors. For a *borderline* case, the contributions might be more balanced.")
        st.markdown(f"")
        st.markdown(f"These detailed breakdowns are invaluable for Anya. They allow her to:")
        st.markdown(f"1.  **Verify decision logic:** Are the model's specific reasons for a decision coherent and justifiable according to PrimeCredit's policy?")
        st.markdown(f"2.  **Identify potential biases:** Do certain demographic features (if present) disproportionately influence decisions in specific cases without valid business rationale? (Note: no demographic features are in this sample data, but this is what a Model Validator would look for).")
        st.markdown(f"3.  **Provide actionable feedback:** Understand what factors led to a denial, which is crucial for communicating with applicants.")
        st.markdown(f"")
        st.markdown(f"This level of detail is exactly what PrimeCredit's internal auditors and potentially regulators would require to validate the fairness and transparency of the model.")
```

#### **Page 5: 4. Counterfactuals**

This page enables the user to generate counterfactual explanations for a selected denied instance.

```python
elif st.session_state.current_page == "4. Counterfactuals":
    st.title('5. "What If?": Understanding Counterfactuals for Actionable Insights')
    st.markdown(f"For a denied loan applicant, merely knowing *why* they were denied (via local explanations) isn't always enough. As Anya, I also need to understand 'what if?' – what minimal changes to their application would have resulted in an approval? This is where counterfactual explanations come in. They identify the smallest, most actionable changes to an applicant's features that would flip the model's decision from denial to approval. This information is invaluable for PrimeCredit Bank, not only for providing constructive feedback to customers but also for potentially refining our lending criteria or identifying areas where applicants can improve their financial standing to become eligible.")
    st.markdown(r"The objective of generating a counterfactual example $x'$ for an original instance $x$ that results in a different prediction $y'$ is to minimize the distance between $x$ and $x'$, subject to the constraint that $x'$ belongs to the feasible input space $\mathcal{{X}}$ and the model $f$ predicts $y'$ for $x'$. This can be formalized as:")
    st.markdown(r"$$ \min_{{x'}} \text{{distance}}(x, x') \quad \text{{s.t.}} \quad f(x') = y' \quad \text{{and}} \quad x' \in \mathcal{{X}} $$")
    st.markdown(r"where $\text{{distance}}(x, x')$ is a measure of proximity (e.g., L1 or L2 norm), and $f(x')$ is the model's prediction for the counterfactual instance $x'$.")
    st.markdown(f"")

    if st.session_state.data_loaded and not st.session_state.X.empty and not st.session_state.y.empty:
        st.subheader("Select a Denied Instance for Counterfactual Generation")

        # Pre-select a denied instance for CF, similar to source.py logic
        denied_indices = st.session_state.y[st.session_state.y == 0].index
        denied_instance_options = denied_indices.tolist() if not denied_indices.empty else []

        if not denied_instance_options:
            st.warning("No 'denied' instances found in the dataset to generate counterfactuals. Cannot proceed.")
            st.session_state.denied_instance_for_cf_idx = None
        else:
            default_denied_idx = denied_instance_options[0] if denied_instance_options else None
            # Set default index carefully to avoid error if default_denied_idx is not in options (e.g., if options is empty)
            default_index_to_select = denied_instance_options.index(default_denied_idx) if default_denied_idx in denied_instance_options else 0

            st.session_state.denied_instance_for_cf_idx = st.selectbox(
                "Select a denied instance ID:",
                options=denied_instance_options,
                index=default_index_to_select,
                help="Choose an instance where the loan was denied to see what minimal changes would approve it."
            )

        if st.button("Generate Counterfactual Example", disabled=not st.session_state.model_loaded or st.session_state.denied_instance_for_cf_idx is None):
            with st.spinner("Generating counterfactual explanation... This might take a moment."):
                try:
                    # Call source.py function to generate counterfactuals
                    counterfactual_data = generate_counterfactual_explanation(
                        st.session_state.model, 
                        st.session_state.X, 
                        st.session_state.feature_names, 
                        st.session_state.denied_instance_for_cf_idx, 
                        1, # Desired class for approval is 1
                        st.session_state.explanation_dir
                    )
                    # Update session state with results
                    st.session_state.counterfactual_result = counterfactual_data
                    if counterfactual_data:
                        st.success("Counterfactual example generated!")
                    else:
                        st.info("No counterfactuals found for the selected instance with desired class 1.")
                except Exception as e:
                    st.error(f"Error generating counterfactuals: {e}")
    else:
        st.warning("Please load a model and data first on the 'Upload & Configure' page.")


    if st.session_state.counterfactual_result:
        st.subheader("Counterfactual Analysis:")
        cf_res = st.session_state.counterfactual_result
        if cf_res:
            st.markdown("##### Original Instance:")
            st.json(cf_res.get('original_instance', {}))
            st.write(f"Original Prediction Probability (desired class 1): `{cf_res.get('original_prediction_prob_desired_class', 0.0):.4f}`")

            st.markdown("##### Counterfactual Instance:")
            st.json(cf_res.get('counterfactual_instance', {}))
            st.write(f"Counterfactual Prediction Probability (desired class 1): `{cf_res.get('counterfactual_prediction_prob_desired_class', 0.0):.4f}`")

            st.markdown("##### Features Changed to Flip Prediction:")
            if cf_res.get('features_changed'):
                st.json(cf_res['features_changed'])
            else:
                st.info("No features needed to be changed (or none found) to flip the prediction to the desired class for this instance.")
        else:
            st.info("No counterfactual data available to display.")
        
        st.markdown(f"")
        st.markdown(f"The counterfactual analysis provides invaluable 'what-if' scenarios for PrimeCredit Bank. For the selected denied loan application, the `counterfactual_result` clearly shows that increasing the `credit_score` by a certain amount or raising the `income` significantly, for example, would have resulted in the loan being approved. The `features_changed` dictionary pinpoints the minimal adjustments needed.")
        st.markdown(f"")
        st.markdown(f"This empowers Anya to:")
        st.markdown(f"1.  **Inform customers:** Instead of just saying 'your loan was denied,' PrimeCredit can advise applicants on specific, actionable steps (e.g., 'If your credit score improved by X points, you would likely be approved').")
        st.markdown(f"2.  **Refine policy:** If generating counterfactuals consistently highlights specific features as critical for flipping decisions, it might indicate areas for policy review or for developing financial literacy programs for customers.")
        st.markdown(f"3.  **Assess model sensitivity:** It reveals how sensitive the model is to changes in specific features, which is a key part of model validation.")
        st.markdown(f"")
        st.markdown(f"This concrete evidence of actionable insights is crucial for establishing trust and demonstrating the model's utility beyond just making a prediction.")

```

#### **Page 6: 5. Validation Summary**

This page allows Anya to generate a comprehensive summary report based on all generated explanations.

```python
elif st.session_state.current_page == "5. Validation Summary":
    st.title("6. Identifying Gaps: Interpretability Analysis and Validation Findings")
    st.markdown(f"After reviewing the global, local, and counterfactual explanations, Anya must now synthesize her findings and identify any interpretability gaps that could prevent the CAM v1.2 from being approved for deployment. This is a critical step for PrimeCredit Bank's risk management framework. An interpretability gap might be a feature that, while statistically significant, lacks a clear business rationale, or cases where local explanations seem inconsistent. I need to document my observations, evaluate the model's transparency, and make a recommendation for its deployment or further refinement.")
    st.markdown(f"")
    st.markdown(f"My analysis will focus on:")
    st.markdown(f"-   **Coherence with Policy:** Do the explanations align with PrimeCredit's established lending policies and regulations?")
    st.markdown(f"-   **Transparency:** Are the reasons for decisions clear, concise, and easily understandable by non-technical stakeholders (e.g., loan officers, customers, auditors)?")
    st.markdown(f"-   **Consistency:** Do similar cases receive similar explanations, and are there any anomalous explanations?")
    st.markdown(f"-   **Actionability:** Do counterfactuals provide practical advice for applicants?")
    st.markdown(f"")
    st.markdown(f"Based on these, I will formulate a summary of my findings and a recommendation.")
    st.markdown(f"")

    # The button to generate summary is disabled if no explanations have been generated
    if st.button("Generate Validation Summary", disabled=(st.session_state.global_importance_df.empty and not st.session_state.local_explanations_data and not st.session_state.counterfactual_result)):
        with st.spinner("Generating explanation summary..."):
            try:
                # Call source.py function to generate the summary.
                # NOTE: The `generate_explanation_summary` function in `source.py`
                # directly accesses global variables `model_hash_val` and `data_hash_val`
                # which are set when `source.py` is initially loaded.
                # Thus, the hashes embedded in the generated `explanation_summary.md` will
                # reflect the hashes of the *sample data* (from `source.py`'s initial execution),
                # not necessarily the dynamic hashes of user-uploaded files.
                # The `config_snapshot.json` (generated on the next page) will contain the correct, dynamic hashes.
                summary_content = generate_explanation_summary(
                    st.session_state.global_importance_df, 
                    st.session_state.local_explanations_data, 
                    st.session_state.counterfactual_result, 
                    st.session_state.explanation_dir
                )
                st.session_state.explanation_summary_md = summary_content
                st.success("Validation summary generated!")
            except Exception as e:
                st.error(f"Error generating validation summary: {e}")

    if st.session_state.explanation_summary_md:
        st.subheader("Validation Summary Report:")
        st.markdown(st.session_state.explanation_summary_md)
        st.markdown(f"") # Consistent spacing
        st.markdown(f"**Note on Hashes in Summary:** Due to strict constraints on modifying `source.py` functions, the model and data hashes displayed directly within this markdown summary (`explanation_summary.md`) are derived from the initial load of `sample_credit_model.pkl` and `sample_credit_data.csv` when `source.py` is first imported. For the *actual* audit-ready hashes of your dynamically uploaded model and data, please refer to the `config_snapshot.json` within the exported artifact bundle.")

        st.markdown(f"")
        st.markdown(f"The `explanation_summary.md` document captures Anya's comprehensive analysis. It consolidates findings from global importance, local decision breakdowns, and counterfactual scenarios. Crucially, it identifies specific 'interpretability gaps' – areas where explanations might be less straightforward or require further context – such as opaque feature interactions or explanations for borderline cases. For each gap, Anya has provided a pragmatic recommendation for PrimeCredit Bank, demonstrating a proactive approach to risk management.")
        st.markdown(f"")
        st.markdown(f"By clearly documenting these observations and providing a recommendation (in this case, approval with caveats), Anya fulfills her role as a Model Validator. This structured summary serves as a primary artifact for review by the Internal Audit team and senior leadership, enabling them to make an informed decision on the CAM v1.2's production readiness with a full understanding of its explainability profile.")
```

#### **Page 7: 6. Export Artifacts**

This final page handles the bundling of all generated artifacts into an audit-ready ZIP file.

```python
elif st.session_state.current_page == "6. Export Artifacts":
    st.title("7. Audit Trail: Reproducibility and Artifact Bundling")
    st.markdown(f"The final, critical step for Anya is to ensure that all her validation work is reproducible and securely bundled for auditing purposes. For PrimeCredit Bank, regulatory compliance demands an immutable record of all explanation artifacts, along with the configuration and hashes that guarantee their traceability to specific model and data versions. This 'audit-ready artifact bundle' acts as indisputable evidence of the model validation process. I will consolidate all generated explanations, configuration details, and an `evidence_manifest.json` containing SHA-256 hashes of each file, into a single, timestamped ZIP archive.")
    st.markdown(r"The `evidence_manifest.json` will list each generated file and its corresponding SHA-256 hash. The SHA-256 hash function takes an input (e.g., a file's content) and produces a fixed-size, 256-bit (32-byte) hexadecimal string. Even a minuscule change to the input will result in a completely different hash, making it an excellent tool for verifying data integrity:")
    st.markdown(r"$$ \text{{SHA-256}}(\text{{file\_content}}) = \text{{hexadecimal\_hash\_string}} $$")
    st.markdown(f"")

    # Button to trigger the entire bundling process
    if st.button("Generate & Bundle All Audit Artifacts", type="primary", disabled=not st.session_state.model_loaded):
        with st.spinner("Generating configuration, manifest, and bundling artifacts..."):
            try:
                # Ensure the explanation directory for the current run exists
                os.makedirs(st.session_state.explanation_dir, exist_ok=True)
                
                # 1. Create config snapshot, using dynamic hashes from session state
                config_file_path = create_config_snapshot(
                    st.session_state.model_hash, 
                    st.session_state.data_hash, 
                    st.session_state.RANDOM_SEED, 
                    st.session_state.explanation_dir
                )
                
                # 2. Define all output files that need to be hashed and bundled.
                # These are files that *should* have been generated in previous steps.
                output_files_candidates = [
                    os.path.join(st.session_state.explanation_dir, 'global_explanation.json'),
                    os.path.join(st.session_state.explanation_dir, 'local_explanation.json'),
                    os.path.join(st.session_state.explanation_dir, 'counterfactual_example.json'),
                    os.path.join(st.session_state.explanation_dir, 'explanation_summary.md'),
                    config_file_path
                ]
                # Filter to only include files that actually exist
                st.session_state.output_files_to_bundle = [f for f in output_files_candidates if os.path.exists(f)]

                # 3. Create evidence manifest
                manifest_file_path = create_evidence_manifest(
                    st.session_state.explanation_dir, 
                    st.session_state.output_files_to_bundle
                )
                st.session_state.output_files_to_bundle.append(manifest_file_path) # Add manifest itself to the list for bundling

                # 4. Bundle all artifacts into a zip file
                zip_archive_path = bundle_artifacts_to_zip(st.session_state.explanation_dir, st.session_state.run_id)
                st.session_state.zip_archive_path = zip_archive_path
                st.success(f"All audit-ready artifacts bundled into: `{zip_archive_path}`")
            except Exception as e:
                st.error(f"Error bundling artifacts: {e}")

    # Display download button if a zip archive was successfully created
    if st.session_state.zip_archive_path and os.path.exists(st.session_state.zip_archive_path):
        with open(st.session_state.zip_archive_path, "rb") as fp:
            st.download_button(
                label="Download Audit-Ready Artifact Bundle",
                data=fp.read(),
                file_name=os.path.basename(st.session_state.zip_archive_path),
                mime="application/zip",
                help="Download a ZIP file containing all explanation artifacts, configuration, and evidence manifest."
            )
        st.markdown(f"")
        st.markdown(f"The final stage of the validation workflow is complete. I have successfully generated a comprehensive set of explanation artifacts, including global and local SHAP analyses, counterfactual examples, and my detailed summary report. Each of these documents, along with a snapshot of the configuration (including model and data hashes) and a manifest of all files with their individual SHA-256 hashes, has been meticulously bundled into a timestamped ZIP archive: `{os.path.basename(st.session_state.zip_archive_path)}`.")
        st.markdown(f"")
        st.markdown(f"This single, self-contained archive is PrimeCredit Bank's **audit-ready artifact bundle**. It ensures:")
        st.markdown(f"1.  **Reproducibility:** The `config_snapshot.json` captures all parameters needed to regenerate these explanations, including the exact model and data hashes.")
        st.markdown(f"2.  **Traceability:** The `evidence_manifest.json` provides cryptographic proof of the integrity and origin of each artifact, linking them directly to the validated model and data versions.")
        st.markdown(f"3.  **Compliance:** All necessary documentation for internal auditors, regulators, and senior stakeholders is readily available and verifiable, significantly reducing regulatory risk and building trust in the AI system.")
        st.markdown(f"")
        st.markdown(f"This completes my model validation task for CAM v1.2, providing PrimeCredit Bank with the necessary confidence to proceed with its deployment, knowing its decisions are explainable, transparent, and auditable.")
```
