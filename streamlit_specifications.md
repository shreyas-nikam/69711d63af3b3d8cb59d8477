
## Streamlit Application Specification: Model Validation & Explainability Approval Workbench

### 1. Application Overview

**Purpose of the Application**

This Streamlit application serves as a Model Validation & Explainability Approval Workbench, operationalizing model interpretability as a formal enterprise control. Its core purpose is to enable Model Validators like Anya Sharma to rigorously assess the transparency, fairness, and compliance of machine learning models before deployment. It ensures that model decisions can be explained to auditors, regulators, and internal stakeholders in a reproducible and defensible manner, generating audit-ready explanation artifacts.

**High-Level Story Flow of the Application**

1.  **Anya (Model Validator) arrives at the workbench (Home Page).** She is introduced to her mission: validating PrimeCredit Bank's new Credit Approval Model (CAM v1.2).
2.  **Anya navigates to "1. Data & Model Loading".** She either loads the provided sample credit model and data or uploads custom model and feature data. The system automatically computes cryptographic hashes for these artifacts to ensure traceability and reproducibility, preparing the data for explanation generation.
3.  **Anya proceeds to "2. Global Explanations".** She triggers the generation of global SHAP explanations to understand the overall drivers of the model's decisions (e.g., top influential features like `credit_score`, `income`). This provides an aggregate view of feature importance.
4.  **Anya moves to "3. Local Explanations".** She selects specific instances from the dataset (e.g., a denied, an approved, and a borderline loan application) to investigate their individual predictions. The system generates detailed local SHAP explanations, showing how each feature contributed to the specific outcome for that applicant.
5.  **Anya explores "4. Counterfactuals".** For a chosen denied loan applicant, she generates counterfactual explanations. The system identifies the minimal changes to the applicant's profile that would have resulted in a loan approval, providing actionable feedback.
6.  **Anya reviews "5. Validation Summary".** She initiates the generation of a comprehensive summary report, synthesizing findings from the global, local, and counterfactual analyses. This report highlights interpretability gaps and provides recommendations for the model's deployment.
7.  **Finally, Anya goes to "6. Export Artifacts".** She bundles all generated explanations, configuration snapshots, and an evidence manifest (with SHA-256 hashes of all files) into a single, timestamped, audit-ready ZIP archive. This completes her validation task, providing PrimeCredit Bank with a verifiable record of the model's explainability.

---

### 2. Code Requirements

**Import Statements**

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
import json
import zipfile
import datetime
import matplotlib.pyplot as plt # For displaying SHAP plots
import shutil # For cleanup of explanation directories
import tempfile # For handling uploaded files

# Import all functions and global variables from source.py
from source import *
```

**`st.session_state` Design and Usage**

`st.session_state` is used extensively to preserve the application state across reruns and to simulate a multi-page experience.

**Initialization:**
All `st.session_state` keys are initialized on the first run of the application.

```python
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'
    st.session_state['model'] = None
    st.session_state['full_data'] = None # Data with target column
    st.session_state['X'] = None # Features only
    st.session_state['y'] = None # Target only
    st.session_state['model_hash_val'] = None
    st.session_state['data_hash_val'] = None
    st.session_state['explainer_local'] = None # SHAP TreeExplainer instance
    st.session_state['global_importance_df'] = pd.DataFrame() # DataFrame for global importance
    st.session_state['local_explanations_data'] = {} # Dictionary to store local explanations
    st.session_state['counterfactual_result'] = {} # Dictionary to store counterfactual results
    st.session_state['explanation_summary_content'] = "" # Markdown string for summary
    st.session_state['EXPLANATION_DIR'] = None # Dynamic directory for current run artifacts
    st.session_state['run_id'] = None # Unique ID for the current run
    st.session_state['instances_for_local_explanation'] = [] # List of selected indices for local explanations
    st.session_state['denied_instance_for_cf_idx'] = None # Index of instance for counterfactual
    st.session_state['config_file'] = None # Path to config snapshot file
    st.session_state['output_files_to_bundle'] = [] # List of files to include in zip
    st.session_state['manifest_file'] = None # Path to evidence manifest file
    st.session_state['zip_archive_path'] = None # Path to the final zip file
    st.session_state['uploaded_model_file_obj'] = None # Raw uploaded file object from st.file_uploader
    st.session_state['uploaded_data_file_obj'] = None # Raw uploaded file object from st.file_uploader
    st.session_state['sample_data_loaded'] = False # Flag if sample data is loaded
    st.session_state['model_loaded'] = False # Flag if model object is in session state
    st.session_state['data_ready'] = False # Flag if data (X, y) is ready for explanation
    st.session_state['global_explanations_generated'] = False # Flag for completion of global explanations
    st.session_state['local_explanations_generated'] = False # Flag for completion of local explanations
    st.session_state['counterfactuals_generated'] = False # Flag for completion of counterfactuals
    st.session_state['summary_generated'] = False # Flag for completion of summary
    st.session_state['artifacts_bundled'] = False # Flag for completion of artifact bundling
    st.session_state['X_train_exp'] = None # Training features for SHAP explainer background
    st.session_state['positive_class_idx_in_model_output'] = None # Index of the positive class in model.predict_proba output
    st.session_state['target_column'] = TARGET_COLUMN # Use TARGET_COLUMN from source.py
    st.session_state['random_seed'] = RANDOM_SEED # Use RANDOM_SEED from source.py
    st.session_state['temp_model_path'] = None # Path to temporarily saved uploaded model
    st.session_state['temp_data_path'] = None # Path to temporarily saved uploaded data
    # Create a persistent directory for temporary uploads
    os.makedirs("temp_uploads", exist_ok=True)
```

**Updating and Reading across Pages:**

*   **`st.session_state.current_page`**: Updated by the sidebar `st.selectbox` to control which page content is rendered.
*   **Model and Data (`model`, `full_data`, `X`, `y`, `model_hash_val`, `data_hash_val`, `explainer_local`, `X_train_exp`, `positive_class_idx_in_model_output`)**:
    *   **Initialized/Updated**: In "1. Data & Model Loading" when "Load Custom Model & Data" or "Load Sample Credit Model & Data" buttons are clicked.
        *   For custom uploads: Temporary files are created from `st.file_uploader` objects, their paths are passed to `load_and_hash_artifacts`.
        *   For sample data: `generate_sample_data_and_model` ensures files exist, then `load_and_hash_artifacts` is called with `MODEL_PATH` and `DATA_PATH` (from `source.py`).
        *   `st.session_state.explainer_local` is initialized with `shap.TreeExplainer(st.session_state.model)`.
        *   `st.session_state.X_train_exp` is set by splitting `st.session_state.X`.
        *   `st.session_state.positive_class_idx_in_model_output` is determined based on `model.classes_` and `TARGET_COLUMN`.
    *   **Read**: These variables are read across all subsequent pages (Global Explanations, Local Explanations, Counterfactuals, Summary, Export) as arguments to `source.py` functions.
    *   **Crucial Assumption for `source.py` globals**: Due to the constraint "Do not redefine, rewrite, stub, or duplicate them" for `source.py` functions, and `source.py` functions internally using global variables (e.g., `full_data` in `generate_counterfactual_explanation`), it is assumed that when `source.py` functions like `load_and_hash_artifacts` are called in `app.py`, the updated data (model, full_data, X, y, hashes, paths, random_seed, target_column) is *implicitly made available to subsequent `source.py` function calls* by updating the respective global variables *within the `source` module's scope*. This is achieved by explicitly assigning `source.global_var = st.session_state.global_var` before calling a function that relies on that global.

*   **Explanation Results (`global_importance_df`, `local_explanations_data`, `counterfactual_result`, `explanation_summary_content`)**:
    *   **Initialized/Updated**: When their respective "Generate" buttons are clicked on pages "2. Global Explanations", "3. Local Explanations", "4. Counterfactuals", and "5. Validation Summary".
    *   **Read**: Used for display on their generation page and as inputs to subsequent explanation generation steps (e.g., `local_explanations_data` is used by `generate_explanation_summary`).

*   **Flags (`model_loaded`, `data_ready`, `global_explanations_generated`, etc.)**:
    *   **Initialized**: All set to `False` initially.
    *   **Updated**: Set to `True` upon successful completion of the corresponding step.
    *   **Read**: Used to disable/enable buttons and control progression through the multi-page workflow. If an upstream step is rerun, all downstream flags are reset to `False` to ensure a fresh explanation pipeline.

*   **Artifact/Run Information (`EXPLANATION_DIR`, `run_id`, `config_file`, `output_files_to_bundle`, `manifest_file`, `zip_archive_path`, `temp_model_path`, `temp_data_path`)**:
    *   **Initialized/Updated**: `run_id` and `EXPLANATION_DIR` are generated and updated in "2. Global Explanations" to ensure a unique directory for each explanation run. Subsequent artifact paths (`config_file`, `manifest_file`, `zip_archive_path`) are updated as they are generated in "6. Export Artifacts". `temp_model_path` and `temp_data_path` are set when custom files are uploaded.
    *   **Read**: Used for passing directory paths to `source.py` functions and for generating the download button.

**UI Interactions and `source.py` Function Calls**

**Sidebar:**
*   `st.sidebar.selectbox("Go to", [...])`: Updates `st.session_state.current_page`.

---

**Page: Home**
*   **Markdown**: Application overview, Model Validator persona introduction.
*   **No `source.py` calls**.

---

**Page: 1. Data & Model Loading**
*   **Markdown**: Importance of reproducible setup, hashing, context for loading data.
*   **Widgets**:
    *   `st.file_uploader("Upload Model (.pkl)", type=["pkl"])`: Stores `UploadedFile` object in `st.session_state.uploaded_model_file_obj`.
    *   `st.file_uploader("Upload Feature Data (.csv)", type=["csv"])`: Stores `UploadedFile` object in `st.session_state.uploaded_data_file_obj`.
    *   `st.button("Load Custom Model & Data", disabled=...)`:
        *   **Calls**:
            *   Saves `st.session_state.uploaded_model_file_obj` and `st.session_state.uploaded_data_file_obj` to temporary files (`temp_model_path`, `temp_data_path`) using `tempfile.NamedTemporaryFile`.
            *   `model, full_data, X, y, model_hash, data_hash = load_and_hash_artifacts(st.session_state.temp_model_path, st.session_state.temp_data_path, st.session_state.target_column, st.session_state.random_seed)`
            *   `st.session_state.X_train_exp, _, _, _ = train_test_split(st.session_state.X, st.session_state.y, test_size=0.2, random_state=st.session_state.random_seed)`
            *   `st.session_state.explainer_local = shap.TreeExplainer(st.session_state.model)`
            *   Identifies `st.session_state.positive_class_idx_in_model_output`.
        *   **Updates**: `st.session_state.model`, `st.session_state.full_data`, `st.session_state.X`, `st.session_state.y`, `st.session_state.model_hash_val`, `st.session_state.data_hash_val`, `st.session_state.model_loaded`, `st.session_state.data_ready`, `st.session_state.sample_data_loaded` (to `False`), `st.session_state.X_train_exp`, `st.session_state.explainer_local`, `st.session_state.positive_class_idx_in_model_output`. Resets all downstream `_generated` flags to `False`.
    *   `st.button("Load Sample Credit Model & Data")`:
        *   **Calls**:
            *   `generate_sample_data_and_model(MODEL_PATH, DATA_PATH, st.session_state.target_column, st.session_state.random_seed)`
            *   `model, full_data, X, y, model_hash, data_hash = load_and_hash_artifacts(MODEL_PATH, DATA_PATH, st.session_state.target_column, st.session_state.random_seed)`
            *   `st.session_state.X_train_exp, _, _, _ = train_test_split(st.session_state.X, st.session_state.y, test_size=0.2, random_state=st.session_state.random_seed)`
            *   `st.session_state.explainer_local = shap.TreeExplainer(st.session_state.model)`
            *   Identifies `st.session_state.positive_class_idx_in_model_output`.
        *   **Updates**: `st.session_state.model`, `st.session_state.full_data`, `st.session_state.X`, `st.session_state.y`, `st.session_state.model_hash_val`, `st.session_state.data_hash_val`, `st.session_state.model_loaded`, `st.session_state.data_ready`, `st.session_state.sample_data_loaded` (to `True`), `st.session_state.X_train_exp`, `st.session_state.explainer_local`, `st.session_state.positive_class_idx_in_model_output`. Resets all downstream `_generated` flags to `False`.
*   **Display**: `st.dataframe(st.session_state.X.head())`, displays `model_hash_val`, `data_hash_val`.
*   `st.markdown(f"...")` for non-formula markdown.

---

**Page: 2. Global Explanations**
*   **Markdown**: Introduction to global explanations, `st.markdown(r"$$ \phi_0 + \sum_{{i=1}}^{{M}} \phi_i(f, x) = f(x) $$")` followed by `st.markdown(r"where $\phi_0$ ...")`.
*   **Widgets**:
    *   `st.button("Generate Global Explanations", disabled=...)`:
        *   **Calls**:
            *   Generates a new `run_id` and `EXPLANATION_DIR` to organize outputs for the current run. Cleans up previous `EXPLANATION_DIR` if exists.
            *   `global_importance_df, _ = generate_global_shap_explanation(st.session_state.model, st.session_state.X_train_exp, st.session_state.X_train_exp.columns.tolist(), st.session_state.EXPLANATION_DIR)`
            *   A temporary SHAP plot is re-rendered to a `matplotlib` figure: `plt.figure(); shap.summary_plot(..., show=False); st.pyplot(plt.gcf())`.
        *   **Updates**: `st.session_state.run_id`, `st.session_state.EXPLANATION_DIR`, `st.session_state.global_importance_df`, `st.session_state.global_explanations_generated`. Resets all downstream `_generated` flags to `False`.
*   **Display**: `st.dataframe(st.session_state.global_importance_df.head(10))`, `st.pyplot(plt.gcf())` for SHAP summary plot.
*   `st.markdown(f"...")` for non-formula markdown.

---

**Page: 3. Local Explanations**
*   **Markdown**: Introduction to local explanations, `st.markdown(r"For a specific instance $x$, the SHAP values $\phi_i(f, x)$ quantify how much each feature $i$ contributes to the prediction $f(x)$ compared to the average prediction $\phi_0$. A positive SHAP value for a feature means it pushed the prediction higher (towards approval), while a negative value pushed it lower (towards denial).")`.
*   **Widgets**:
    *   `st.multiselect("Select up to 3 instance IDs...", options=st.session_state.X.index.tolist(), default=..., max_selections=3)`: Updates `st.session_state.instances_for_local_explanation`. Pre-selects a denied, approved, and borderline case if none are chosen.
    *   `st.button("Generate Local Explanations", disabled=...)`:
        *   **Calls**:
            *   `local_explanations_data, _ = generate_local_shap_explanations(st.session_state.model, st.session_state.X, st.session_state.instances_for_local_explanation, st.session_state.explainer_local, st.session_state.EXPLANATION_DIR)`
            *   For each instance, a temporary SHAP plot is re-rendered: `plt.figure(); shap.waterfall_plot(..., show=False); st.pyplot(plt.gcf())`.
        *   **Updates**: `st.session_state.local_explanations_data`, `st.session_state.local_explanations_generated`. Resets all downstream `_generated` flags to `False`.
*   **Display**: For each selected instance: `st.dataframe` for original features, predicted probability, and `st.pyplot(plt.gcf())` for SHAP waterfall plot.
*   `st.markdown(f"...")` for non-formula markdown.

---

**Page: 4. Counterfactuals**
*   **Markdown**: Introduction to counterfactual explanations, `st.markdown(r"$$ \min_{{x'}} \text{{distance}}(x, x') \quad \text{{s.t.}} \quad f(x') = y' \quad \text{{and}} \quad x' \in \mathcal{{X}} $$")` followed by `st.markdown(r"where $\text{{distance}}(x, x')$ ...")`.
*   **Widgets**:
    *   `st.selectbox("Select a denied instance ID...", options=denied_indices, index=...)`: Updates `st.session_state.denied_instance_for_cf_idx`. Pre-selects the first denied instance if none is chosen.
    *   `st.button("Generate Counterfactual Example", disabled=...)`:
        *   **Calls**:
            *   **Pre-call global updates**: `source.full_data = st.session_state.full_data`, `source.X = st.session_state.X`, `source.y = st.session_state.y`, `source.credit_model = st.session_state.model`, `source.MODEL_PATH`, `source.DATA_PATH`, `source.model_hash_val`, `source.data_hash_val`, `source.RANDOM_SEED` are explicitly updated to reflect `st.session_state` values. This is necessary because `source.py` functions (e.g., `dice_ml.Data` constructor within `generate_counterfactual_explanation`) rely on global variables defined in `source.py`, which would otherwise only reflect the state at `source.py`'s initial import.
            *   `counterfactual_result = generate_counterfactual_explanation(st.session_state.model, st.session_state.X, st.session_state.X.columns.tolist(), st.session_state.denied_instance_for_cf_idx, 1, st.session_state.EXPLANATION_DIR)`
        *   **Updates**: `st.session_state.counterfactual_result`, `st.session_state.counterfactuals_generated`. Resets all downstream `_generated` flags to `False`.
*   **Display**: `st.dataframe` for original instance features, predicted probability, `st.dataframe` for counterfactual instance features, predicted probability, and `st.write` for features changed.
*   `st.markdown(f"...")` for non-formula markdown.

---

**Page: 5. Validation Summary**
*   **Markdown**: Introduction to interpretability analysis and validation findings.
*   **Widgets**:
    *   `st.button("Generate Explanation Summary", disabled=...)`:
        *   **Calls**:
            *   **Pre-call global updates**: `source.model_hash_val`, `source.data_hash_val`, `source.TARGET_COLUMN` are explicitly updated to reflect `st.session_state` values.
            *   `generate_explanation_summary(st.session_state.global_importance_df, st.session_state.local_explanations_data, st.session_state.counterfactual_result, st.session_state.EXPLANATION_DIR)`
            *   Reads the generated `explanation_summary.md` file into `st.session_state.explanation_summary_content`.
        *   **Updates**: `st.session_state.explanation_summary_content`, `st.session_state.summary_generated`. Resets all downstream `_generated` flags to `False`.
*   **Display**: `st.markdown(st.session_state.explanation_summary_content)`.
*   `st.markdown(f"...")` for non-formula markdown.

---

**Page: 6. Export Artifacts**
*   **Markdown**: Introduction to audit trail and artifact bundling, `st.markdown(r"$$ \text{{SHA-256}}(\text{{file\_content}}) = \text{{hexadecimal\_hash\_string}} $$")` followed by `st.markdown(r"where $\text{{SHA-256}}$ ...")`.
*   **Widgets**:
    *   `st.button("Export Audit-Ready Bundle (.zip)", disabled=...)`:
        *   **Calls**:
            *   **Pre-call global updates**: `source.model_hash_val`, `source.data_hash_val`, `source.RANDOM_SEED`, `source.MODEL_PATH`, `source.DATA_PATH`, `source.TARGET_COLUMN` are explicitly updated to reflect `st.session_state` values.
            *   `config_file = create_config_snapshot(st.session_state.model_hash_val, st.session_state.data_hash_val, st.session_state.random_seed, st.session_state.EXPLANATION_DIR)`
            *   `st.session_state.output_files_to_bundle` is populated with paths to `global_explanation.json`, `local_explanation.json`, `counterfactual_example.json`, `explanation_summary.md`, and `config_file`.
            *   `manifest_file = create_evidence_manifest(st.session_state.EXPLANATION_DIR, st.session_state.output_files_to_bundle)`
            *   Adds `manifest_file` to `st.session_state.output_files_to_bundle`.
            *   `zip_archive_path = bundle_artifacts_to_zip(st.session_state.EXPLANATION_DIR, st.session_state.run_id)`
        *   **Updates**: `st.session_state.config_file`, `st.session_state.output_files_to_bundle`, `st.session_state.manifest_file`, `st.session_state.zip_archive_path`, `st.session_state.artifacts_bundled`.
    *   `st.download_button(...)`: Appears after bundling is complete, allowing download of the `zip_archive_path`.
*   **Display**: Confirmation message and `st.download_button`.
*   `st.markdown(f"...")` for non-formula markdown.
*   **Cleanup**: At the end of the script execution (or on each rerun), temporary files and directories created for uploaded files (`temp_uploads`) or explanation artifacts (`reports/session_05_validation_run_*`) should be cleaned up after download or session end. For this specification, assuming cleanup happens implicitly after download or app exit, or manual cleanup by user/system. Explicit `shutil.rmtree` is included for old `EXPLANATION_DIR` when a new one is created.

---
