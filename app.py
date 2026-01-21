import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import zipfile
import hashlib
import json
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

# Import all functions and necessary constants from source.py
from source import (
    RANDOM_SEED, MODEL_PATH, DATA_PATH, TARGET_COLUMN,
    calculate_file_hash, load_and_hash_artifacts,
    generate_sample_data_and_model,
    generate_global_shap_explanation, generate_local_shap_explanations,
    generate_counterfactual_explanation, generate_explanation_summary,
    create_config_snapshot, create_evidence_manifest, bundle_artifacts_to_zip
)

# Set page config
st.set_page_config(page_title="QuLab: Lab 5: Interpretability & Explainability Control Workbench", layout="wide")

# QuLab Header
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 5: Interpretability & Explainability Control Workbench")
st.divider()

# --- Initialization & Helper Functions ---

# Global initialization to ensure sample data exists for demonstration/testing.
if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    # Using print logic as per spec, usually st.write/st.info is better but following spec structure
    print(f"Sample model/data not found, generating to {MODEL_PATH} and {DATA_PATH}...")
    try:
        generate_sample_data_and_model(MODEL_PATH, DATA_PATH, TARGET_COLUMN, RANDOM_SEED)
        print("Sample model/data generation complete.")
    except Exception as e:
        print(f"Warning: Could not generate sample data: {e}")
else:
    print(f"Sample model/data found at {MODEL_PATH} and {DATA_PATH}.")

# Helper function to clear previous explanation artifacts
def clear_explanation_directory(base_dir="reports"):
    """Clears all subdirectories within the base explanation reports directory."""
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)
    os.makedirs(base_dir, exist_ok=True) # Ensure the base directory exists

# Helper function to display SHAP plots
def display_shap_plot(plot_func, *args, **kwargs):
    """Generates and displays a SHAP plot in Streamlit, ensuring new figures are created and closed."""
    fig = plt.figure() # Create a new figure
    plot_func(*args, **kwargs, show=False) # Call SHAP plot function
    st.pyplot(fig) # Display the figure in Streamlit
    plt.close(fig) # Close the figure to prevent display issues in subsequent renders

# --- Session State Management ---

if 'page' not in st.session_state:
    st.session_state.page = 'Welcome'
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'model_hash' not in st.session_state:
    st.session_state.model_hash = None
if 'data_hash' not in st.session_state:
    st.session_state.data_hash = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'global_importance_df' not in st.session_state:
    st.session_state.global_importance_df = pd.DataFrame()
if 'global_shap_values' not in st.session_state:
    st.session_state.global_shap_values = None
if 'local_explanations_data' not in st.session_state:
    st.session_state.local_explanations_data = {}
if 'local_shap_explanations_list' not in st.session_state:
    st.session_state.local_shap_explanations_list = []
if 'counterfactual_result' not in st.session_state:
    st.session_state.counterfactual_result = {}
if 'explanation_dir' not in st.session_state:
    st.session_state.explanation_dir = None
if 'run_id' not in st.session_state:
    st.session_state.run_id = None
if 'zip_archive_path' not in st.session_state:
    st.session_state.zip_archive_path = None
if 'selected_instance_for_local_exp' not in st.session_state:
    st.session_state.selected_instance_for_local_exp = None
if 'denied_instance_for_cf_idx' not in st.session_state:
    st.session_state.denied_instance_for_cf_idx = None
if 'X_train_exp' not in st.session_state:
    st.session_state.X_train_exp = None
if 'X_test_exp' not in st.session_state:
    st.session_state.X_test_exp = None
if 'current_uploaded_model_name' not in st.session_state:
    st.session_state.current_uploaded_model_name = None
if 'current_uploaded_data_name' not in st.session_state:
    st.session_state.current_uploaded_data_name = None

# --- Navigation ---

with st.sidebar:
    st.title("Navigation")
    page_selection = st.radio(
        "Go to",
        ('Welcome', 'Setup & Data', 'Global Explanations', 'Local Explanations', 'Counterfactuals', 'Summary & Audit')
    )
    if page_selection:
        st.session_state.page = page_selection

# --- Main Content Area ---

# 1. Welcome Page
if st.session_state.page == 'Welcome':
    st.title("Model Explanation & Explainability Control Workbench")
    st.markdown(f"")
    st.markdown(f"## Validating PrimeCredit Bank's Loan Approval Model")
    st.markdown(f"")
    st.markdown(f"### 1. Introduction: The Model Validator's Mission at PrimeCredit Bank")
    st.markdown(f"")
    st.markdown(f"As Anya Sharma, a dedicated Model Validator at PrimeCredit Bank, my primary responsibility is to ensure that all machine learning models used in critical business processes are transparent, fair, and compliant with internal governance and external regulatory standards. Today, my focus is on a newly developed **Credit Approval Model (CAM v1.2)**, which will determine loan eligibility for our customers. Before this model can be deployed, I must rigorously assess its interpretability and explainability.")
    st.markdown(f"")
    st.markdown(f"My goal is to thoroughly vet this model, identifying any interpretability gaps that could lead to biased decisions, regulatory scrutiny, or a lack of trust from stakeholders. I need to demonstrate that the model's decisions are defensible and understandable, not just to me, but also to internal auditors and future regulators. This workbench will help me generate, analyze, and document the required explanations as audit-ready artifacts.")

# 2. Setup & Data Page
elif st.session_state.page == 'Setup & Data':
    st.title("2. Setting the Stage: Environment Setup and Data Ingestion")
    st.markdown(f"")
    st.markdown(f"My first step is to prepare my environment and load the necessary model and data for validation. Reproducibility is paramount in model validation; therefore, I will fix a random seed and compute SHA-256 hashes for both the model and dataset to ensure traceability and detect any unauthorized changes.")
    st.markdown(f"")
    st.markdown(f"### a. Story + Context + Real-World Relevance")
    st.markdown(f"")
    st.markdown(f"To begin, I need to install all required Python libraries. Following this, I'll import them and load the pre-trained `sample_credit_model.pkl` and its corresponding feature dataset, `sample_credit_data.csv`. This dataset represents historical loan applications with various features and an outcome indicating whether the loan was approved. Ensuring the integrity of the model and data is crucial; hence, I'll calculate unique cryptographic hashes for each, which will serve as a foundational element for auditability.")
    st.markdown(f"")

    st.subheader("Upload Model and Data")
    uploaded_model_file = st.file_uploader("Upload Model (.pkl or .joblib)", type=['pkl', 'joblib'], key='model_uploader')
    uploaded_data_file = st.file_uploader("Upload Data (.csv)", type=['csv'], key='data_uploader')

    # Logic to detect new file uploads and clear session state for fresh processing
    if (uploaded_model_file and uploaded_model_file.name != st.session_state.current_uploaded_model_name) or \
       (uploaded_data_file and uploaded_data_file.name != st.session_state.current_uploaded_data_name):
        # Reset relevant session state variables when new files are uploaded
        st.session_state.model = None
        st.session_state.data = None
        st.session_state.X = None
        st.session_state.y = None
        st.session_state.model_hash = None
        st.session_state.data_hash = None
        st.session_state.explainer = None
        st.session_state.global_importance_df = pd.DataFrame()
        st.session_state.global_shap_values = None
        st.session_state.local_explanations_data = {}
        st.session_state.local_shap_explanations_list = []
        st.session_state.counterfactual_result = {}
        st.session_state.explanation_dir = None
        st.session_state.run_id = None
        st.session_state.zip_archive_path = None
        st.session_state.selected_instance_for_local_exp = None
        st.session_state.denied_instance_for_cf_idx = None
        st.session_state.X_train_exp = None
        st.session_state.X_test_exp = None
        if uploaded_model_file:
            st.session_state.current_uploaded_model_name = uploaded_model_file.name
        if uploaded_data_file:
            st.session_state.current_uploaded_data_name = uploaded_data_file.name

    if uploaded_model_file and uploaded_data_file:
        st.success(f"Model: `{uploaded_model_file.name}` and Data: `{uploaded_data_file.name}` uploaded.")

        uploaded_files_dir = "uploaded_files_temp"
        os.makedirs(uploaded_files_dir, exist_ok=True)
        temp_model_path = os.path.join(uploaded_files_dir, uploaded_model_file.name)
        temp_data_path = os.path.join(uploaded_files_dir, uploaded_data_file.name)

        # Save uploaded files to local temporary paths
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        with open(temp_data_path, "wb") as f:
            f.write(uploaded_data_file.getbuffer())

        if st.button("Load and Hash Artifacts", key='load_artifacts_button'):
            with st.spinner("Loading model and data, calculating hashes..."):
                # Clear previous session run artifacts to ensure a fresh start for each load
                clear_explanation_directory("reports")

                # Generate a new unique run ID and directory for this session
                st.session_state.run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                st.session_state.explanation_dir = f'reports/session_05_validation_run_{st.session_state.run_id}'
                os.makedirs(st.session_state.explanation_dir, exist_ok=True)
                
                try:
                    # Call load_and_hash_artifacts from source.py with paths to uploaded files
                    model, full_data, X, y, model_hash_val, data_hash_val = load_and_hash_artifacts(
                        temp_model_path, temp_data_path, TARGET_COLUMN, RANDOM_SEED
                    )
                    # Store loaded artifacts and hashes in session state
                    st.session_state.model = model
                    st.session_state.data = full_data
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.model_hash = model_hash_val
                    st.session_state.data_hash = data_hash_val

                    # Prepare data subsets for SHAP explainer (X_train_exp for background, X_test_exp for plotting sample)
                    st.session_state.X_train_exp, st.session_state.X_test_exp, _, _ = train_test_split(
                        X, y, test_size=0.2, random_state=RANDOM_SEED
                    )
                    
                    # Initialize SHAP TreeExplainer and store it in session state
                    st.session_state.explainer = shap.TreeExplainer(model)
                    st.success("Model and data loaded successfully and hashes calculated!")
                except Exception as e:
                    st.error(f"Error loading artifacts: {e}. Please ensure correct file formats and `TARGET_COLUMN` (`{TARGET_COLUMN}`).")
                    st.exception(e)
    else:
        st.info("Please upload both a model file (e.g., `sample_credit_model.pkl`) and a data file (e.g., `sample_credit_data.csv`) to proceed.")

    # Display summary of loaded artifacts if available in session state
    if st.session_state.model is not None and st.session_state.data is not None:
        st.subheader("Loaded Artifacts Summary:")
        st.markdown(f"**Model Type:** `{type(st.session_state.model)}`")
        st.markdown(f"**Model Hash (SHA-256):** `{st.session_state.model_hash}`")
        st.markdown(f"**Data Hash (SHA-256):** `{st.session_state.data_hash}`")
        st.markdown(f"**Features:** `{', '.join(st.session_state.X.columns.tolist())}`")
        st.markdown(f"**Target Column:** `{TARGET_COLUMN}`")
        
    st.markdown(f"")
    st.markdown(f"### c. Explanation of Execution")
    st.markdown(f"")
    st.markdown(f"The initial setup is complete. I've successfully loaded the Credit Approval Model and the associated feature dataset. Crucially, I've generated cryptographic hashes for both artifacts: `{st.session_state.model_hash}` for the model and `{st.session_state.data_hash}` for the data. These hashes are vital for maintaining an immutable audit trail; any future change to either the model or the dataset would result in a different hash, immediately signaling a potential issue to an auditor. This step aligns with PrimeCredit Bank's stringent requirements for data and model integrity. The data has also been pre-processed, separating features from the target variable, making it ready for explanation generation.")

# 3. Global Explanations Page
elif st.session_state.page == 'Global Explanations':
    st.title("3. Unveiling Overall Behavior: Global Model Explanations")
    st.markdown(f"")
    st.markdown(f"### a. Story + Context + Real-World Relevance")
    st.markdown(f"")
    st.markdown(f"As a Model Validator, I first need to grasp the overall behavior of the CAM v1.2. Which factors generally drive its decisions for approving or denying loans? Global explanations provide an aggregate view of feature importance, revealing which features have the most impact across all predictions. This helps me verify if the model's general logic aligns with PrimeCredit's lending policies and expert domain knowledge. For tree-based models like our `RandomForestClassifier`, SHAP (SHapley Additive exPlanations) values are an excellent choice for this. The SHAP value $\phi_i$ for a feature $i$ represents the average marginal contribution of that feature value to the prediction across all possible coalitions of features.")
    st.markdown(f"")
    st.markdown(r"$$ \phi_0 + \sum_{{i=1}}^{{M}} \phi_i(f, x) = f(x) $$")
    st.markdown(r"where $\phi_0$ is the expected model output (the base value), $M$ is the number of features, $\phi_i(f, x)$ is the SHAP value for feature $i$ for instance $x$, and $f(x)$ is the model's prediction for instance $x$.")
    st.markdown(f"")

    if st.session_state.model is not None and st.session_state.X_train_exp is not None and st.session_state.explainer is not None and st.session_state.explanation_dir is not None:
        if st.button("Generate Global Explanations", key='generate_global_exp_button'):
            with st.spinner("Generating global SHAP explanations..."):
                try:
                    # Call generate_global_shap_explanation from source.py
                    global_importance_df, global_shap_values = generate_global_shap_explanation(
                        st.session_state.model, st.session_state.X_train_exp, st.session_state.X_train_exp.columns.tolist(), st.session_state.explanation_dir
                    )
                    # Update session state with results
                    st.session_state.global_importance_df = global_importance_df
                    st.session_state.global_shap_values = global_shap_values
                    st.success("Global explanations generated successfully!")
                except Exception as e:
                    st.error(f"Error generating global explanations: {e}")
                    st.exception(e)
        
        if not st.session_state.global_importance_df.empty:
            st.subheader("Global Feature Importance:")
            st.dataframe(st.session_state.global_importance_df)
            
            # Display SHAP summary plot if values are available
            if st.session_state.global_shap_values is not None and st.session_state.X_test_exp is not None:
                st.markdown("### Visualization: Global SHAP Summary Plot")
                # Ensure sample_data for plotting is trimmed to match shap_values length if necessary
                # Typically shap_values correspond to the data passed. If generated on X_train_exp, we plot with X_train_exp or the subset used.
                # The source.py function documentation implies shap_values match the input X.
                # In global generation (source.py usually samples or uses full X_train), let's assume it returns values for X_train_exp or a sample thereof.
                # For visualization, we use the corresponding data.
                
                # Re-checking logic: generate_global_shap_explanation likely computes on X_train_exp.
                # So we plot against X_train_exp.
                shap_vals = st.session_state.global_shap_values
                data_for_plot = st.session_state.X_train_exp
                
                # If shap values are large, summary_plot handles sampling, but let's ensure dimensions match
                if shap_vals.shape[0] != data_for_plot.shape[0]:
                     # Fallback if source.py did internal sampling
                     st.warning("Shape mismatch for plotting. Displaying importance DataFrame only.")
                else:
                    display_shap_plot(shap.summary_plot, shap_vals, data_for_plot, plot_type="bar")
        else:
            st.info("Click 'Generate Global Explanations' to see results.")
    else:
        st.warning("Please upload and load model/data in 'Setup & Data' first to enable global explanations.")

    st.markdown(f"")
    st.markdown(f"### c. Explanation of Execution")
    st.markdown(f"")
    st.markdown(f"The global SHAP explanation reveals the overall drivers of the Credit Approval Model. From the summary plot and the `global_importance_df`, I can clearly see which features, such as `credit_score` and `income`, are most influential in the model's decisions regarding loan approval. This high-level overview confirms that the model is largely relying on expected financial health indicators, which aligns with PrimeCredit Bank's established lending criteria. This gives me initial confidence that the model's general behavior is sensible and explainable to senior stakeholders. However, global explanations only tell part of the story; I need to investigate specific individual decisions to ensure consistency and fairness.")

# 4. Local Explanations Page
elif st.session_state.page == 'Local Explanations':
    st.title("4. Deep Dive into Individual Decisions: Local Explanations for Specific Loan Applications")
    st.markdown(f"")
    st.markdown(f"### a. Story + Context + Real-World Relevance")
    st.markdown(f"")
    st.markdown(f"While global explanations are useful, they don't explain why a *specific* loan applicant was approved or denied. As a Model Validator, I frequently encounter requests to understand individual decisions, especially for denied applications or those with unusual profiles. For PrimeCredit Bank, it's crucial to provide clear, defensible reasons to customers for loan denials or approvals. I will select a few representative cases from our `sample_credit_data` to generate local explanations using SHAP. This allows me to examine the contribution of each feature to that particular prediction.")
    st.markdown(f"")
    st.markdown(f"For a specific instance $x$, the SHAP values $\phi_i(f, x)$ quantify how much each feature $i$ contributes to the prediction $f(x)$ compared to the average prediction $\phi_0$. A positive SHAP value for a feature means it pushed the prediction higher (towards approval), while a negative value pushed it lower (towards denial).")
    st.markdown(f"")

    if st.session_state.model is not None and st.session_state.X is not None and st.session_state.y is not None and st.session_state.explainer is not None and st.session_state.explanation_dir is not None:
        st.subheader("Select Instances for Local Explanation")
        
        # Pre-select some interesting indices for convenience
        denied_indices = st.session_state.y[st.session_state.y == 0].index.tolist()
        approved_indices = st.session_state.y[st.session_state.y == 1].index.tolist()
        
        default_indices = []
        if denied_indices: default_indices.append(denied_indices[0]) # First denied example
        if approved_indices:
            for idx in approved_indices: # First approved example (different from denied)
                if idx not in default_indices:
                    default_indices.append(idx)
                    break
        # Add a borderline example if possible
        if len(st.session_state.X) > 0 and st.session_state.model is not None and hasattr(st.session_state.model, 'classes_') and 1 in st.session_state.model.classes_:
            probabilities = st.session_state.model.predict_proba(st.session_state.X)[:, np.where(st.session_state.model.classes_ == 1)[0][0]]
            borderline_idx_pos_in_X = np.argmin(np.abs(probabilities - 0.5))
            borderline_idx = st.session_state.X.index[borderline_idx_pos_in_X]
            if borderline_idx not in default_indices:
                default_indices.append(borderline_idx)
        
        selected_indices_for_local_exp = st.multiselect(
            "Choose instance indices from the dataset for detailed explanation:",
            options=st.session_state.X.index.tolist(),
            default=default_indices[:3], # Show up to 3 example indices by default
            key='local_exp_instance_selector'
        )

        if st.button("Generate Local Explanations", key='generate_local_exp_button'):
            if not selected_indices_for_local_exp:
                st.warning("Please select at least one instance to explain.")
            else:
                with st.spinner(f"Generating local SHAP explanations for {len(selected_indices_for_local_exp)} instances..."):
                    try:
                        # Call generate_local_shap_explanations from source.py
                        local_explanations_data, shap_values_list = generate_local_shap_explanations(
                            st.session_state.model, st.session_state.X, selected_indices_for_local_exp, 
                            st.session_state.explainer, st.session_state.explanation_dir
                        )
                        # Update session state with results
                        st.session_state.local_explanations_data = local_explanations_data
                        st.session_state.local_shap_explanations_list = shap_values_list
                        
                        # Store one denied instance for convenient pre-selection on the Counterfactuals page
                        denied_for_cf = next((idx for idx in selected_indices_for_local_exp if st.session_state.y.loc[idx] == 0), None)
                        if denied_for_cf is None and denied_indices: # Fallback to first denied if none in selected_indices
                            denied_for_cf = denied_indices[0]
                        st.session_state.denied_instance_for_cf_idx = denied_for_cf

                        st.success("Local explanations generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating local explanations: {e}")
                        st.exception(e)

        if st.session_state.local_explanations_data:
            st.subheader("Local Explanation Results:")
            for i, (inst_key, data) in enumerate(st.session_state.local_explanations_data.items()):
                original_idx = int(inst_key.split('_')[1])
                st.markdown(f"#### Instance ID: {original_idx}")
                st.markdown(f"**Original Features:** `{data['original_features']}`")
                st.markdown(f"**Model Predicted Approval Probability:** `{data['model_prediction']:.4f}`")
                st.markdown(f"**Top SHAP Contributions:**")
                sorted_shap = sorted(data['shap_values'].items(), key=lambda item: abs(item[1]), reverse=True)
                for feature, value in sorted_shap[:5]:
                    st.markdown(f"- `{feature}`: `{value:.4f}`")
                
                # Display SHAP waterfall plot for the instance
                if i < len(st.session_state.local_shap_explanations_list):
                    st.markdown("##### Visualization: SHAP Waterfall Plot")
                    display_shap_plot(shap.waterfall_plot, st.session_state.local_shap_explanations_list[i], max_display=10)
                st.markdown("---")
        else:
            st.info("Select instances and click 'Generate Local Explanations' to see results.")
    else:
        st.warning("Please upload and load model/data in 'Setup & Data' first to enable local explanations.")

    st.markdown(f"")
    st.markdown(f"### c. Explanation of Execution")
    st.markdown(f"")
    st.markdown(f"The local SHAP explanations provide critical insights into individual loan decisions. For instance, analyzing the waterfall plot for a *denied* application (e.g., `instance_ID`), I can clearly see that a low `credit_score` and high `debt_to_income` ratio were the primary negative contributors, pushing the loan approval probability below the threshold. Conversely, for an *approved* application, a high `credit_score` and `income` might be the dominant positive factors. For a *borderline* case, the contributions might be more balanced.")
    st.markdown(f"")
    st.markdown(f"These detailed breakdowns are invaluable for Anya. They allow her to:")
    st.markdown(f"1.  **Verify decision logic:** Are the model's specific reasons for a decision coherent and justifiable according to PrimeCredit's policy?")
    st.markdown(f"2.  **Identify potential biases:** Do certain demographic features (if present) disproportionately influence decisions in specific cases without valid business rationale? (Note: no demographic features are in this sample data, but this is what a Model Validator would look for).")
    st.markdown(f"3.  **Provide actionable feedback:** Understand what factors led to a denial, which is crucial for communicating with applicants.")
    st.markdown(f"")
    st.markdown(f"This level of detail is exactly what PrimeCredit's internal auditors and potentially regulators would require to validate the fairness and transparency of the model.")

# 5. Counterfactuals Page
elif st.session_state.page == 'Counterfactuals':
    st.title("5. 'What If?': Understanding Counterfactuals for Actionable Insights")
    st.markdown(f"")
    st.markdown(f"### a. Story + Context + Real-World Relevance")
    st.markdown(f"")
    st.markdown(f"For a denied loan applicant, merely knowing *why* they were denied (via local explanations) isn't always enough. As Anya, I also need to understand \"what if?\" – what minimal changes to their application would have resulted in an approval? This is where counterfactual explanations come in. They identify the smallest, most actionable changes to an applicant's features that would flip the model's decision from denial to approval. This information is invaluable for PrimeCredit Bank, not only for providing constructive feedback to customers but also for potentially refining our lending criteria or identifying areas where applicants can improve their financial standing to become eligible.")
    st.markdown(f"")
    st.markdown(r"$$ \min_{{x'}} \text{{distance}}(x, x') \quad \text{{s.t.}} \quad f(x') = y' \quad \text{{and}} \quad x' \in \mathcal{{X}} $$")
    st.markdown(r"where $\text{{distance}}(x, x')$ is a measure of proximity (e.g., L1 or L2 norm), and $f(x')$ is the model's prediction for the counterfactual instance $x'$.")
    st.markdown(f"")

    if st.session_state.model is not None and st.session_state.X is not None and st.session_state.data is not None and st.session_state.explanation_dir is not None:
        st.subheader("Generate Counterfactual Explanation")
        
        # Get available denied instances for selection
        denied_indices_options = st.session_state.y[st.session_state.y == 0].index.tolist()
        
        # Pre-select a denied instance if available (from local explanations page or first available)
        default_cf_idx = st.session_state.denied_instance_for_cf_idx
        if default_cf_idx is None and denied_indices_options:
            default_cf_idx = denied_indices_options[0]

        if denied_indices_options: # Only show selector if denied instances exist
            selected_cf_instance_idx = st.selectbox(
                "Select a denied instance (where original `loan_approved` == 0) for counterfactual analysis:",
                options=denied_indices_options,
                # Set default index if default_cf_idx is in options, else 0 or None
                index=denied_indices_options.index(default_cf_idx) if default_cf_idx in denied_indices_options else (0 if denied_indices_options else None),
                format_func=lambda x: f"Instance {x} (Current Pred: {st.session_state.model.predict_proba(st.session_state.X.loc[[x]])[0][np.where(st.session_state.model.classes_ == 1)[0][0]]:.4f})",
                key='cf_instance_selector'
            )
            # Assuming we want to flip a denied loan to an approved one (desired class 1)
            desired_class = st.radio("Desired Outcome:", options=[1, 0], index=0, format_func=lambda x: "Approval (1)" if x == 1 else "Denial (0)", key='cf_desired_class')

            if st.button("Generate Counterfactuals", key='generate_cf_button'):
                if selected_cf_instance_idx is None:
                    st.warning("Please select an instance to generate counterfactuals.")
                else:
                    with st.spinner(f"Generating counterfactual explanation for instance {selected_cf_instance_idx}..."):
                        try:
                            # Call generate_counterfactual_explanation from source.py
                            counterfactual_result = generate_counterfactual_explanation(
                                st.session_state.model, st.session_state.X, st.session_state.X.columns.tolist(),
                                selected_cf_instance_idx, desired_class, st.session_state.explanation_dir
                            )
                            # Update session state with results
                            st.session_state.counterfactual_result = counterfactual_result
                            st.success("Counterfactuals generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating counterfactuals: {e}. DiCE might struggle to find a counterfactual given the data/constraints.")
                            st.exception(e)
        else:
            st.info("No denied instances found in the dataset to generate counterfactuals. Please ensure your dataset contains instances with `loan_approved == 0`.")

        if st.session_state.counterfactual_result:
            st.subheader("Counterfactual Explanation Results:")
            cf_data = st.session_state.counterfactual_result
            if cf_data.get('original_instance'):
                st.markdown(f"**Original Instance (ID {selected_cf_instance_idx}):**")
                st.json(cf_data['original_instance'])
                st.markdown(f"**Original Prediction Probability (Desired Class {desired_class}):** `{cf_data.get('original_prediction_prob_desired_class', 0.0):.4f}`")
                
                if cf_data.get('counterfactual_instance'):
                    st.markdown(f"**Counterfactual Instance:**")
                    st.json(cf_data['counterfactual_instance'])
                    st.markdown(f"**Counterfactual Prediction Probability (Desired Class {desired_class}):** `{cf_data.get('counterfactual_prediction_prob_desired_class', 0.0):.4f}`")
                    st.markdown(f"**Minimal Feature Changes to Flip Prediction:**")
                    if cf_data.get('features_changed'):
                        for feature, changes in cf_data['features_changed'].items():
                            st.markdown(f"- **{feature}**: from `{changes['original_value']:.2f}` to `{changes['counterfactual_value']:.2f}`")
                    else:
                        st.markdown("No significant features changed or no counterfactual found that flips the prediction within constraints.")
                else:
                    st.warning("No counterfactual instance was generated by DiCE for this selection. This may happen if a flip is not possible within the feature ranges or with current settings.")
            else:
                st.info("No counterfactual data available for display.")
        else:
            st.info("Select a denied instance and click 'Generate Counterfactuals' to see results.")
    else:
        st.warning("Please upload and load model/data in 'Setup & Data' first to enable counterfactual generation.")

    st.markdown(f"")
    st.markdown(f"### c. Explanation of Execution")
    st.markdown(f"")
    st.markdown(f"The counterfactual analysis provides invaluable \"what-if\" scenarios for PrimeCredit Bank. For the selected denied loan application (`instance_ID`), the `counterfactual_result` clearly shows that increasing the `credit_score` by a certain amount or raising the `income` significantly, for example, would have resulted in the loan being approved. The `features_changed` dictionary pinpoints the minimal adjustments needed.")
    st.markdown(f"")
    st.markdown(f"This empowers Anya to:")
    st.markdown(f"1.  **Inform customers:** Instead of just saying \"your loan was denied,\" PrimeCredit can advise applicants on specific, actionable steps (e.g., \"If your credit score improved by X points, you would likely be approved\").")
    st.markdown(f"2.  **Refine policy:** If generating counterfactuals consistently highlights specific features as critical for flipping decisions, it might indicate areas for policy review or for developing financial literacy programs for customers.")
    st.markdown(f"3.  **Assess model sensitivity:** It reveals how sensitive the model is to changes in specific features, which is a key part of model validation.")
    st.markdown(f"")
    st.markdown(f"This concrete evidence of actionable insights is crucial for establishing trust and demonstrating the model's utility beyond just making a prediction.")

# 6. Summary & Audit Page
elif st.session_state.page == 'Summary & Audit':
    st.title("6. Identifying Gaps: Interpretability Analysis and Validation Findings")
    st.markdown(f"")
    st.markdown(f"### a. Story + Context + Real-World Relevance")
    st.markdown(f"")
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

    if st.session_state.global_importance_df is not None and st.session_state.local_explanations_data is not None and st.session_state.counterfactual_result is not None and st.session_state.model_hash is not None and st.session_state.data_hash is not None and st.session_state.explanation_dir is not None:
        if st.button("Generate Explanation Summary Report", key='generate_summary_button'):
            with st.spinner("Generating summary report..."):
                try:
                    # Call generate_explanation_summary from source.py
                    # Note: The source.py function uses global model_hash_val/data_hash_val if inside source, 
                    # but here we pass artifacts or it might use session state. 
                    # The prompt snippet signature: generate_explanation_summary(global_df, local_data, cf_result, dir)
                    generate_explanation_summary(
                        st.session_state.global_importance_df, st.session_state.local_explanations_data,
                        st.session_state.counterfactual_result, st.session_state.explanation_dir
                    )
                    st.success("Summary report generated successfully! Scroll down to view.")
                except Exception as e:
                    st.error(f"Error generating summary report: {e}")
                    st.exception(e)
        
        summary_path = os.path.join(st.session_state.explanation_dir, 'explanation_summary.md')
        if os.path.exists(summary_path):
            st.subheader("Generated Explanation Summary:")
            with open(summary_path, 'r') as f:
                summary_content = f.read()
            st.markdown(summary_content) # Display the markdown content
        else:
            st.info("Click 'Generate Explanation Summary Report' to create the report. Ensure all previous explanation steps are completed.")
    else:
        st.warning("Please ensure model/data are loaded in 'Setup & Data' and all explanation types (global, local, counterfactual) have been generated, to create the summary report.")

    st.markdown(f"---")
    st.title("7. Audit Trail: Reproducibility and Artifact Bundling")
    st.markdown(f"")
    st.markdown(f"### a. Story + Context + Real-World Relevance")
    st.markdown(f"")
    st.markdown(f"The final, critical step for Anya is to ensure that all her validation work is reproducible and securely bundled for auditing purposes. For PrimeCredit Bank, regulatory compliance demands an immutable record of all explanation artifacts, along with the configuration and hashes that guarantee their traceability to specific model and data versions. This \"audit-ready artifact bundle\" acts as indisputable evidence of the model validation process. I will consolidate all generated explanations, configuration details, and an `evidence_manifest.json` containing SHA-256 hashes of each file, into a single, timestamped ZIP archive.")
    st.markdown(f"")
    st.markdown(r"$$ \text{{SHA-256}}(\text{{file\_content}}) = \text{{hexadecimal\_hash\_string}} $$")
    st.markdown(r"where $\text{{SHA-256}}$ is the cryptographic hash function, $\text{{file\_content}}$ is the input data (e.g., content of a file), and $\text{{hexadecimal\_hash\_string}}$ is the unique 256-bit hexadecimal output.")
    st.markdown(f"")

    if st.session_state.model_hash is not None and st.session_state.data_hash is not None and st.session_state.explanation_dir is not None and st.session_state.run_id is not None:
        if st.button("Export All Audit-Ready Artifacts", key='export_artifacts_button'):
            with st.spinner("Bundling all artifacts into a ZIP file..."):
                try:
                    # 1. Create configuration snapshot
                    config_file_path = create_config_snapshot(
                        st.session_state.model_hash, st.session_state.data_hash, RANDOM_SEED, st.session_state.explanation_dir
                    )

                    # 2. Define all output files that need to be hashed and bundled
                    output_files_to_bundle = [
                        os.path.join(st.session_state.explanation_dir, 'global_explanation.json'),
                        os.path.join(st.session_state.explanation_dir, 'local_explanation.json'),
                        os.path.join(st.session_state.explanation_dir, 'counterfactual_example.json'),
                        os.path.join(st.session_state.explanation_dir, 'explanation_summary.md'),
                        config_file_path
                    ]
                    # Filter out files that might not have been created (e.g., if CF generation was skipped)
                    output_files_to_bundle = [f for f in output_files_to_bundle if os.path.exists(f)]

                    # 3. Create evidence manifest
                    manifest_file_path = create_evidence_manifest(st.session_state.explanation_dir, output_files_to_bundle)
                    output_files_to_bundle.append(manifest_file_path) # Add manifest itself to the bundle

                    # 4. Bundle all artifacts into a zip file
                    zip_archive_path = bundle_artifacts_to_zip(st.session_state.explanation_dir, st.session_state.run_id)
                    st.session_state.zip_archive_path = zip_archive_path
                    st.success("Audit-ready artifacts bundled successfully!")
                except Exception as e:
                    st.error(f"Error exporting artifacts: {e}")
                    st.exception(e)
        
        if st.session_state.zip_archive_path and os.path.exists(st.session_state.zip_archive_path):
            st.subheader("Download Audit Bundle:")
            with open(st.session_state.zip_archive_path, "rb") as fp:
                st.download_button(
                    label="Download Audit ZIP File",
                    data=fp.read(),
                    file_name=os.path.basename(st.session_state.zip_archive_path),
                    mime="application/zip"
                )
            st.markdown(f"Your audit-ready archive is available at: `{st.session_state.zip_archive_path}`")
        else:
            st.info("Click 'Export All Audit-Ready Artifacts' to generate and download the bundle. Ensure all previous explanation steps are completed.")
    else:
        st.warning("Please ensure model and data are loaded in 'Setup & Data' first, and all explanation types (global, local, counterfactual) have been generated, to enable artifact bundling.")

    st.markdown(f"")
    st.markdown(f"### c. Explanation of Execution")
    st.markdown(f"")
    st.markdown(f"The final stage of the validation workflow is complete. I have successfully generated a comprehensive set of explanation artifacts, including global and local SHAP analyses, counterfactual examples, and my detailed summary report. Each of these documents, along with a snapshot of the configuration (including model and data hashes) and a manifest of all files with their individual SHA-256 hashes, has been meticulously bundled into a timestamped ZIP archive: `Session_05_<run_id>.zip`.")
    st.markdown(f"")
    st.markdown(f"This single, self-contained archive is PrimeCredit Bank's **audit-ready artifact bundle**. It ensures:")
    st.markdown(f"1.  **Reproducibility:** The `config_snapshot.json` captures all parameters needed to regenerate these explanations.")
    st.markdown(f"2.  **Traceability:** The `evidence_manifest.json` provides cryptographic proof of the integrity and origin of each artifact, linking them directly to the validated model and data versions.")
    st.markdown(f"3.  **Compliance:** All necessary documentation for internal auditors, regulators, and senior stakeholders is readily available and verifiable, significantly reducing regulatory risk and building trust in the AI system.")
    st.markdown(f"")
    st.markdown(f"This completes my model validation task for CAM v1.2, providing PrimeCredit Bank with the necessary confidence to proceed with its deployment, knowing its decisions are explainable, transparent, and auditable.")


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
