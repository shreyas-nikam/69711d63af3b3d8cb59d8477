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
    generate_sample_data_and_model, generate_healthcare_data_and_model, generate_fraud_data_and_model,
    initialize_model_and_data,
    generate_global_shap_explanation, generate_local_shap_explanations,
    generate_counterfactual_explanation, generate_explanation_summary,
    create_config_snapshot, create_evidence_manifest, bundle_artifacts_to_zip,
    select_instances_for_explanation, select_denied_instance_for_cf
)

# Set page config
st.set_page_config(
    page_title="QuLab: Lab 5: Model Interpretability & Explainability Workbench", layout="wide")

# QuLab Header
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 5: Model Interpretability & Explainability Workbench")
st.divider()

# --- Initialization & Helper Functions ---

# Global initialization to ensure sample data exists for demonstration/testing.
if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    # Using print logic as per spec, usually st.write/st.info is better but following spec structure
    print(
        f"Sample model/data not found, generating to {MODEL_PATH} and {DATA_PATH}...")
    try:
        generate_sample_data_and_model(
            MODEL_PATH, DATA_PATH, TARGET_COLUMN, RANDOM_SEED)
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
    os.makedirs(base_dir, exist_ok=True)  # Ensure the base directory exists

# Helper function to display SHAP plots


def display_shap_plot(plot_func, *args, **kwargs):
    """Generates and displays a SHAP plot in Streamlit, ensuring new figures are created and closed."""
    fig = plt.figure()  # Create a new figure
    plot_func(*args, **kwargs, show=False)  # Call SHAP plot function
    st.pyplot(fig)  # Display the figure in Streamlit
    # Close the figure to prevent display issues in subsequent renders
    plt.close(fig)

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
if 'use_case' not in st.session_state:
    st.session_state.use_case = None
if 'positive_class' not in st.session_state:
    st.session_state.positive_class = 1

# --- Navigation ---

with st.sidebar:
    st.title("Navigation")
    page_selection = st.selectbox(
        "Go to",
        ['Welcome', 'Setup & Data', 'Global Explanations',
         'Local Explanations', 'Counterfactuals', 'Summary & Audit'],
        index=['Welcome', 'Setup & Data', 'Global Explanations',
               'Local Explanations', 'Counterfactuals', 'Summary & Audit'].index(st.session_state.page)
    )
    if page_selection:
        st.session_state.page = page_selection

# --- Main Content Area ---

# 1. Welcome Page
if st.session_state.page == 'Welcome':
    st.markdown("### Introduction")
    st.markdown("")
    st.markdown("This workbench provides comprehensive model explainability and interpretability analysis for machine learning models across multiple domains. As organizations increasingly rely on AI/ML models for critical decision-making, understanding model behavior becomes paramount for:")
    st.markdown("""
- **Regulatory Compliance**: Meeting requirements from regulators and auditors
- **Risk Management**: Identifying potential biases and model weaknesses
- **Stakeholder Trust**: Providing transparent, defensible explanations
- **Model Validation**: Ensuring model decisions align with domain knowledge

### Explanation Methods

This workbench generates audit-ready artifacts using:    
- **Global Explanations**: SHAP-based feature importance across all predictions    
- **Local Explanations**: Instance-specific feature contributions    
- **Counterfactual Explanations**: Minimal changes needed to flip predictions    
- **Comprehensive Documentation**: SHA-256 hashed artifacts for traceability
""")

# 2. Setup & Data Page
elif st.session_state.page == 'Setup & Data':
    st.title("2. Setup & Data Ingestion")
    st.markdown("")
    st.markdown(
        "Load pre-configured sample data and trained model for analysis.")
    st.markdown("")

    st.subheader("Load Pre-configured Sample Dataset")
    st.markdown(
        "Select a use case to load sample data and trained model for demonstration.")

    use_case_selection = st.radio(
        "Choose a use case:",
        options=[
            "Use Case A — Credit Approval Model",
            "Use Case B — Healthcare Risk Scoring Model",
            "Use Case C — Fraud Detection Model"
        ],
        key='setup_use_case_selector'
    )

    # Map use case to files and target column
    use_case_config = {
        "Use Case A — Credit Approval Model": {
            "model_path": "sample_credit_model.pkl",
            "data_path": "sample_credit_data.csv",
            "target_col": "loan_approved",
            "generator": generate_sample_data_and_model
        },
        "Use Case B — Healthcare Risk Scoring Model": {
            "model_path": "sample_healthcare_model.pkl",
            "data_path": "sample_healthcare_data.csv",
            "target_col": "high_risk",
            "generator": generate_healthcare_data_and_model
        },
        "Use Case C — Fraud Detection Model": {
            "model_path": "sample_fraud_model.pkl",
            "data_path": "sample_fraud_data.csv",
            "target_col": "is_fraud",
            "generator": generate_fraud_data_and_model
        }
    }

    config = use_case_config[use_case_selection]

    if st.button("Load Sample Dataset", key='load_sample_button'):
        with st.spinner(f"Loading {use_case_selection}..."):
            try:
                # Generate sample data if it doesn't exist
                config['generator'](
                    config['model_path'],
                    config['data_path'],
                    config['target_col'],
                    RANDOM_SEED
                )

                # Clear previous session run artifacts
                clear_explanation_directory("reports")

                # Generate a new unique run ID and directory for this session
                st.session_state.run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                st.session_state.explanation_dir = f'reports/session_05_validation_run_{st.session_state.run_id}'
                os.makedirs(st.session_state.explanation_dir,
                            exist_ok=True)

                # Load artifacts
                model, full_data, X, y, model_hash_val, data_hash_val = load_and_hash_artifacts(
                    config['model_path'],
                    config['data_path'],
                    config['target_col'],
                    RANDOM_SEED
                )

                # Store in session state
                st.session_state.model = model
                st.session_state.data = full_data
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.model_hash = model_hash_val
                st.session_state.data_hash = data_hash_val
                st.session_state.target_column = config['target_col']
                st.session_state.use_case = use_case_selection

                # Set positive class based on use case
                if use_case_selection == "Use Case A — Credit Approval Model":
                    st.session_state.positive_class = 1
                else:
                    st.session_state.positive_class = 0

                # Prepare data subsets for SHAP explainer
                st.session_state.X_train_exp, st.session_state.X_test_exp, _, _ = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_SEED
                )

                # Initialize SHAP TreeExplainer
                st.session_state.explainer = shap.TreeExplainer(model)
                st.success(f"{use_case_selection} loaded successfully!")

            except Exception as e:
                st.error(f"Error loading sample data: {e}")
                st.exception(e)

    # Display summary of loaded artifacts if available in session state
    if st.session_state.model is not None and st.session_state.data is not None:
        st.markdown("---")
        st.subheader("Loaded Artifacts Summary:")
        if st.session_state.use_case:
            st.markdown(f"**Use Case:** `{st.session_state.use_case}`")
        st.markdown(
            f"**Model Type:** `{type(st.session_state.model).__name__}`")
        st.markdown(
            f"**Model Hash (SHA-256):** `{st.session_state.model_hash}`")
        st.markdown(f"**Data Hash (SHA-256):** `{st.session_state.data_hash}`")
        st.markdown(
            f"**Features:** `{', '.join(st.session_state.X.columns.tolist())}`")
        st.markdown(f"**Target Column:** `{st.session_state.target_column}`")
        st.markdown(
            f"**Dataset Size:** `{len(st.session_state.data)} samples`")

# 3. Global Explanations Page
elif st.session_state.page == 'Global Explanations':
    st.title("3. Global Model Explanations")
    st.markdown("")
    st.markdown(
        "Generate global SHAP explanations to understand overall feature importance across all predictions.")
    st.markdown("")

    if st.session_state.model is not None and st.session_state.X_train_exp is not None and st.session_state.explainer is not None and st.session_state.explanation_dir is not None:
        if st.button("Generate Global Explanations", key='generate_global_exp_button'):
            with st.spinner("Generating global SHAP explanations..."):
                try:
                    # Call generate_global_shap_explanation from source.py
                    global_importance_df, global_shap_values, explainer = generate_global_shap_explanation(
                        st.session_state.model, st.session_state.X_train_exp, st.session_state.X_train_exp.columns.tolist(),
                        st.session_state.explanation_dir, st.session_state.X_test_exp, RANDOM_SEED, st.session_state.positive_class
                    )
                    # Update session state with results
                    st.session_state.global_importance_df = global_importance_df
                    st.session_state.global_shap_values = global_shap_values
                    st.session_state.explainer = explainer
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

                shap_vals = st.session_state.global_shap_values
                data_for_plot = st.session_state.X_test_exp

                # SHAP summary_plot can handle sampling internally, no need for shape check
                display_shap_plot(shap.summary_plot, shap_vals,
                                  data_for_plot, plot_type="bar")
        else:
            st.info("Click 'Generate Global Explanations' to see results.")
    else:
        st.warning(
            "Please upload and load model/data in 'Setup & Data' first to enable global explanations.")

# 4. Local Explanations Page
elif st.session_state.page == 'Local Explanations':
    st.title("4. Local Explanations")
    st.markdown("")
    st.markdown(
        "Generate local SHAP explanations to understand individual predictions and feature contributions.")
    st.markdown("")

    if st.session_state.model is not None and st.session_state.X is not None and st.session_state.y is not None and st.session_state.explainer is not None and st.session_state.explanation_dir is not None:
        st.subheader("Select Instances for Local Explanation")

        # Pre-select some interesting indices for convenience
        denied_indices = st.session_state.y[st.session_state.y == 0].index.tolist(
        )
        approved_indices = st.session_state.y[st.session_state.y == 1].index.tolist(
        )

        default_indices = []
        if denied_indices:
            default_indices.append(denied_indices[0])  # First denied example
        if approved_indices:
            # First approved example (different from denied)
            for idx in approved_indices:
                if idx not in default_indices:
                    default_indices.append(idx)
                    break
        # Add a borderline example if possible
        if len(st.session_state.X) > 0 and st.session_state.model is not None and hasattr(st.session_state.model, 'classes_') and 1 in st.session_state.model.classes_:
            probabilities = st.session_state.model.predict_proba(
                st.session_state.X)[:, np.where(st.session_state.model.classes_ == 1)[0][0]]
            borderline_idx_pos_in_X = np.argmin(np.abs(probabilities - 0.5))
            borderline_idx = st.session_state.X.index[borderline_idx_pos_in_X]
            if borderline_idx not in default_indices:
                default_indices.append(borderline_idx)

        selected_indices_for_local_exp = st.multiselect(
            "Choose instance indices from the dataset for detailed explanation:",
            options=st.session_state.X.index.tolist(),
            # Show up to 3 example indices by default
            default=default_indices[:3],
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
                            st.session_state.explainer, st.session_state.explanation_dir, st.session_state.positive_class
                        )
                        # Update session state with results
                        st.session_state.local_explanations_data = local_explanations_data
                        st.session_state.local_shap_explanations_list = shap_values_list

                        # Store one denied instance for convenient pre-selection on the Counterfactuals page
                        denied_for_cf = next(
                            (idx for idx in selected_indices_for_local_exp if st.session_state.y.loc[idx] == 0), None)
                        if denied_for_cf is None and denied_indices:  # Fallback to first denied if none in selected_indices
                            denied_for_cf = denied_indices[0]
                        st.session_state.denied_instance_for_cf_idx = denied_for_cf

                        st.success(
                            "Local explanations generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating local explanations: {e}")
                        st.exception(e)

        if st.session_state.local_explanations_data:
            st.subheader("Local Explanation Results:")
            for i, (inst_key, data) in enumerate(st.session_state.local_explanations_data.items()):
                original_idx = int(inst_key.split('_')[1])
                st.markdown(f"#### Instance ID: {original_idx}")
                st.markdown(
                    f"**Original Features:** `{data['original_features']}`")
                st.markdown(
                    f"**Model Predicted Approval Probability:** `{data['model_prediction']:.4f}`")
                st.markdown(f"**Top SHAP Contributions:**")
                sorted_shap = sorted(data['shap_values'].items(
                ), key=lambda item: abs(item[1]), reverse=True)
                for feature, value in sorted_shap[:5]:
                    st.markdown(f"- `{feature}`: `{value:.4f}`")

                # Display SHAP waterfall plot for the instance
                if i < len(st.session_state.local_shap_explanations_list):
                    st.markdown("##### Visualization: SHAP Waterfall Plot")
                    display_shap_plot(
                        shap.waterfall_plot, st.session_state.local_shap_explanations_list[i], max_display=10)
                st.markdown("---")
        else:
            st.info(
                "Select instances and click 'Generate Local Explanations' to see results.")
    else:
        st.warning(
            "Please upload and load model/data in 'Setup & Data' first to enable local explanations.")

# 5. Counterfactuals Page
elif st.session_state.page == 'Counterfactuals':
    st.title("5. Counterfactual Explanations")
    st.markdown("")
    st.markdown(
        "Generate counterfactual explanations to identify minimal changes needed to flip predictions.")
    st.markdown("")

    if st.session_state.model is not None and st.session_state.X is not None and st.session_state.data is not None and st.session_state.explanation_dir is not None:
        st.subheader("Generate Counterfactual Explanation")

        # Get available denied instances for selection
        denied_indices_options = st.session_state.y[st.session_state.y == 0].index.tolist(
        )

        # Pre-select a denied instance if available (from local explanations page or first available)
        default_cf_idx = st.session_state.denied_instance_for_cf_idx
        if default_cf_idx is None and denied_indices_options:
            default_cf_idx = denied_indices_options[0]

        if denied_indices_options:  # Only show selector if denied instances exist
            selected_cf_instance_idx = st.selectbox(
                "Select a denied instance (where original `loan_approved` == 0) for counterfactual analysis:",
                options=denied_indices_options,
                # Set default index if default_cf_idx is in options, else 0 or None
                index=denied_indices_options.index(default_cf_idx) if default_cf_idx in denied_indices_options else (
                    0 if denied_indices_options else None),
                format_func=lambda x: f"Instance {x}",
                key='cf_instance_selector'
            )
            # Set desired class options based on use case
            if st.session_state.use_case == 'Credit Approval':
                cf_options = [1, 0]

                def cf_format_func(
                    x): return "Approval (1)" if x == 1 else "Denial (0)"
                cf_index = 0  # Default to approval
            elif st.session_state.use_case == 'Healthcare Risk Scoring':
                cf_options = [0, 1]

                def cf_format_func(
                    x): return "Low Risk (0)" if x == 0 else "High Risk (1)"
                cf_index = 0  # Default to low risk
            elif st.session_state.use_case == 'Fraud Detection':
                cf_options = [0, 1]
                def cf_format_func(
                    x): return "Not Fraud (0)" if x == 0 else "Fraud (1)"
                cf_index = 0  # Default to not fraud
            else:
                cf_options = [1, 0]
                def cf_format_func(x): return f"Class {x}"
                cf_index = 0

            desired_class = st.radio("Desired Outcome:", options=cf_options,
                                     index=cf_index, format_func=cf_format_func, key='cf_desired_class')

            if st.button("Generate Counterfactuals", key='generate_cf_button'):
                if selected_cf_instance_idx is None:
                    st.warning(
                        "Please select an instance to generate counterfactuals.")
                else:
                    with st.spinner(f"Generating counterfactual explanation for instance {selected_cf_instance_idx}..."):
                        try:
                            # Call generate_counterfactual_explanation from source.py
                            counterfactual_result = generate_counterfactual_explanation(
                                st.session_state.model, st.session_state.X, st.session_state.data,
                                st.session_state.X.columns.tolist(), selected_cf_instance_idx, desired_class,
                                st.session_state.explanation_dir, st.session_state.target_column or TARGET_COLUMN
                            )
                            # Update session state with results
                            st.session_state.counterfactual_result = counterfactual_result
                            st.success(
                                "Counterfactuals generated successfully!")

                            # Show the counterfactual result
                            st.json(counterfactual_result)
                        except Exception as e:
                            st.error(
                                f"Error generating counterfactuals: {e}. DiCE might struggle to find a counterfactual given the data/constraints.")
                            st.exception(e)
        else:
            st.info("No denied instances found in the dataset to generate counterfactuals. Please ensure your dataset contains instances with `loan_approved == 0`.")

        if st.session_state.counterfactual_result:
            st.subheader("Counterfactual Explanation Results:")
            cf_data = st.session_state.counterfactual_result
            if cf_data.get('original_instance'):
                st.markdown(
                    f"**Original Instance (ID {selected_cf_instance_idx}):**")
                st.json(cf_data['original_instance'])
                st.markdown(
                    f"**Original Prediction Probability (Desired Class {desired_class}):** `{cf_data.get('original_prediction_prob_desired_class', 0.0):.4f}`")

                if cf_data.get('counterfactual_instance'):
                    st.markdown(f"**Counterfactual Instance:**")
                    st.json(cf_data['counterfactual_instance'])
                    st.markdown(
                        f"**Counterfactual Prediction Probability (Desired Class {desired_class}):** `{cf_data.get('counterfactual_prediction_prob_desired_class', 0.0):.4f}`")
                    st.markdown(
                        f"**Minimal Feature Changes to Flip Prediction:**")
                    if cf_data.get('features_changed'):
                        for feature, changes in cf_data['features_changed'].items():
                            st.markdown(
                                f"- **{feature}**: from `{changes['original_value']:.2f}` to `{changes['counterfactual_value']:.2f}`")
                    else:
                        st.markdown(
                            "No significant features changed or no counterfactual found that flips the prediction within constraints.")
                else:
                    st.warning(
                        "No counterfactual instance was generated by DiCE for this selection. This may happen if a flip is not possible within the feature ranges or with current settings.")
            else:
                st.info("No counterfactual data available for display.")
        else:
            st.info(
                "Select a denied instance and click 'Generate Counterfactuals' to see results.")
    else:
        st.warning(
            "Please upload and load model/data in 'Setup & Data' first to enable counterfactual generation.")

# 6. Summary & Audit Page
elif st.session_state.page == 'Summary & Audit':
    st.title("6. Summary & Audit")
    st.markdown("")
    st.markdown(
        "Generate summary report and bundle all artifacts for audit purposes.")
    st.markdown("")

    if st.session_state.global_importance_df is not None and st.session_state.local_explanations_data is not None and st.session_state.counterfactual_result is not None and st.session_state.model_hash is not None and st.session_state.data_hash is not None and st.session_state.explanation_dir is not None:
        if st.button("Generate Explanation Summary Report", key='generate_summary_button'):
            with st.spinner("Generating summary report..."):
                try:
                    # Call generate_explanation_summary from source.py
                    generate_explanation_summary(
                        st.session_state.global_importance_df, st.session_state.local_explanations_data,
                        st.session_state.counterfactual_result, st.session_state.model_hash,
                        st.session_state.data_hash, st.session_state.explanation_dir
                    )
                    st.success(
                        "Summary report generated successfully! Scroll down to view.")
                except Exception as e:
                    st.error(f"Error generating summary report: {e}")
                    st.exception(e)

        summary_path = os.path.join(
            st.session_state.explanation_dir, 'explanation_summary.md')
        if os.path.exists(summary_path):
            st.subheader("Generated Explanation Summary:")
            with open(summary_path, 'r') as f:
                summary_content = f.read()
            st.markdown(summary_content)  # Display the markdown content
        else:
            st.info("Click 'Generate Explanation Summary Report' to create the report. Ensure all previous explanation steps are completed.")
    else:
        st.warning("Please ensure model/data are loaded in 'Setup & Data' and all explanation types (global, local, counterfactual) have been generated, to create the summary report.")

    st.markdown("---")
    st.subheader("Audit Trail & Artifact Bundling")

    if st.session_state.model_hash is not None and st.session_state.data_hash is not None and st.session_state.explanation_dir is not None and st.session_state.run_id is not None:
        if st.button("Export All Audit-Ready Artifacts", key='export_artifacts_button'):
            with st.spinner("Bundling all artifacts into a ZIP file..."):
                try:
                    # 1. Create configuration snapshot
                    config_file_path = create_config_snapshot(
                        st.session_state.model_hash, st.session_state.data_hash, RANDOM_SEED,
                        st.session_state.explanation_dir, MODEL_PATH, DATA_PATH,
                        st.session_state.target_column or TARGET_COLUMN
                    )

                    # 2. Define all output files that need to be hashed and bundled
                    output_files_to_bundle = [
                        os.path.join(st.session_state.explanation_dir,
                                     'global_explanation.json'),
                        os.path.join(st.session_state.explanation_dir,
                                     'local_explanation.json'),
                        os.path.join(st.session_state.explanation_dir,
                                     'counterfactual_example.json'),
                        os.path.join(st.session_state.explanation_dir,
                                     'explanation_summary.md'),
                        config_file_path
                    ]
                    # Filter out files that might not have been created (e.g., if CF generation was skipped)
                    output_files_to_bundle = [
                        f for f in output_files_to_bundle if os.path.exists(f)]

                    # 3. Create evidence manifest
                    manifest_file_path = create_evidence_manifest(
                        st.session_state.explanation_dir, output_files_to_bundle)
                    # Add manifest itself to the bundle
                    output_files_to_bundle.append(manifest_file_path)

                    # 4. Bundle all artifacts into a zip file
                    zip_archive_path = bundle_artifacts_to_zip(
                        st.session_state.explanation_dir, st.session_state.run_id)
                    st.session_state.zip_archive_path = zip_archive_path
                    st.success("Audit-ready artifacts bundled successfully!")
                except Exception as e:
                    st.error(f"Error exporting artifacts: {e}")
                    st.exception(e)

        if st.session_state.zip_archive_path and os.path.exists(st.session_state.zip_archive_path):
            st.subheader("Download Audit Bundle:")
            with open(st.session_state.zip_archive_path, "rb") as fp:
                zip_data = fp.read()

            if st.download_button(
                label="Download Audit ZIP File",
                data=zip_data,
                file_name=os.path.basename(st.session_state.zip_archive_path),
                mime="application/zip",
                key="download_zip_button",
                on_click=lambda: None
            ):
                # Clean up session-specific files
                import shutil
                try:
                    # Delete uploaded files directory
                    if 'session_id' in st.session_state:
                        uploaded_dir = f"uploaded_files_temp/session_{st.session_state.session_id}"
                        if os.path.exists(uploaded_dir):
                            shutil.rmtree(uploaded_dir)

                    # Delete explanation directory and zip file
                    if st.session_state.explanation_dir and os.path.exists(st.session_state.explanation_dir):
                        shutil.rmtree(os.path.dirname(
                            st.session_state.explanation_dir))

                    # Clear all session state variables
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]

                    st.success(
                        "✅ Your session has been cleared. Please start again from 'Setup & Data' page.")
                    st.info(
                        "💡 Tip: Navigate to 'Setup & Data' to begin a new validation session.")
                except Exception as e:
                    st.warning(
                        f"Session cleared, but cleanup encountered an issue: {e}")
        else:
            st.info("Click 'Export All Audit-Ready Artifacts' to generate and download the bundle. Ensure all previous explanation steps are completed.")
    else:
        st.warning("Please ensure model and data are loaded in 'Setup & Data' first, and all explanation types (global, local, counterfactual) have been generated, to enable artifact bundling.")


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
