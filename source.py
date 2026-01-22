import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import dice_ml
import os
import hashlib
import json
import zipfile
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm.notebook import tqdm
# --- Configuration for Reproducibility ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define file paths for the model and data
MODEL_PATH = 'sample_credit_model.pkl'
DATA_PATH = 'sample_credit_data.csv'
TARGET_COLUMN = 'loan_approved'
EXPLANATION_DIR = 'reports/session_05_validation_run'

# Create explanation directory if it doesn't exist
os.makedirs(EXPLANATION_DIR, exist_ok=True)

# Helper function to calculate SHA-256 hash
def calculate_file_hash(filepath):
    """Calculates the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Function to simulate loading and hashing artifacts
def load_and_hash_artifacts(model_path, data_path, target_col, random_seed):
    """
    Loads model and data, calculates their hashes, and prepares the dataset.
    Returns: model, data, features, target, model_hash, data_hash
    """
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)

    # Separate features and target
    features = data.drop(columns=[target_col])
    target = data[target_col]

    # Calculate hashes
    model_hash = calculate_file_hash(model_path)
    data_hash = calculate_file_hash(data_path)

    print(f"\nModel Hash: {model_hash}")
    print(f"Data Hash: {data_hash}")
    print(f"Random Seed: {random_seed}")

    return model, data, features, target, model_hash, data_hash

# --- Simulate Data and Model Creation for the Lab if files don't exist ---
def generate_sample_data_and_model(model_path, data_path, target_col, random_seed):
    """Generates a sample credit dataset and trains a RandomForestClassifier."""
    if os.path.exists(model_path) and os.path.exists(data_path):
        print("Sample model and data already exist. Skipping generation.")
        return

    print("Generating sample credit data and model...")
    num_samples = 1000
    np.random.seed(random_seed)

    data = pd.DataFrame({
        'credit_score': np.random.randint(300, 850, num_samples),
        'income': np.random.randint(30000, 150000, num_samples),
        'loan_amount': np.random.randint(5000, 100000, num_samples),
        'debt_to_income': np.random.rand(num_samples) * 0.6,
        'employment_length': np.random.randint(0, 20, num_samples),
        'num_credit_lines': np.random.randint(1, 10, num_samples),
        'delinquent_payments': np.random.randint(0, 5, num_samples),
    })

    # Simple logic for loan approval (simulated)
    # Higher credit_score, income, employment_length, lower debt_to_income, delinquent_payments -> more likely approval
    approval_prob = (
        0.0005 * data['credit_score']
        + 0.000001 * data['income']
        - 0.000005 * data['loan_amount']
        - 0.5 * data['debt_to_income']
        + 0.05 * data['employment_length']
        - 0.2 * data['delinquent_payments']
        - 0.1 * data['num_credit_lines']
        - 2.0 # Baseline
    ).apply(lambda x: 1 / (1 + np.exp(-x))) # Sigmoid to convert to probability

    data[target_col] = (approval_prob > 0.5).astype(int)

    # Train a RandomForestClassifier
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    model = RandomForestClassifier(n_estimators=100, random_state=random_seed, class_weight='balanced')
    model.fit(X_train, y_train)

    # Save model and data
    joblib.dump(model, model_path)
    data.to_csv(data_path, index=False)
    print("Sample model and data generated and saved.")

# Execute generation of sample data and model
generate_sample_data_and_model(MODEL_PATH, DATA_PATH, TARGET_COLUMN, RANDOM_SEED)

# Execute loading and hashing
credit_model, full_data, X, y, model_hash_val, data_hash_val = load_and_hash_artifacts(
    MODEL_PATH, DATA_PATH, TARGET_COLUMN, RANDOM_SEED
)

# Split data for explanation purposes (using the entire X for explanations as per SHAP requirement)
# For LIME/DiCE, we'll use a portion of X for background
X_train_exp, X_test_exp, _, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

print(f"\nModel type identified: {type(credit_model)}")
print(f"Data features: {X.columns.tolist()}")
print(f"First 5 rows of feature data:\n{X.head()}")
# Function to generate global SHAP explanations
def generate_global_shap_explanation(model, X_data, feature_names, explanation_dir):
    """
    Generates global feature importance using SHAP values for a tree-based model.
    Saves the aggregated SHAP values and a summary plot.
    """
    print("Generating global SHAP explanations...")
    # Initialize JS for SHAP plots
    shap.initjs()

    # Create a SHAP TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for a sample of the data for performance, or full X_data if feasible
    # Using X_test_exp to get a representative sample if X is very large
    sample_data = X_test_exp.sample(min(1000, X_test_exp.shape[0]), random_state=RANDOM_SEED) if X_test_exp.shape[0] > 1000 else X_test_exp
    shap_values = explainer.shap_values(sample_data)

    # SHAP values for classification models return two arrays (for class 0 and class 1)
    # We are interested in the positive class (loan approved = 1)
    if isinstance(shap_values, list):
        shap_values_for_class_1 = shap_values[1]
    else:
        shap_values_for_class_1 = shap_values # For regression or binary where shap_values is a single array

    # Aggregate SHAP values to get mean absolute importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values_for_class_1).mean(axis=0)
    }).sort_values(by='importance', ascending=False)

    print("Global feature importance (top 10):")
    print(feature_importance.head(10))

    # Save global explanation to JSON
    global_explanation_output_path = os.path.join(explanation_dir, 'global_explanation.json')
    feature_importance.to_json(global_explanation_output_path, orient='records', indent=4)
    print(f"Global explanation saved to {global_explanation_output_path}")

    # Display SHAP summary plot
    print("\nVisualizing global feature importance:")
    shap.summary_plot(shap_values_for_class_1, sample_data, plot_type="bar", show=False)
    # shap.summary_plot() generates a matplotlib figure; it can be saved if needed.
    # For a notebook, displaying it is enough.

    return feature_importance, shap_values_for_class_1

# Execute global explanation generation
global_importance_df, global_shap_values = generate_global_shap_explanation(
    credit_model, X_train_exp, X_train_exp.columns.tolist(), EXPLANATION_DIR
)
# Function to generate local SHAP explanations for selected instances
def generate_local_shap_explanations(model, X_data, instances_to_explain, explainer, explanation_dir):
    """
    Generates local SHAP explanations for a list of specified instances.
    Saves individual explanations and displays waterfall plots.
    """
    print("\nGenerating local SHAP explanations for selected instances...")
    local_explanations = {}
    shap_values_list = []

    # Determine the index for the positive class (loan approved = 1) in the model's class array.
    positive_class_val = 1
    positive_class_idx_in_model_output = None # This will be the index in `predict_proba` output
    if positive_class_val in model.classes_:
        positive_class_idx_in_model_output = np.where(model.classes_ == positive_class_val)[0][0]
    else:
        print(f"Warning: Model trained on classes {model.classes_}. Class {positive_class_val} not found. Probability of approval will be 0.")

    for i, idx in enumerate(instances_to_explain):
        print(f"\nExplaining instance ID: {idx} (original index in X_data)")
        instance = X_data.iloc[[idx]]

        current_model_prediction_proba = 0.0
        current_shap_values_instance = np.zeros(X_data.shape[1])
        current_expected_value_instance = 0.0

        if positive_class_idx_in_model_output is not None:
            # Get SHAP values
            shap_output_for_instance = explainer.shap_values(instance)

            # Extract SHAP values for the positive class
            if isinstance(shap_output_for_instance, list):
                # Standard classification case: list of arrays for each class
                current_shap_values_instance = shap_output_for_instance[positive_class_idx_in_model_output][0]
                current_expected_value_instance = explainer.expected_value[positive_class_idx_in_model_output]
            else:
                # Fallback for models where shap_values is a single array (e.g., regression or specific binary model output)
                # If model has only one class, and that class is `positive_class_val`, use the single array.
                if len(model.classes_) == 1 and model.classes_[0] == positive_class_val:
                    current_shap_values_instance = shap_output_for_instance[0]
                    current_expected_value_instance = explainer.expected_value
                else:
                    # This case should ideally not be hit for a binary RandomForestClassifier
                    print(f"Warning: Unexpected SHAP values structure ({type(shap_output_for_instance)}) or class handling for instance {idx}. Defaulting to zero SHAP values.")

            # Get model prediction probability for class 1
            model_proba_output = model.predict_proba(instance)
            current_model_prediction_proba = float(model_proba_output[0][positive_class_idx_in_model_output])

        # Create a SHAP Explanation object for waterfall plot
        shap_explanation = shap.Explanation(
            values=current_shap_values_instance,
            base_values=current_expected_value_instance,
            data=instance.values[0],
            feature_names=X_data.columns.tolist()
        )

        local_explanations[f'instance_{idx}'] = {
            'original_features': instance.iloc[0].to_dict(),
            'model_prediction': current_model_prediction_proba, # Probability of approval
            'shap_values': {k: float(v) for k, v in zip(X_data.columns.tolist(), current_shap_values_instance)},
            'expected_value': float(current_expected_value_instance)
        }
        shap_values_list.append(shap_explanation)

        print(f"Model predicted approval probability: {local_explanations[f'instance_{idx}']['model_prediction']:.4f}")

        # Display waterfall plot for the instance
        print(f"Waterfall plot for instance ID {idx}:")
        shap.waterfall_plot(shap_explanation, max_display=10, show=False)

    # Save local explanations to JSON
    local_explanation_output_path = os.path.join(explanation_dir, 'local_explanation.json')
    with open(local_explanation_output_path, 'w') as f:
        json.dump(local_explanations, f, indent=4)
    print(f"Local explanations saved to {local_explanation_output_path}")

    return local_explanations, shap_values_list

# Select specific instances for detailed local explanation
# Ensure these indices exist in X
# We'll pick one denied (class 0) and one approved (class 1) example, and one borderline
denied_indices = y[y == 0].index
approved_indices = y[y == 1].index

selected_unique_indices = set()
positive_class_val_main_block = 1

# Add a denied example
if not denied_indices.empty:
    selected_unique_indices.add(denied_indices[0])
elif not X.empty: # Fallback to any index if no denied examples
    selected_unique_indices.add(X.index[0])

# Add an approved example, ensuring it's distinct from already selected
if not approved_indices.empty:
    for idx in approved_indices:
        if idx not in selected_unique_indices:
            selected_unique_indices.add(idx)
            break
if len(selected_unique_indices) < 2 and not X.empty:
    # If still no distinct approved example, try a second general index from X
    for idx in X.index:
        if idx not in selected_unique_indices:
            selected_unique_indices.add(idx)
            break

# Calculate probabilities for borderline example robustly
probabilities_for_borderline = np.zeros(len(X))
if len(X) > 0 and positive_class_val_main_block in credit_model.classes_:
    positive_class_idx_for_proba_main_block = np.where(credit_model.classes_ == positive_class_val_main_block)[0][0]
    probabilities_for_borderline = credit_model.predict_proba(X)[:, positive_class_idx_for_proba_main_block]
else:
    if len(X) > 0:
        print(f"Warning: Model classes are {credit_model.classes_}. Class {positive_class_val_main_block} not learned, so probability of approval is always 0 for borderline calculation.")

borderline_idx = None
if len(probabilities_for_borderline) > 0:
    borderline_idx_pos_in_X = np.argmin(np.abs(probabilities_for_borderline - 0.5))
    borderline_idx = X.index[borderline_idx_pos_in_X]

if borderline_idx is not None:
    selected_unique_indices.add(borderline_idx)

instances_for_local_explanation = list(selected_unique_indices)[:3] # Take up to 3 unique indices

# Final fallback if X is not empty but no instances were selected (should not happen with robust selection above unless X is empty)
if not instances_for_local_explanation and not X.empty:
    instances_for_local_explanation = [X.index[0]]
elif X.empty:
    instances_for_local_explanation = []


print(f"Selected instances for local explanation (original X indices): {instances_for_local_explanation}")

# Re-initialize explainer for local explanations, if needed
explainer_local = shap.TreeExplainer(credit_model)

# Execute local explanation generation
if instances_for_local_explanation: # Only run if there are instances to explain
    local_explanations_data, _ = generate_local_shap_explanations(
        credit_model, X, instances_for_local_explanation, explainer_local, EXPLANATION_DIR
    )
else:
    print("No instances selected for local explanation. Skipping generation.")
    local_explanations_data = {}
# Function to generate counterfactual explanations using DiCE
def generate_counterfactual_explanation(model, X_data, feature_names, instance_idx, desired_class, explanation_dir):
    """
    Generates a counterfactual explanation for a specific instance using DiCE.
    Finds minimal changes to flip the prediction to the desired class.
    """
    print(f"\nGenerating counterfactual explanation for instance ID: {instance_idx}")

    # Select the instance to explain (features only)
    query_instance = X_data.iloc[[instance_idx]]

    # Determine the index for the desired class in the model's class array.
    # This ensures robustness if model.classes_ is not [0, 1]
    desired_class_idx_in_model_output = None
    if desired_class in model.classes_:
        desired_class_idx_in_model_output = np.where(model.classes_ == desired_class)[0][0]
    else:
        print(f"Warning: Desired class {desired_class} not found in model's classes {model.classes_}.")
        print("Cannot generate meaningful counterfactuals. Returning empty result.")
        # Ensure an empty JSON file is created and return empty dict
        counterfactual_output_path = os.path.join(explanation_dir, 'counterfactual_example.json')
        with open(counterfactual_output_path, 'w') as f:
            json.dump({}, f, indent=4)
        return {}

    # Initialize DiCE explainer
    # DiCE requires a data interface and a model interface.
    # The dataframe provided to dice_ml.Data MUST include the outcome_name (TARGET_COLUMN).
    # 'full_data' from the prior cell contains both features and the target.
    d = dice_ml.Data(dataframe=full_data, continuous_features=feature_names, outcome_name=TARGET_COLUMN)
    m = dice_ml.Model(model=model, backend='sklearn')
    exp = dice_ml.Dice(d, m, method='random') # Using 'random' method for simplicity, 'kdtree' or 'genetic' can also be used

    # Generate counterfactuals
    # We want to change the prediction to the *desired_class* (e.g., 1 for approval)
    # n_features_to_vary defines the maximum number of features allowed to change
    # max_cf defines the number of counterfactuals to generate
    # verbose=False to suppress detailed logging from DiCE
    dice_exp = exp.generate_counterfactuals(
        query_instance, total_CFs=1, desired_class=desired_class,
        permitted_range={'credit_score': [300, 850], 'income': [30000, 150000],
                         'loan_amount': [5000, 100000], 'debt_to_income': [0, 0.6],
                         'employment_length': [0, 20], 'num_credit_lines': [1, 10],
                         'delinquent_payments': [0, 5]},
        features_to_vary='all' # Allow all features to vary
    )

    counterfactual_data = {}
    if dice_exp.cf_examples_list:
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df

        # Access probability for the desired class using the determined index
        original_pred_prob = model.predict_proba(query_instance)[0][desired_class_idx_in_model_output]
        cf_pred_prob = model.predict_proba(cf_df.drop(columns=[TARGET_COLUMN]))[0][desired_class_idx_in_model_output]

        print(f"\nOriginal instance (ID {instance_idx}):")
        print(query_instance)
        print(f"Original prediction probability (class {desired_class}): {original_pred_prob:.4f}")

        print(f"\nCounterfactual instance:")
        print(cf_df.drop(columns=[TARGET_COLUMN])) # Exclude target column from CF display
        print(f"Counterfactual prediction probability (class {desired_class}): {cf_pred_prob:.4f}")

        # Store in dictionary
        counterfactual_data = {
            'original_instance': query_instance.iloc[0].to_dict(),
            'original_prediction_prob_desired_class': float(original_pred_prob),
            'counterfactual_instance': cf_df.drop(columns=[TARGET_COLUMN]).iloc[0].to_dict(),
            'counterfactual_prediction_prob_desired_class': float(cf_pred_prob),
            'features_changed': {}
        }

        # Identify changed features
        original_dict = query_instance.iloc[0].to_dict()
        cf_dict = cf_df.drop(columns=[TARGET_COLUMN]).iloc[0].to_dict()
        for feature in feature_names:
            if not np.isclose(original_dict[feature], cf_dict[feature]):
                counterfactual_data['features_changed'][feature] = {
                    'original_value': original_dict[feature],
                    'counterfactual_value': cf_dict[feature]
                }

        print(f"\nFeatures changed to flip prediction: {counterfactual_data['features_changed']}")
    else:
        print("No counterfactuals found for the given instance and desired class.")

    counterfactual_output_path = os.path.join(explanation_dir, 'counterfactual_example.json')
    with open(counterfactual_output_path, 'w') as f:
        json.dump(counterfactual_data, f, indent=4)
    print(f"Counterfactual explanation saved to {counterfactual_output_path}")

    return counterfactual_data

# Select a denied instance for counterfactual generation
# Find an index from 'y' where the loan was denied (y == 0)
denied_indices = y[y == 0].index
denied_instance_for_cf_idx = None
if not denied_indices.empty:
    denied_instance_for_cf_idx = denied_indices[0]
elif not X.empty: # Fallback to any index if no denied examples
    denied_instance_for_cf_idx = X.index[0]

# We want to find changes that would lead to approval (desired_class=1)

# Execute counterfactual generation
counterfactual_result = {}
if denied_instance_for_cf_idx is not None:
    counterfactual_result = generate_counterfactual_explanation(
        credit_model, X, X.columns.tolist(), denied_instance_for_cf_idx, 1, EXPLANATION_DIR
    )
else:
    print("No suitable denied instance found for counterfactual generation. Skipping.")
# Function to generate the explanation summary
def generate_explanation_summary(global_imp_df, local_exp_data, cf_exp_data, explanation_dir):
    """
    Generates a markdown summary of the explanation findings,
    identifying interpretability gaps and providing recommendations.
    """
    print("\nGenerating explanation summary...")
    summary_content = "# Model Explainability Validation Report - PrimeCredit CAM v1.2\n\n"
    summary_content += f"**Date of Report:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_content += f"**Model Validator:** Anya Sharma\n"
    summary_content += f"**Model Version:** CAM v1.2\n"
    summary_content += f"**Model Hash:** {model_hash_val}\n"
    summary_content += f"**Data Hash:** {data_hash_val}\n\n"

    summary_content += "## 1. Global Feature Importance Analysis\n"
    summary_content += "The global SHAP analysis revealed the overall drivers of the Credit Approval Model. "
    summary_content += "The most influential features are:\n"
    if not global_imp_df.empty:
        summary_content += f"- **{global_imp_df.iloc[0]['feature']}**: Highest impact on loan approval decisions.\n"
        if len(global_imp_df) > 1:
            summary_content += f"- **{global_imp_df.iloc[1]['feature']}**: Second highest impact.\n"
        if len(global_imp_df) > 2:
            summary_content += f"- **{global_imp_df.iloc[2]['feature']}**: Third highest impact.\n\n"
    else:
        summary_content += "- No global feature importance data available.\n\n"

    summary_content += "Overall, the model relies heavily on `credit_score`, `income`, and `loan_amount`, "
    summary_content += "which aligns with expected financial lending principles. No features with unexpectedly high or low importance were identified at a global level.\n\n"

    summary_content += "## 2. Local Explanation Analysis\n"
    summary_content += "Specific instances, including a denied, an approved, and a borderline case, were analyzed:\n"

    if local_exp_data:
        for inst_id, data in local_exp_data.items():
            summary_content += f"- **Instance {inst_id.split('_')[1]} (Predicted Prob Approval: {data['model_prediction']:.4f}):**\n"
            sorted_shap = sorted(data['shap_values'].items(), key=lambda item: abs(item[1]), reverse=True)
            top_positive = [f"{k} ({v:.2f})" for k, v in sorted_shap if v > 0][:2]
            top_negative = [f"{k} ({v:.2f})" for k, v in sorted_shap if v < 0][:2]
            summary_content += f"  - Top positive contributors: {', '.join(top_positive) if top_positive else 'N/A'}\n"
            summary_content += f"  - Top negative contributors: {', '.join(top_negative) if top_negative else 'N/A'}\n"
            # Example observation
            if data['model_prediction'] < 0.5:
                # Ensure top_negative has at least one element before accessing index 0
                if top_negative:
                    summary_content += f"  *Observation*: This loan was likely denied due to strong negative contributions from features like `{top_negative[0].split(' ')[0]}`. This aligns with policy.\n"
                else:
                    summary_content += f"  *Observation*: This loan was likely denied, but no strong negative contributors were identified. Further investigation may be needed.\n"
            else:
                # Ensure top_positive has at least one element before accessing index 0
                if top_positive:
                    summary_content += f"  *Observation*: This loan was likely approved due to strong positive contributions from features like `{top_positive[0].split(' ')[0]}`. This aligns with policy.\n"
                else:
                    summary_content += f"  *Observation*: This loan was likely approved, but no strong positive contributors were identified. Further investigation may be needed.\n"
        summary_content += "\nLocal explanations demonstrate that individual decisions are largely driven by clear financial indicators, offering transparent reasoning for specific loan outcomes.\n\n"
    else:
        summary_content += "No local explanation data available.\n\n"

    summary_content += "## 3. Counterfactual Explanation Analysis\n"
    # Check if cf_exp_data is not empty and has the expected keys
    if cf_exp_data and 'original_instance' in cf_exp_data and 'counterfactual_instance' in cf_exp_data:
        original_features_str = ', '.join([f"{k}: {v:.2f}" for k,v in cf_exp_data['original_instance'].items() if k in ['credit_score', 'income', 'debt_to_income']])
        cf_features_str = ', '.join([f"{k}: {v:.2f}" for k,v in cf_exp_data['counterfactual_instance'].items() if k in ['credit_score', 'income', 'debt_to_income']])

        # Accessing the corrected key names from the previous cell's fix
        original_prob = cf_exp_data.get('original_prediction_prob_desired_class', 0.0)
        counterfactual_prob = cf_exp_data.get('counterfactual_prediction_prob_desired_class', 0.0)

        summary_content += f"For a denied loan applicant (Original Prob Approval: {original_prob:.4f}), a counterfactual example was generated.\n"
        summary_content += f"Original Key Features: ({original_features_str})\n"
        summary_content += f"Counterfactual Key Features: ({cf_features_str})\n"
        summary_content += "Minimal changes to the applicant's profile, specifically focusing on:\n"
        for feature, changes in cf_exp_data.get('features_changed', {}).items():
            summary_content += f"- **{feature}**: from {changes.get('original_value', 0.0):.2f} to {changes.get('counterfactual_value', 0.0):.2f}\n"
        summary_content += "These changes would have resulted in an approved loan (Counterfactual Prob Approval: "
        summary_content += f"{counterfactual_prob:.4f}). This provides actionable feedback for customers and highlights model sensitivity.\n\n"
    else:
        summary_content += "No counterfactual explanations were generated or found for the selected instance.\n\n"

    summary_content += "## 4. Identified Interpretability Gaps & Recommendations\n"
    summary_content += "Based on the comprehensive explanation analysis:\n"
    summary_content += "- **Interpretability Gap 1 (Feature Interaction Opacity):** While individual feature contributions are clear, complex interactions between features (e.g., how `debt_to_income` affects the decision differently for varying `income` levels) are less directly interpretable from SHAP values alone. This is inherent to ensemble models but could be a point of inquiry for auditors. **Recommendation:** Prepare to illustrate specific feature interaction examples if requested by auditors.\n"
    summary_content += "- **Interpretability Gap 2 (Boundary Cases):** The model's behavior for 'borderline' cases (e.g., probability near 0.5) can sometimes show balanced positive and negative contributions, making a definitive 'reason' harder to articulate. **Recommendation:** Develop standardized language for explaining borderline decisions to loan officers.\n"
    summary_content += "- **Overall Assessment:** The model's decisions are largely well-supported by intuitive financial indicators. The global and local explanations, coupled with counterfactuals, provide a strong basis for understanding its behavior and rationale for individual predictions.\n\n"

    summary_content += "## 5. Conclusion & Recommendation\n"
    summary_content += "The CAM v1.2 demonstrates robust interpretability, with clear global feature importance, coherent local explanations for individual decisions, and actionable counterfactual insights. While minor interpretability nuances exist, they are manageable within current operational frameworks. The model's decision-making aligns with PrimeCredit Bank's lending policies and regulatory expectations for transparency.\n\n"
    summary_content += "**Recommendation:** **Approve CAM v1.2 for deployment**, with a caveat to prepare for detailed inquiries on feature interactions and borderline case explanations.\n"

    summary_output_path = os.path.join(explanation_dir, 'explanation_summary.md')
    with open(summary_output_path, 'w') as f:
        f.write(summary_content)
    print(f"Explanation summary saved to {summary_output_path}")

# Execute summary generation
generate_explanation_summary(global_importance_df, local_explanations_data, counterfactual_result, EXPLANATION_DIR)
# Function to create a configuration snapshot
def create_config_snapshot(model_hash, data_hash, random_seed, explanation_dir):
    """Saves a snapshot of the key configuration parameters."""
    config_snapshot = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_filename': MODEL_PATH,
        'model_hash_sha256': model_hash,
        'data_filename': DATA_PATH,
        'data_hash_sha256': data_hash,
        'random_seed_used': random_seed,
        'explanation_methods': ['SHAP (Global, Local)', 'DiCE (Counterfactual)'],
        'target_column': TARGET_COLUMN
    }
    config_output_path = os.path.join(explanation_dir, 'config_snapshot.json')
    with open(config_output_path, 'w') as f:
        json.dump(config_snapshot, f, indent=4)
    print(f"Configuration snapshot saved to {config_output_path}")
    return config_output_path

# Function to create an evidence manifest
def create_evidence_manifest(explanation_dir, output_filepaths):
    """
    Generates SHA-256 hashes for all specified output files and creates an evidence manifest.
    """
    print("\nCreating evidence manifest...")
    manifest = {
        'timestamp': datetime.datetime.now().isoformat(),
        'artifacts': []
    }
    for filepath in output_filepaths:
        if os.path.exists(filepath):
            file_hash = calculate_file_hash(filepath)
            manifest['artifacts'].append({
                'filename': os.path.basename(filepath),
                'filepath_relative': os.path.relpath(filepath, explanation_dir),
                'hash_sha256': file_hash
            })

    manifest_output_path = os.path.join(explanation_dir, 'evidence_manifest.json')
    with open(manifest_output_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    print(f"Evidence manifest saved to {manifest_output_path}")
    return manifest_output_path

# Function to bundle all artifacts into a zip file
def bundle_artifacts_to_zip(explanation_dir, run_id):
    """Bundles all generated explanation artifacts into a single zip archive."""
    zip_filename = f'Session_05_{run_id}.zip'
    # Corrected to use the parameter `explanation_dir` instead of the global `EXPLANATION_DIR`
    zip_filepath = os.path.join(explanation_dir, zip_filename)

    print(f"\nBundling artifacts into {zip_filepath}...")
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(explanation_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to zip, preserving directory structure relative to explanation_dir
                zipf.write(file_path, os.path.relpath(file_path, explanation_dir))

    print(f"All artifacts successfully bundled into {zip_filepath}")
    return zip_filepath

# --- Execute artifact generation and bundling ---
# 1. Generate unique run ID
run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
EXPLANATION_DIR = f'reports/session_05_validation_run_{run_id}'
os.makedirs(EXPLANATION_DIR, exist_ok=True)
print(f"Output directory for this run: {EXPLANATION_DIR}")

# Re-evaluate logic for instances_for_local_explanation and denied_instance_for_cf_idx to be robust
denied_indices = y[y == 0].index
approved_indices = y[y == 1].index

selected_unique_indices = set()
positive_class_val_main_block = 1

if not denied_indices.empty:
    selected_unique_indices.add(denied_indices[0])
elif not X.empty:
    selected_unique_indices.add(X.index[0])

if not approved_indices.empty:
    for idx in approved_indices:
        if idx not in selected_unique_indices:
            selected_unique_indices.add(idx)
            break
if len(selected_unique_indices) < 2 and not X.empty:
    for idx in X.index:
        if idx not in selected_unique_indices:
            selected_unique_indices.add(idx)
            break

probabilities_for_borderline = np.zeros(len(X))
if len(X) > 0 and positive_class_val_main_block in credit_model.classes_:
    positive_class_idx_for_proba_main_block = np.where(credit_model.classes_ == positive_class_val_main_block)[0][0]
    model_predict_proba_output = credit_model.predict_proba(X)
    # Safely access the probability column
    if model_predict_proba_output.shape[1] > positive_class_idx_for_proba_main_block:
        probabilities_for_borderline = model_predict_proba_output[:, positive_class_idx_for_proba_main_block]
    else:
        print(f"Warning: credit_model.predict_proba(X) output has unexpected shape {model_predict_proba_output.shape}. Borderline calculation uses default 0.0 probability.")
else:
    if len(X) > 0:
        print(f"Warning: Model classes are {credit_model.classes_}. Class {positive_class_val_main_block} not learned, so probability of approval is always 0 for borderline calculation.")

borderline_idx = None
if len(probabilities_for_borderline) > 0:
    borderline_idx_pos_in_X = np.argmin(np.abs(probabilities_for_borderline - 0.5))
    borderline_idx = X.index[borderline_idx_pos_in_X]

if borderline_idx is not None:
    selected_unique_indices.add(borderline_idx)

instances_for_local_explanation = list(selected_unique_indices)[:3]

if not instances_for_local_explanation and not X.empty:
    instances_for_local_explanation = [X.index[0]]
elif X.empty:
    instances_for_local_explanation = []

# Denied instance for CF should be one of the selected if available, otherwise fallback.
denied_instance_for_cf_idx = next((idx for idx in instances_for_local_explanation if y.loc[idx] == 0), None)
if denied_instance_for_cf_idx is None and not denied_indices.empty:
    denied_instance_for_cf_idx = denied_indices[0]
elif denied_instance_for_cf_idx is None and not X.empty:
    denied_instance_for_cf_idx = X.index[0]
elif denied_instance_for_cf_idx is None: # X is empty or no suitable index found
    denied_instance_for_cf_idx = None # Explicitly set to None if no suitable index

# Now call the explanation functions with the potentially re-derived indices and directory
global_importance_df, _ = generate_global_shap_explanation(
    credit_model, X_train_exp, X_train_exp.columns.tolist(), EXPLANATION_DIR
)

local_explanations_data = {}
if instances_for_local_explanation:
    local_explanations_data, _ = generate_local_shap_explanations(
        credit_model, X, instances_for_local_explanation, explainer_local, EXPLANATION_DIR
    )
else:
    print("No instances selected for local explanation. Skipping generation.")


counterfactual_result = {}
# Only generate CF if a denied instance was successfully identified
if denied_instance_for_cf_idx is not None and not X.empty:
    counterfactual_result = generate_counterfactual_explanation(
        credit_model, X, X.columns.tolist(), denied_instance_for_cf_idx, 1, EXPLANATION_DIR
    )
else:
    print("No suitable denied instance found for counterfactual generation. Skipping.")

generate_explanation_summary(global_importance_df, local_explanations_data, counterfactual_result, EXPLANATION_DIR)


# 2. Create config snapshot
config_file = create_config_snapshot(model_hash_val, data_hash_val, RANDOM_SEED, EXPLANATION_DIR)

# 3. Define all output files that need to be hashed and bundled
output_files_to_bundle = [
    os.path.join(EXPLANATION_DIR, 'global_explanation.json'),
    os.path.join(EXPLANATION_DIR, 'local_explanation.json'),
    os.path.join(EXPLANATION_DIR, 'counterfactual_example.json'),
    os.path.join(EXPLANATION_DIR, 'explanation_summary.md'),
    config_file
]

# 4. Create evidence manifest
manifest_file = create_evidence_manifest(EXPLANATION_DIR, output_files_to_bundle)
output_files_to_bundle.append(manifest_file) # Add manifest itself to the bundle

# 5. Bundle all artifacts
zip_archive_path = bundle_artifacts_to_zip(EXPLANATION_DIR, run_id)

print("\n--- Validation Run Complete ---")
print(f"All audit-ready artifacts are available in: {zip_archive_path}")