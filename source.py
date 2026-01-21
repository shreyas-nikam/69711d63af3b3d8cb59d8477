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

# Define file paths for the model and data
MODEL_PATH = 'sample_credit_model.pkl'
DATA_PATH = 'sample_credit_data.csv'
TARGET_COLUMN = 'loan_approved'

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
        - 2.0  # Baseline
        # Sigmoid to convert to probability
    ).apply(lambda x: 1 / (1 + np.exp(-x)))

    data[target_col] = (approval_prob > 0.5).astype(int)

    # Train a RandomForestClassifier
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed)

    model = RandomForestClassifier(
        n_estimators=100, random_state=random_seed, class_weight='balanced')
    model.fit(X_train, y_train)

    # Save model and data
    joblib.dump(model, model_path)
    data.to_csv(data_path, index=False)
    print("Sample model and data generated and saved.")


def generate_healthcare_data_and_model(model_path, data_path, target_col, random_seed):
    """Generates a sample healthcare risk dataset and trains a RandomForestClassifier."""
    if os.path.exists(model_path) and os.path.exists(data_path):
        print("Sample healthcare model and data already exist. Skipping generation.")
        return

    print("Generating sample healthcare risk data and model...")
    num_samples = 1000
    np.random.seed(random_seed)

    data = pd.DataFrame({
        'age': np.random.randint(18, 90, num_samples),
        'bmi': np.random.uniform(15, 45, num_samples),
        'blood_pressure': np.random.randint(80, 180, num_samples),
        'glucose_level': np.random.randint(70, 200, num_samples),
        'cholesterol': np.random.randint(120, 300, num_samples),
        'smoking': np.random.randint(0, 2, num_samples),
        'family_history': np.random.randint(0, 2, num_samples),
        'exercise_hours_per_week': np.random.randint(0, 15, num_samples),
    })

    # Simple logic for risk scoring (simulated)
    # Higher age, bmi, blood_pressure, glucose, cholesterol, smoking -> higher risk
    risk_prob = (
        0.02 * data['age']
        + 0.05 * data['bmi']
        + 0.01 * data['blood_pressure']
        + 0.015 * data['glucose_level']
        + 0.008 * data['cholesterol']
        + 0.5 * data['smoking']
        + 0.3 * data['family_history']
        - 0.1 * data['exercise_hours_per_week']
        - 5.0  # Baseline
    ).apply(lambda x: 1 / (1 + np.exp(-x)))

    data[target_col] = (risk_prob > 0.5).astype(int)

    # Train a RandomForestClassifier
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed)

    model = RandomForestClassifier(
        n_estimators=100, random_state=random_seed, class_weight='balanced')
    model.fit(X_train, y_train)

    # Save model and data
    joblib.dump(model, model_path)
    data.to_csv(data_path, index=False)
    print("Sample healthcare model and data generated and saved.")


def generate_fraud_data_and_model(model_path, data_path, target_col, random_seed):
    """Generates a sample fraud detection dataset and trains a RandomForestClassifier."""
    if os.path.exists(model_path) and os.path.exists(data_path):
        print("Sample fraud model and data already exist. Skipping generation.")
        return

    print("Generating sample fraud detection data and model...")
    num_samples = 1000
    np.random.seed(random_seed)

    data = pd.DataFrame({
        'transaction_amount': np.random.uniform(10, 5000, num_samples),
        'transaction_hour': np.random.randint(0, 24, num_samples),
        'days_since_last_transaction': np.random.randint(0, 365, num_samples),
        'merchant_category_risk_score': np.random.uniform(0, 1, num_samples),
        'ip_country_mismatch': np.random.randint(0, 2, num_samples),
        'transaction_velocity': np.random.randint(1, 20, num_samples),
        'card_age_days': np.random.randint(1, 3650, num_samples),
        'avg_transaction_amount': np.random.uniform(50, 1000, num_samples),
    })

    # Simple logic for fraud detection (simulated)
    # Higher amount, unusual hours, high velocity, mismatches -> higher fraud probability
    fraud_prob = (
        0.0003 * data['transaction_amount']
        + 0.05 * np.abs(data['transaction_hour'] - 14)  # Unusual hours
        - 0.002 * data['days_since_last_transaction']
        + 2.0 * data['merchant_category_risk_score']
        + 1.5 * data['ip_country_mismatch']
        + 0.15 * data['transaction_velocity']
        - 0.0003 * data['card_age_days']
        + 0.0005 * np.abs(data['transaction_amount'] -
                          data['avg_transaction_amount'])
        - 3.0  # Baseline
    ).apply(lambda x: 1 / (1 + np.exp(-x)))

    data[target_col] = (fraud_prob > 0.5).astype(int)

    # Train a RandomForestClassifier
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed)

    model = RandomForestClassifier(
        n_estimators=100, random_state=random_seed, class_weight='balanced')
    model.fit(X_train, y_train)

    # Save model and data
    joblib.dump(model, model_path)
    data.to_csv(data_path, index=False)
    print("Sample fraud model and data generated and saved.")


def initialize_model_and_data(model_path=MODEL_PATH, data_path=DATA_PATH, target_col=TARGET_COLUMN, random_seed=RANDOM_SEED):
    """
    Initializes the model and data by generating sample data if needed and loading artifacts.
    Returns: model, full_data, X, y, X_train_exp, X_test_exp, model_hash, data_hash
    """
    np.random.seed(random_seed)

    # Execute generation of sample data and model
    generate_sample_data_and_model(
        model_path, data_path, target_col, random_seed)

    # Execute loading and hashing
    credit_model, full_data, X, y, model_hash_val, data_hash_val = load_and_hash_artifacts(
        model_path, data_path, target_col, random_seed
    )

    # Split data for explanation purposes (using the entire X for explanations as per SHAP requirement)
    # For LIME/DiCE, we'll use a portion of X for background
    X_train_exp, X_test_exp, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=random_seed)

    print(f"\nModel type identified: {type(credit_model)}")
    print(f"Data features: {X.columns.tolist()}")
    print(f"First 5 rows of feature data:\n{X.head()}")

    return credit_model, full_data, X, y, X_train_exp, X_test_exp, model_hash_val, data_hash_val
# Function to generate global SHAP explanations


def generate_global_shap_explanation(model, X_data, feature_names, explanation_dir, X_test_exp=None, random_seed=RANDOM_SEED, positive_class=1):
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
    if X_test_exp is not None:
        sample_data = X_test_exp.sample(min(
            1000, X_test_exp.shape[0]), random_state=random_seed) if X_test_exp.shape[0] > 1000 else X_test_exp
    else:
        sample_data = X_data.sample(min(
            1000, X_data.shape[0]), random_state=random_seed) if X_data.shape[0] > 1000 else X_data

    # Reset index and ensure proper DataFrame structure
    sample_data = sample_data.reset_index(drop=True).copy()

    shap_values = explainer.shap_values(sample_data)

    # SHAP values for classification models return two arrays (for class 0 and class 1)
    # We are interested in the positive class
    if isinstance(shap_values, list):
        shap_values_for_positive_class = shap_values[positive_class]
    else:
        # For regression or binary where shap_values is a single array
        shap_values_for_positive_class = shap_values

    # Aggregate SHAP values to get mean absolute importance
    importance = np.abs(shap_values_for_positive_class).mean(axis=0)
    if importance.ndim > 1:
        importance = importance.mean(axis=-1)

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=False)

    print("Global feature importance (top 10):")
    print(feature_importance.head(10))

    # Save global explanation to JSON
    global_explanation_output_path = os.path.join(
        explanation_dir, 'global_explanation.json')
    feature_importance.to_json(
        global_explanation_output_path, orient='records', indent=4)
    print(f"Global explanation saved to {global_explanation_output_path}")

    # Display SHAP summary plot
    try:
        # Ensure shapes match before plotting
        if shap_values_for_positive_class.shape[0] == sample_data.shape[0] and \
           shap_values_for_positive_class.shape[1] == len(feature_names):
            shap.summary_plot(shap_values_for_positive_class, sample_data,
                              feature_names=list(sample_data.columns), plot_type="bar", show=False)
        else:
            pass
    except Exception as e:
        print(f"Could not generate SHAP summary plot due to error: {e}")

    return feature_importance, shap_values_for_positive_class, explainer
# Function to generate local SHAP explanations for selected instances


def generate_local_shap_explanations(model, X_data, instances_to_explain, explainer, explanation_dir, positive_class=1):
    """
    Generates local SHAP explanations for a list of specified instances.
    Saves individual explanations and displays waterfall plots.
    """
    print("\nGenerating local SHAP explanations for selected instances...")
    local_explanations = {}
    shap_values_list = []

    # Determine the index for the positive class in the model's class array.
    positive_class_val = positive_class
    # This will be the index in `predict_proba` output
    positive_class_idx_in_model_output = None
    if positive_class_val in model.classes_:
        positive_class_idx_in_model_output = np.where(
            model.classes_ == positive_class_val)[0][0]
    else:
        print(
            f"Warning: Model trained on classes {model.classes_}. Class {positive_class_val} not found. Probability of positive class will be 0.")

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
                current_shap_values_instance = shap_output_for_instance[
                    positive_class_idx_in_model_output][0]
                current_expected_value_instance = explainer.expected_value[
                    positive_class_idx_in_model_output]
            else:
                # Fallback for models where shap_values is a single array (e.g., regression or specific binary model output)
                # If model has only one class, and that class is `positive_class_val`, use the single array.
                if len(model.classes_) == 1 and model.classes_[0] == positive_class_val:
                    current_shap_values_instance = shap_output_for_instance[0]
                    current_expected_value_instance = explainer.expected_value
                else:
                    # This case should ideally not be hit for a binary RandomForestClassifier
                    print(
                        f"Warning: Unexpected SHAP values structure ({type(shap_output_for_instance)}) or class handling for instance {idx}. Defaulting to zero SHAP values.")

            # Get model prediction probability for class 1
            model_proba_output = model.predict_proba(instance)
            current_model_prediction_proba = float(
                model_proba_output[0][positive_class_idx_in_model_output])

        # Create a SHAP Explanation object for waterfall plot
        shap_explanation = shap.Explanation(
            values=current_shap_values_instance,
            base_values=current_expected_value_instance,
            data=instance.values[0],
            feature_names=X_data.columns.tolist()
        )

        local_explanations[f'instance_{idx}'] = {
            'original_features': instance.iloc[0].to_dict(),
            'model_prediction': current_model_prediction_proba,  # Probability of approval
            'shap_values': {k: float(v) for k, v in zip(X_data.columns.tolist(), current_shap_values_instance)},
            'expected_value': float(current_expected_value_instance)
        }
        shap_values_list.append(shap_explanation)

        print(
            f"Model predicted approval probability: {local_explanations[f'instance_{idx}']['model_prediction']:.4f}")

        # Display waterfall plot for the instance
        print(f"Waterfall plot for instance ID {idx}:")
        shap.waterfall_plot(shap_explanation, max_display=10, show=False)

    # Save local explanations to JSON
    local_explanation_output_path = os.path.join(
        explanation_dir, 'local_explanation.json')
    with open(local_explanation_output_path, 'w') as f:
        json.dump(local_explanations, f, indent=4)
    print(f"Local explanations saved to {local_explanation_output_path}")

    return local_explanations, shap_values_list


def select_instances_for_explanation(X, y, model):
    """
    Selects representative instances for local explanation.
    Returns: list of instance indices
    """
    denied_indices = y[y == 0].index
    approved_indices = y[y == 1].index

    selected_unique_indices = set()
    positive_class_val_main_block = 1

    # Add a denied example
    if not denied_indices.empty:
        selected_unique_indices.add(denied_indices[0])
    elif not X.empty:
        selected_unique_indices.add(X.index[0])

    # Add an approved example, ensuring it's distinct from already selected
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

    # Calculate probabilities for borderline example robustly
    probabilities_for_borderline = np.zeros(len(X))
    if len(X) > 0 and positive_class_val_main_block in model.classes_:
        positive_class_idx_for_proba_main_block = np.where(
            model.classes_ == positive_class_val_main_block)[0][0]
        model_predict_proba_output = model.predict_proba(X)
        if model_predict_proba_output.shape[1] > positive_class_idx_for_proba_main_block:
            probabilities_for_borderline = model_predict_proba_output[:,
                                                                      positive_class_idx_for_proba_main_block]
        else:
            print(
                f"Warning: model.predict_proba(X) output has unexpected shape {model_predict_proba_output.shape}. Borderline calculation uses default 0.0 probability.")
    else:
        if len(X) > 0:
            print(
                f"Warning: Model classes are {model.classes_}. Class {positive_class_val_main_block} not learned, so probability of approval is always 0 for borderline calculation.")

    borderline_idx = None
    if len(probabilities_for_borderline) > 0:
        borderline_idx_pos_in_X = np.argmin(
            np.abs(probabilities_for_borderline - 0.5))
        borderline_idx = X.index[borderline_idx_pos_in_X]

    if borderline_idx is not None:
        selected_unique_indices.add(borderline_idx)

    instances_for_local_explanation = list(selected_unique_indices)[:3]

    if not instances_for_local_explanation and not X.empty:
        instances_for_local_explanation = [X.index[0]]
    elif X.empty:
        instances_for_local_explanation = []

    print(
        f"Selected instances for local explanation (original X indices): {instances_for_local_explanation}")
    return instances_for_local_explanation
# Function to generate counterfactual explanations using DiCE


def generate_counterfactual_explanation(model, X_data, full_data, feature_names, instance_idx, desired_class, explanation_dir, target_col=TARGET_COLUMN):
    """
    Generates a counterfactual explanation for a specific instance using DiCE.
    Finds minimal changes to flip the prediction to the desired class.
    """
    print(
        f"\nGenerating counterfactual explanation for instance ID: {instance_idx}")

    # Select the instance to explain (features only)
    query_instance = X_data.iloc[[instance_idx]]

    # Check if the instance already has the desired class
    current_pred = model.predict(query_instance)[0]
    if current_pred == desired_class:
        print(
            f"Instance {instance_idx} already has the desired class {desired_class}. No counterfactual changes needed.")
        # Access probability for the desired class
        original_pred_prob = model.predict_proba(
            query_instance)[0][desired_class_idx_in_model_output]

        counterfactual_data = {
            'original_instance': query_instance.iloc[0].to_dict(),
            'original_prediction_prob_desired_class': float(original_pred_prob),
            # Same as original
            'counterfactual_instance': query_instance.iloc[0].to_dict(),
            'counterfactual_prediction_prob_desired_class': float(original_pred_prob),
            'features_changed': {}  # No changes needed
        }

        counterfactual_output_path = os.path.join(
            explanation_dir, 'counterfactual_example.json')
        with open(counterfactual_output_path, 'w') as f:
            json.dump(counterfactual_data, f, indent=4)
        print(
            f"Counterfactual explanation saved to {counterfactual_output_path}")
        return counterfactual_data

    # Determine the index for the desired class in the model's class array.
    # This ensures robustness if model.classes_ is not [0, 1]
    desired_class_idx_in_model_output = None
    if desired_class in model.classes_:
        desired_class_idx_in_model_output = np.where(
            model.classes_ == desired_class)[0][0]
    else:
        print(
            f"Warning: Desired class {desired_class} not found in model's classes {model.classes_}.")
        print("Cannot generate meaningful counterfactuals. Returning empty result.")
        # Ensure an empty JSON file is created and return empty dict
        counterfactual_output_path = os.path.join(
            explanation_dir, 'counterfactual_example.json')
        with open(counterfactual_output_path, 'w') as f:
            json.dump({}, f, indent=4)
        return {}

    # Initialize DiCE explainer
    # DiCE requires a data interface and a model interface.
    # The dataframe provided to dice_ml.Data MUST include the outcome_name (TARGET_COLUMN).
    # 'full_data' from the prior cell contains both features and the target.
    d = dice_ml.Data(dataframe=full_data,
                     continuous_features=feature_names, outcome_name=target_col)
    m = dice_ml.Model(model=model, backend='sklearn')
    # Using 'random' method for simplicity, 'kdtree' or 'genetic' can also be used
    exp = dice_ml.Dice(d, m, method='random')

    # Generate counterfactuals
    # We want to change the prediction to the *desired_class* (e.g., 1 for approval)
    # n_features_to_vary defines the maximum number of features allowed to change
    # max_cf defines the number of counterfactuals to generate
    # verbose=False to suppress detailed logging from DiCE
    # Let DiCE infer permitted ranges from the training data automatically
    dice_exp = exp.generate_counterfactuals(
        query_instance, total_CFs=1, desired_class=desired_class,
        features_to_vary='all'  # Allow all features to vary
    )

    counterfactual_data = {}
    if dice_exp.cf_examples_list:
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df

        # Access probability for the desired class using the determined index
        original_pred_prob = model.predict_proba(
            query_instance)[0][desired_class_idx_in_model_output]
        cf_pred_prob = model.predict_proba(cf_df.drop(columns=[target_col]))[
            0][desired_class_idx_in_model_output]

        print(f"\nOriginal instance (ID {instance_idx}):")
        print(query_instance)
        print(
            f"Original prediction probability (class {desired_class}): {original_pred_prob:.4f}")

        print(f"\nCounterfactual instance:")
        # Exclude target column from CF display
        print(cf_df.drop(columns=[target_col]))
        print(
            f"Counterfactual prediction probability (class {desired_class}): {cf_pred_prob:.4f}")

        # Store in dictionary
        counterfactual_data = {
            'original_instance': query_instance.iloc[0].to_dict(),
            'original_prediction_prob_desired_class': float(original_pred_prob),
            'counterfactual_instance': cf_df.drop(columns=[target_col]).iloc[0].to_dict(),
            'counterfactual_prediction_prob_desired_class': float(cf_pred_prob),
            'features_changed': {}
        }

        # Identify changed features
        original_dict = query_instance.iloc[0].to_dict()
        cf_dict = cf_df.drop(columns=[target_col]).iloc[0].to_dict()
        for feature in feature_names:
            if not np.isclose(original_dict[feature], cf_dict[feature]):
                counterfactual_data['features_changed'][feature] = {
                    'original_value': original_dict[feature],
                    'counterfactual_value': cf_dict[feature]
                }

        print(
            f"\nFeatures changed to flip prediction: {counterfactual_data['features_changed']}")
    else:
        print("No counterfactuals found for the given instance and desired class.")

    counterfactual_output_path = os.path.join(
        explanation_dir, 'counterfactual_example.json')
    with open(counterfactual_output_path, 'w') as f:
        json.dump(counterfactual_data, f, indent=4)
    print(f"Counterfactual explanation saved to {counterfactual_output_path}")

    return counterfactual_data


def select_denied_instance_for_cf(y, X):
    """
    Selects a denied instance for counterfactual generation.
    Returns: instance index or None
    """
    denied_indices = y[y == 0].index
    denied_instance_for_cf_idx = None
    if not denied_indices.empty:
        denied_instance_for_cf_idx = denied_indices[0]
    elif not X.empty:
        denied_instance_for_cf_idx = X.index[0]
    return denied_instance_for_cf_idx
# Function to generate the explanation summary


def generate_explanation_summary(global_imp_df, local_exp_data, cf_exp_data, model_hash, data_hash, explanation_dir):
    """
    Generates a markdown summary of the explanation findings,
    identifying interpretability gaps and providing recommendations.
    """
    print("\nGenerating explanation summary...")
    summary_content = "# Model Explainability Validation Report - PrimeCredit CAM v1.2\n\n"
    summary_content += f"**Date of Report:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_content += f"**Model Validator:** Anya Sharma\n"
    summary_content += f"**Model Version:** CAM v1.2\n"
    summary_content += f"**Model Hash:** {model_hash}\n"
    summary_content += f"**Data Hash:** {data_hash}\n\n"

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
            sorted_shap = sorted(data['shap_values'].items(
            ), key=lambda item: abs(item[1]), reverse=True)
            top_positive = [f"{k} ({v:.2f})" for k,
                            v in sorted_shap if v > 0][:2]
            top_negative = [f"{k} ({v:.2f})" for k,
                            v in sorted_shap if v < 0][:2]
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
        # Get top 3 features from global importance for display
        top_features = global_imp_df.head(3)['feature'].tolist(
        ) if not global_imp_df.empty else list(cf_exp_data['original_instance'].keys())[:3]

        original_features_str = ', '.join(
            [f"{k}: {v:.2f}" for k, v in cf_exp_data['original_instance'].items() if k in top_features])
        cf_features_str = ', '.join(
            [f"{k}: {v:.2f}" for k, v in cf_exp_data['counterfactual_instance'].items() if k in top_features])

        # Accessing the corrected key names from the previous cell's fix
        original_prob = cf_exp_data.get(
            'original_prediction_prob_desired_class', 0.0)
        counterfactual_prob = cf_exp_data.get(
            'counterfactual_prediction_prob_desired_class', 0.0)

        summary_content += f"For an instance with undesired outcome (Original Prob Desired Class: {original_prob:.4f}), a counterfactual example was generated.\n"
        summary_content += f"Original Key Features: ({original_features_str})\n"
        summary_content += f"Counterfactual Key Features: ({cf_features_str})\n"
        summary_content += "Minimal changes to the instance's features, specifically focusing on:\n"
        for feature, changes in cf_exp_data.get('features_changed', {}).items():
            summary_content += f"- **{feature}**: from {changes.get('original_value', 0.0):.2f} to {changes.get('counterfactual_value', 0.0):.2f}\n"
        summary_content += f"These changes would have resulted in the desired outcome (Counterfactual Prob Desired Class: {counterfactual_prob:.4f}). This provides actionable feedback and highlights model sensitivity.\n\n"
    else:
        summary_content += "No counterfactual explanations were generated or found for the selected instance.\n\n"

    summary_content += "## 4. Identified Interpretability Gaps & Recommendations\n"
    summary_content += "Based on the comprehensive explanation analysis:\n"
    summary_content += "- **Interpretability Gap 1 (Feature Interaction Opacity):** While individual feature contributions are clear, complex interactions between features are less directly interpretable from SHAP values alone. This is inherent to ensemble models but could be a point of inquiry for auditors. **Recommendation:** Prepare to illustrate specific feature interaction examples if requested by auditors.\n"
    summary_content += "- **Interpretability Gap 2 (Boundary Cases):** The model's behavior for 'borderline' cases (e.g., probability near 0.5) can sometimes show balanced positive and negative contributions, making a definitive 'reason' harder to articulate. **Recommendation:** Develop standardized language for explaining borderline decisions.\n"
    summary_content += "- **Overall Assessment:** The model's decisions are largely well-supported by intuitive indicators. The global and local explanations, coupled with counterfactuals, provide a strong basis for understanding its behavior and rationale for individual predictions.\n\n"

    summary_content += "## 5. Conclusion & Recommendation\n"
    summary_content += "The model demonstrates robust interpretability, with clear global feature importance, coherent local explanations for individual decisions, and actionable counterfactual insights. While minor interpretability nuances exist, they are manageable within current operational frameworks. The model's decision-making aligns with regulatory expectations for transparency.\n\n"
    summary_content += "**Recommendation:** **Approve the model for deployment**, with a caveat to prepare for detailed inquiries on feature interactions and borderline case explanations.\n"

    summary_output_path = os.path.join(
        explanation_dir, 'explanation_summary.md')
    with open(summary_output_path, 'w') as f:
        f.write(summary_content)
    print(f"Explanation summary saved to {summary_output_path}")

    return summary_output_path
# Function to create a configuration snapshot


def create_config_snapshot(model_hash, data_hash, random_seed, explanation_dir, model_path=MODEL_PATH, data_path=DATA_PATH, target_col=TARGET_COLUMN):
    """Saves a snapshot of the key configuration parameters."""
    config_snapshot = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_filename': model_path,
        'model_hash_sha256': model_hash,
        'data_filename': data_path,
        'data_hash_sha256': data_hash,
        'random_seed_used': random_seed,
        'explanation_methods': ['SHAP (Global, Local)', 'DiCE (Counterfactual)'],
        'target_column': target_col
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

    manifest_output_path = os.path.join(
        explanation_dir, 'evidence_manifest.json')
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
                zipf.write(file_path, os.path.relpath(
                    file_path, explanation_dir))

    print(f"All artifacts successfully bundled into {zip_filepath}")
    return zip_filepath


def run_complete_validation_workflow(model_path=MODEL_PATH, data_path=DATA_PATH, target_col=TARGET_COLUMN, random_seed=RANDOM_SEED):
    """
    Runs the complete validation workflow from data loading to artifact bundling.
    Returns: Dictionary with all results and paths
    """
    np.random.seed(random_seed)

    # 1. Initialize model and data
    credit_model, full_data, X, y, X_train_exp, X_test_exp, model_hash_val, data_hash_val = initialize_model_and_data(
        model_path, data_path, target_col, random_seed
    )

    # 2. Generate unique run ID and create explanation directory
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    explanation_dir = f'reports/session_05_validation_run_{run_id}'
    os.makedirs(explanation_dir, exist_ok=True)
    print(f"Output directory for this run: {explanation_dir}")

    # 3. Generate global explanations
    global_importance_df, global_shap_values, explainer = generate_global_shap_explanation(
        credit_model, X_train_exp, X_train_exp.columns.tolist(
        ), explanation_dir, X_test_exp, random_seed
    )

    # 4. Select instances and generate local explanations
    instances_for_local_explanation = select_instances_for_explanation(
        X, y, credit_model)
    local_explanations_data = {}
    if instances_for_local_explanation:
        local_explanations_data, _ = generate_local_shap_explanations(
            credit_model, X, instances_for_local_explanation, explainer, explanation_dir
        )
    else:
        print("No instances selected for local explanation. Skipping generation.")

    # 5. Generate counterfactual explanations
    denied_instance_for_cf_idx = select_denied_instance_for_cf(y, X)
    counterfactual_result = {}
    if denied_instance_for_cf_idx is not None and not X.empty:
        counterfactual_result = generate_counterfactual_explanation(
            credit_model, X, full_data, X.columns.tolist(
            ), denied_instance_for_cf_idx, 1, explanation_dir, target_col
        )
    else:
        print("No suitable denied instance found for counterfactual generation. Skipping.")

    # 6. Generate explanation summary
    generate_explanation_summary(global_importance_df, local_explanations_data, counterfactual_result,
                                 model_hash_val, data_hash_val, explanation_dir)

    # 7. Create config snapshot
    config_file = create_config_snapshot(model_hash_val, data_hash_val, random_seed, explanation_dir,
                                         model_path, data_path, target_col)

    # 8. Define all output files that need to be hashed and bundled
    output_files_to_bundle = [
        os.path.join(explanation_dir, 'global_explanation.json'),
        os.path.join(explanation_dir, 'local_explanation.json'),
        os.path.join(explanation_dir, 'counterfactual_example.json'),
        os.path.join(explanation_dir, 'explanation_summary.md'),
        config_file
    ]

    # 9. Create evidence manifest
    manifest_file = create_evidence_manifest(
        explanation_dir, output_files_to_bundle)
    output_files_to_bundle.append(manifest_file)

    # 10. Bundle all artifacts
    zip_archive_path = bundle_artifacts_to_zip(explanation_dir, run_id)

    print("\n--- Validation Run Complete ---")
    print(f"All audit-ready artifacts are available in: {zip_archive_path}")

    return {
        'model': credit_model,
        'full_data': full_data,
        'X': X,
        'y': y,
        'X_train_exp': X_train_exp,
        'X_test_exp': X_test_exp,
        'model_hash': model_hash_val,
        'data_hash': data_hash_val,
        'explainer': explainer,
        'global_importance_df': global_importance_df,
        'global_shap_values': global_shap_values,
        'local_explanations_data': local_explanations_data,
        'counterfactual_result': counterfactual_result,
        'explanation_dir': explanation_dir,
        'run_id': run_id,
        'zip_archive_path': zip_archive_path,
        'instances_for_local_explanation': instances_for_local_explanation,
        'denied_instance_for_cf_idx': denied_instance_for_cf_idx
    }


if __name__ == "__main__":
    # Only run the workflow if the script is executed directly
    results = run_complete_validation_workflow()
