id: 69711d63af3b3d8cb59d8477_documentation
summary: Lab 5: Interpretability & Explainability Control Workbench Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Building an Interpretability & Explainability Control Workbench with Streamlit and SHAP/DiCE

## 1. Introduction to the Interpretability & Explainability Control Workbench
Duration: 0:08:00

Welcome to the QuLab: Interpretability & Explainability Control Workbench! In the rapidly evolving landscape of machine learning, merely building accurate models is no longer sufficient. Especially in high-stakes domains like finance, healthcare, or legal systems, understanding *why* a model makes a particular decision is paramount. This is where **eXplainable Artificial Intelligence (XAI)** comes into play.

This codelab will guide you through building and utilizing a comprehensive Streamlit application that serves as a **Model Interpretability and Explainability Control Workbench**. Our focus will be on the critical role of a **Model Validator** at a fictional financial institution, PrimeCredit Bank.

<aside class="positive">
<b>Why is Model Interpretability and Explainability Critical?</b>
<ul>
  <li><b>Regulatory Compliance:</b> Regulations like GDPR's "right to explanation" or financial industry guidelines (e.g., SR 11-7) demand transparency in algorithmic decision-making.</li>
  <li><b>Trust and Adoption:</b> Stakeholders (customers, business leaders, regulators) are more likely to trust and adopt models they can understand.</li>
  <li><b>Bias Detection:</b> Explanations can reveal unintended biases in models, which is crucial for fairness and ethical AI.</li>
  <li><b>Debugging and Improvement:</b> Understanding model logic helps data scientists debug errors and improve model performance.</li>
  <li><b>Actionable Insights:</b> Explanations can provide specific advice on how to change inputs to achieve a desired outcome.</li>
</ul>
</aside>

**Context: Anya Sharma, Model Validator at PrimeCredit Bank**

As Anya Sharma, a dedicated Model Validator at PrimeCredit Bank, your primary responsibility is to ensure that all machine learning models used in critical business processes are transparent, fair, and compliant with internal governance and external regulatory standards. Today, your focus is on a newly developed **Credit Approval Model (CAM v1.2)**, which will determine loan eligibility for our customers. Before this model can be deployed, you must rigorously assess its interpretability and explainability.

Your mission is to thoroughly vet this model, identifying any interpretability gaps that could lead to biased decisions, regulatory scrutiny, or a lack of trust from stakeholders. You need to demonstrate that the model's decisions are defensible and understandable, not just to you, but also to internal auditors and future regulators. This workbench will help you generate, analyze, and document the required explanations as audit-ready artifacts.

**Core Concepts We Will Explore:**

1.  **SHA-256 Hashing:** Ensuring the integrity and traceability of model and data artifacts.
2.  **SHAP (SHapley Additive exPlanations):** A powerful game-theoretic approach to explain the output of any machine learning model.
    *   **Global Explanations:** Understanding the overall behavior of the model and which features are generally most important.
    *   **Local Explanations:** Decomposing individual predictions to understand the contribution of each feature for a specific instance.
3.  **Counterfactual Explanations (using DiCE):** Answering "what if" questions by identifying the smallest changes to an instance's features that would flip a model's prediction (e.g., from denied to approved).
4.  **Auditability & Reproducibility:** Generating a comprehensive audit trail, including configuration snapshots and evidence manifests, to ensure all findings are verifiable.

**Application Workflow Overview:**

The Streamlit application provides a structured workflow, mimicking Anya's validation process.
<aside class="positive">
**Conceptual Workflow of the Workbench:**

1.  **Setup & Data:** Upload the ML model (e.g., `.pkl`) and its associated dataset (`.csv`). The system calculates cryptographic hashes for both artifacts, ensuring their integrity and creating an immutable record.
2.  **Global Explanations:** Compute and visualize feature importance across the *entire dataset* using SHAP, providing an aggregate view of what drives the model's decisions.
3.  **Local Explanations:** Select specific instances from the dataset and generate SHAP explanations for *each individual prediction*, detailing why a particular loan was approved or denied.
4.  **Counterfactuals:** For a *denied* loan application, generate counterfactual explanations that show the minimal changes needed to flip the outcome to an *approval*, offering actionable advice.
5.  **Summary & Audit:** Synthesize all findings into a comprehensive report. This step also bundles all generated explanations, configuration settings, and an audit manifest (containing hashes of all generated files) into a single, timestamped ZIP archive, ready for compliance checks.
</aside>

Let's begin by setting up our environment and ingesting the model and data.

## 2. Setting the Stage: Environment Setup and Data Ingestion
Duration: 0:05:00

My first step is to prepare my environment and load the necessary model and data for validation. Reproducibility is paramount in model validation; therefore, I will fix a random seed and compute SHA-256 hashes for both the model and dataset to ensure traceability and detect any unauthorized changes.

### a. Story + Context + Real-World Relevance

To begin, I need to install all required Python libraries. Following this, I'll import them and load the pre-trained `sample_credit_model.pkl` and its corresponding feature dataset, `sample_credit_data.csv`. This dataset represents historical loan applications with various features and an outcome indicating whether the loan was approved. Ensuring the integrity of the model and data is crucial; hence, I'll calculate unique cryptographic hashes for each, which will serve as a foundational element for auditability.

The application starts by ensuring sample data and a model exist locally for demonstration. If not, it generates them using the `generate_sample_data_and_model` function from `source.py`.

```python
# Initialization logic from the Streamlit app
if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    print(f"Sample model/data not found, generating to {MODEL_PATH} and {DATA_PATH}...")
    try:
        generate_sample_data_and_model(MODEL_PATH, DATA_PATH, TARGET_COLUMN, RANDOM_SEED)
        print("Sample model/data generation complete.")
    except Exception as e:
        print(f"Warning: Could not generate sample data: {e}")
else:
    print(f"Sample model/data found at {MODEL_PATH} and {DATA_PATH}.")
```

### b. Execution Steps

1.  **Navigate to "Setup & Data":** In the Streamlit sidebar, select the "Setup & Data" page.
2.  **Upload Model and Data:**
    *   Click "Browse files" under "Upload Model (.pkl or .joblib)" and select your model file (e.g., `sample_credit_model.pkl`).
    *   Click "Browse files" under "Upload Data (.csv)" and select your dataset file (e.g., `sample_credit_data.csv`).
    *   The application will save these temporarily.
3.  **Load and Hash Artifacts:** Click the "Load and Hash Artifacts" button.

<aside class="positive">
<b>Important:</b> If you upload new files, the application automatically clears previous explanation artifacts and resets the session state to ensure a fresh, reproducible validation run. A unique `run_id` is generated for each new session, creating a dedicated directory for explanation reports.
</aside>

When you click "Load and Hash Artifacts", the application executes the `load_and_hash_artifacts` function:

```python
# From source.py or equivalent logic
def load_and_hash_artifacts(model_path, data_path, target_column, random_seed):
    """Loads model and data, calculates their SHA-256 hashes, and splits data."""
    
    # Calculate model hash
    with open(model_path, 'rb') as f:
        model_hash_val = calculate_file_hash(f)
    model = joblib.load(model_path)

    # Load data and calculate hash
    full_data = pd.read_csv(data_path)
    with open(data_path, 'rb') as f:
        data_hash_val = calculate_file_hash(f)

    # Split data into features (X) and target (y)
    X = full_data.drop(columns=[target_column])
    y = full_data[target_column]
    
    # Initialize SHAP explainer (TreeExplainer for tree-based models)
    explainer = shap.TreeExplainer(model)

    return model, full_data, X, y, model_hash_val, data_hash_val
```

This function performs the following critical actions:
*   Loads the model using `joblib.load()`.
*   Loads the dataset using `pd.read_csv()`.
*   Calculates the **SHA-256 hash** for both the model and data files using `calculate_file_hash()`. This hash serves as a unique digital fingerprint, guaranteeing the integrity and version control of these crucial artifacts.
*   Splits the data into features (`X`) and the target variable (`y`).
*   Initializes a `shap.TreeExplainer` for the loaded model, which will be used for subsequent explanation generation.

The application will then display a summary of the loaded artifacts, including the model type, its hash, the data hash, and the features identified.

<aside class="negative">
If you encounter an error, ensure your uploaded model is a `.pkl` or `.joblib` file and your data is a `.csv` file. Also, confirm that the `TARGET_COLUMN` defined in `source.py` (e.g., `loan_approved`) exists in your CSV data.
</aside>

### c. Explanation of Execution

The initial setup is complete. You've successfully loaded the Credit Approval Model and the associated feature dataset. Crucially, you've generated cryptographic hashes for both artifacts: `<model_hash>` for the model and `<data_hash>` for the data. These hashes are vital for maintaining an immutable audit trail; any future change to either the model or the dataset would result in a different hash, immediately signaling a potential issue to an auditor. This step aligns with PrimeCredit Bank's stringent requirements for data and model integrity. The data has also been pre-processed, separating features from the target variable, making it ready for explanation generation.

## 3. Unveiling Overall Behavior: Global Model Explanations
Duration: 0:07:00

### a. Story + Context + Real-World Relevance

As a Model Validator, you first need to grasp the overall behavior of the CAM v1.2. Which factors generally drive its decisions for approving or denying loans? Global explanations provide an aggregate view of feature importance, revealing which features have the most impact across all predictions. This helps you verify if the model's general logic aligns with PrimeCredit's lending policies and expert domain knowledge. For tree-based models like our `RandomForestClassifier`, SHAP (SHapley Additive exPlanations) values are an excellent choice for this. The SHAP value $\phi_i$ for a feature $i$ represents the average marginal contribution of that feature value to the prediction across all possible coalitions of features.

The fundamental idea behind SHAP is to connect optimal credit allocation with local explanations using **Shapley values** from cooperative game theory. For a prediction $f(x)$ for an instance $x$, SHAP explains this prediction as a sum of feature contributions $\phi_i$ plus a base value $\phi_0$ (the expected output of the model):

$$
f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(f, x)
$$

where $\phi_0$ is the expected model output (the base value), $M$ is the number of features, $\phi_i(f, x)$ is the SHAP value for feature $i$ for instance $x$, and $f(x)$ is the model's prediction for instance $x$. For global explanations, we often look at the average absolute SHAP value for each feature.

### b. Execution Steps

1.  **Navigate to "Global Explanations":** In the Streamlit sidebar, select the "Global Explanations" page.
2.  **Generate Global Explanations:** Click the "Generate Global Explanations" button.

This will invoke the `generate_global_shap_explanation` function:

```python
# From source.py or equivalent logic
def generate_global_shap_explanation(model, X_data, feature_names, explanation_dir):
    """Generates global SHAP explanations and saves plots/data."""
    
    explainer = shap.TreeExplainer(model)
    # Calculate SHAP values for a subset of the training data
    # X_data here is typically X_train_exp from session state
    shap_values = explainer.shap_values(X_data)
    
    # For classification models, shap_values can be a list of arrays (one for each class)
    # We typically use the SHAP values for the positive class (class 1 for approval)
    if isinstance(shap_values, list):
        shap_values_for_plot = shap_values[1] # Assuming class 1 is the positive outcome
    else:
        shap_values_for_plot = shap_values
    
    # Create a DataFrame for global importance
    global_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Absolute_SHAP_Value': np.abs(shap_values_for_plot).mean(axis=0)
    }).sort_values(by='Mean_Absolute_SHAP_Value', ascending=False)

    # Save global importance as JSON
    output_path = os.path.join(explanation_dir, 'global_explanation.json')
    global_importance_df.to_json(output_path, orient='records', indent=4)
    
    return global_importance_df, shap_values_for_plot
```

The function performs the following:
*   Initializes a `shap.TreeExplainer` (if not already in session state) for the model.
*   Computes SHAP values across a representative subset of the data (often the training set, `X_train_exp`).
*   Calculates the mean absolute SHAP value for each feature, which indicates its global importance.
*   Saves this global importance data as a JSON file for auditability.

The Streamlit app will then display:
*   A table showing the "Global Feature Importance" (mean absolute SHAP values).
*   A **Global SHAP Summary Plot**, which visually represents the top features.

```python
# Example of displaying the plot in Streamlit
def display_shap_plot(plot_func, *args, **kwargs):
    fig = plt.figure() # Create a new figure
    plot_func(*args, **kwargs, show=False) # Call SHAP plot function
    st.pyplot(fig) # Display the figure in Streamlit
    plt.close(fig) # Close the figure to prevent display issues

# Displayed in Streamlit:
display_shap_plot(shap.summary_plot, shap_vals, data_for_plot, plot_type="bar")
```

### c. Explanation of Execution

The global SHAP explanation reveals the overall drivers of the Credit Approval Model. From the summary plot and the `global_importance_df`, I can clearly see which features, such as `credit_score` and `income`, are most influential in the model's decisions regarding loan approval. This high-level overview confirms that the model is largely relying on expected financial health indicators, which aligns with PrimeCredit Bank's established lending criteria. This gives me initial confidence that the model's general behavior is sensible and explainable to senior stakeholders. However, global explanations only tell part of the story; I need to investigate specific individual decisions to ensure consistency and fairness.

## 4. Deep Dive into Individual Decisions: Local Explanations for Specific Loan Applications
Duration: 0:10:00

### a. Story + Context + Real-World Relevance

While global explanations are useful, they don't explain why a *specific* loan applicant was approved or denied. As a Model Validator, I frequently encounter requests to understand individual decisions, especially for denied applications or those with unusual profiles. For PrimeCredit Bank, it's crucial to provide clear, defensible reasons to customers for loan denials or approvals. I will select a few representative cases from our `sample_credit_data` to generate local explanations using SHAP. This allows me to examine the contribution of each feature to that particular prediction.

For a specific instance $x$, the SHAP values $\phi_i(f, x)$ quantify how much each feature $i$ contributes to the prediction $f(x)$ compared to the average prediction $\phi_0$. A positive SHAP value for a feature means it pushed the prediction higher (towards approval), while a negative value pushed it lower (towards denial).

The `shap.waterfall_plot` is particularly effective for local explanations. It visualizes how each feature's SHAP value contributes to moving the model's output from the base value ($\phi_0$) to the final prediction $f(x)$.

### b. Execution Steps

1.  **Navigate to "Local Explanations":** In the Streamlit sidebar, select the "Local Explanations" page.
2.  **Select Instances for Local Explanation:** Use the multiselect dropdown to choose specific instance IDs from the dataset. The app provides some default examples, including denied and approved cases, and potentially a borderline case.
3.  **Generate Local Explanations:** Click the "Generate Local Explanations" button.

This will execute the `generate_local_shap_explanations` function:

```python
# From source.py or equivalent logic
def generate_local_shap_explanations(model, X_data, selected_indices, explainer, explanation_dir):
    """Generates local SHAP explanations for selected instances and saves plots/data."""
    
    local_explanations_data = {}
    shap_values_list = []
    
    for idx in selected_indices:
        instance = X_data.loc[[idx]]
        shap_values_instance = explainer.shap_values(instance)
        
        # For classification, we use SHAP values for the positive class (class 1)
        if isinstance(shap_values_instance, list):
            shap_values_for_plot = shap_values_instance[1]
            expected_value_for_plot = explainer.expected_value[1]
        else:
            shap_values_for_plot = shap_values_instance
            expected_value_for_plot = explainer.expected_value

        # Create a SHAP explanation object for waterfall plot
        shap_explanation = shap.Explanation(
            values=shap_values_for_plot[0], # Take the first instance if batch prediction
            base_values=expected_value_for_plot,
            data=instance.iloc[0].values,
            feature_names=X_data.columns.tolist()
        )
        shap_values_list.append(shap_explanation)
        
        # Get model prediction probability for the positive class
        predicted_prob_positive = model.predict_proba(instance)[:, np.where(model.classes_ == 1)[0][0]][0]

        # Store explanation data
        local_explanations_data[f"instance_{idx}"] = {
            "original_features": instance.iloc[0].to_dict(),
            "model_prediction": predicted_prob_positive,
            "shap_values": {feat: val for feat, val in zip(X_data.columns, shap_values_for_plot[0])}
        }
    
    # Save local explanations as JSON
    output_path = os.path.join(explanation_dir, 'local_explanation.json')
    with open(output_path, 'w') as f:
        json.dump(local_explanations_data, f, indent=4)
        
    return local_explanations_data, shap_values_list
```

This function performs the following for each selected instance:
*   Extracts the instance's features.
*   Calculates SHAP values for that specific instance using the initialized `explainer`.
*   Obtains the model's predicted probability for the positive class (e.g., loan approval).
*   Stores the original features, prediction, and SHAP contributions in a dictionary.
*   Creates a `shap.Explanation` object, which is required for `shap.waterfall_plot`.
*   Saves all local explanation data as a JSON file.

The Streamlit app will then display for each selected instance:
*   Its original feature values.
*   The model's predicted approval probability.
*   The top 5 SHAP contributions.
*   A **SHAP Waterfall Plot** visualizing the contribution of each feature to the prediction.

```python
# Example of displaying the plot in Streamlit
# ... (display_shap_plot function is the same as before) ...

# Displayed in Streamlit:
display_shap_plot(shap.waterfall_plot, st.session_state.local_shap_explanations_list[i], max_display=10)
```

### c. Explanation of Execution

The local SHAP explanations provide critical insights into individual loan decisions. For instance, analyzing the waterfall plot for a *denied* application (e.g., `instance_ID`), I can clearly see that a low `credit_score` and high `debt_to_income` ratio were the primary negative contributors, pushing the loan approval probability below the threshold. Conversely, for an *approved* application, a high `credit_score` and `income` might be the dominant positive factors. For a *borderline* case, the contributions might be more balanced.

These detailed breakdowns are invaluable for Anya. They allow her to:
1.  **Verify decision logic:** Are the model's specific reasons for a decision coherent and justifiable according to PrimeCredit's policy?
2.  **Identify potential biases:** Do certain demographic features (if present) disproportionately influence decisions in specific cases without valid business rationale? (Note: no demographic features are in this sample data, but this is what a Model Validator would look for).
3.  **Provide actionable feedback:** Understand what factors led to a denial, which is crucial for communicating with applicants.

This level of detail is exactly what PrimeCredit's internal auditors and potentially regulators would require to validate the fairness and transparency of the model.

## 5. 'What If?': Understanding Counterfactuals for Actionable Insights
Duration: 0:12:00

### a. Story + Context + Real-World Relevance

For a denied loan applicant, merely knowing *why* they were denied (via local explanations) isn't always enough. As Anya, I also need to understand "what if?" – what minimal changes to their application would have resulted in an approval? This is where **counterfactual explanations** come in. They identify the smallest, most actionable changes to an applicant's features that would flip the model's decision from denial to approval. This information is invaluable for PrimeCredit Bank, not only for providing constructive feedback to customers but also for potentially refining our lending criteria or identifying areas where applicants can improve their financial standing to become eligible.

Mathematically, a counterfactual explanation $x'$ for an instance $x$ aims to find a new instance $x'$ that is as close as possible to $x$, but for which the model's prediction $f(x')$ is a desired outcome $y'$. This can be formulated as an optimization problem:

$$
\min_{x'} \text{distance}(x, x') \quad \text{s.t.} \quad f(x') = y' \quad \text{and} \quad x' \in \mathcal{X}
$$

where $\text{distance}(x, x')$ is a measure of proximity (e.g., L1 or L2 norm), $f(x')$ is the model's prediction for the counterfactual instance $x'$, and $\mathcal{X}$ represents the set of valid feature values (e.g., within reasonable ranges). The `DiCE` (Diverse Counterfactual Explanations) library is often used for this purpose, generating multiple, diverse counterfactuals.

### b. Execution Steps

1.  **Navigate to "Counterfactuals":** In the Streamlit sidebar, select the "Counterfactuals" page.
2.  **Select a Denied Instance:** Use the dropdown to choose an instance that was originally denied (i.e., `loan_approved == 0`). The application will attempt to pre-select one.
3.  **Choose Desired Outcome:** Select "Approval (1)" as the desired outcome, as we want to find changes that lead to approval for a denied loan.
4.  **Generate Counterfactuals:** Click the "Generate Counterfactuals" button.

This will execute the `generate_counterfactual_explanation` function:

```python
# From source.py or equivalent logic (simplified for brevity)
import dice_ml

def generate_counterfactual_explanation(model, X_data, feature_names, instance_idx, desired_class, explanation_dir):
    """Generates counterfactual explanations for a selected instance."""
    
    # Setup DiCE data and model objects
    d = dice_ml.Data(
        dataframe=X_data.reset_index(drop=True), # DiCE typically works on full dataframe
        continuous_features=feature_names,
        outcome_name=TARGET_COLUMN # DiCE needs an outcome name, often a dummy for generating CFs
    )
    m = dice_ml.Model(model=model, model_type='classifier')
    exp = dice_ml.Dice(d, m, method='random') # Using 'random' method for simplicity

    # Select the instance for which to generate CFs
    query_instance = X_data.loc[[instance_idx]]

    # Generate counterfactuals
    dice_exp = exp.generate_counterfactuals(
        query_instance, 
        total_CFs=1, 
        desired_class=desired_class # Desired class (e.g., 1 for approval)
    )

    # Extract original and counterfactual instance data
    original_instance_dict = query_instance.iloc[0].to_dict()
    cf_data = dice_exp.cf_examples_list[0].final_cfs_df.iloc[0].to_dict() if dice_exp.cf_examples_list else None
    
    # Calculate original and counterfactual prediction probabilities
    original_pred_prob = model.predict_proba(query_instance)[:, np.where(model.classes_ == desired_class)[0][0]][0]
    counterfactual_pred_prob = model.predict_proba(pd.DataFrame([cf_data], columns=feature_names))[:, np.where(model.classes_ == desired_class)[0][0]][0] if cf_data else None

    # Identify changed features
    features_changed = {}
    if cf_data:
        for feature in feature_names:
            original_value = original_instance_dict.get(feature)
            cf_value = cf_data.get(feature)
            if original_value is not None and cf_value is not None and original_value != cf_value:
                features_changed[feature] = {
                    "original_value": original_value,
                    "counterfactual_value": cf_value
                }

    result = {
        "original_instance": original_instance_dict,
        "original_prediction_prob_desired_class": original_pred_prob,
        "counterfactual_instance": cf_data,
        "counterfactual_prediction_prob_desired_class": counterfactual_pred_prob,
        "features_changed": features_changed
    }

    # Save counterfactual data as JSON
    output_path = os.path.join(explanation_dir, 'counterfactual_example.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

    return result
```

This function performs the following:
*   Initializes `DiCE` with the model and data.
*   Generates one (or more) counterfactual instances for the selected denied applicant, aiming to achieve the `desired_class` (approval).
*   Compares the original instance with the generated counterfactual to highlight minimal feature changes required to flip the prediction.
*   Saves the counterfactual data as a JSON file.

The Streamlit app will then display:
*   The original instance's feature values and its prediction probability for the desired class.
*   The generated counterfactual instance's feature values and its prediction probability.
*   A summary of the "Minimal Feature Changes to Flip Prediction", showing exactly which features need to change and by how much.

### c. Explanation of Execution

The counterfactual analysis provides invaluable "what-if" scenarios for PrimeCredit Bank. For the selected denied loan application (`instance_ID`), the `counterfactual_result` clearly shows that increasing the `credit_score` by a certain amount or raising the `income` significantly, for example, would have resulted in the loan being approved. The `features_changed` dictionary pinpoints the minimal adjustments needed.

This empowers Anya to:
1.  **Inform customers:** Instead of just saying "your loan was denied," PrimeCredit can advise applicants on specific, actionable steps (e.g., "If your credit score improved by X points, you would likely be approved").
2.  **Refine policy:** If generating counterfactuals consistently highlights specific features as critical for flipping decisions, it might indicate areas for policy review or for developing financial literacy programs for customers.
3.  **Assess model sensitivity:** It reveals how sensitive the model is to changes in specific features, which is a key part of model validation.

This concrete evidence of actionable insights is crucial for establishing trust and demonstrating the model's utility beyond just making a prediction.

## 6. Audit Trail and Comprehensive Reporting
Duration: 0:15:00

### a. Story + Context + Real-World Relevance

After reviewing the global, local, and counterfactual explanations, Anya must now synthesize her findings and identify any interpretability gaps that could prevent the CAM v1.2 from being approved for deployment. This is a critical step for PrimeCredit Bank's risk management framework. An interpretability gap might be a feature that, while statistically significant, lacks a clear business rationale, or cases where local explanations seem inconsistent. I need to document my observations, evaluate the model's transparency, and make a recommendation for its deployment or further refinement.

My analysis will focus on:
*   **Coherence with Policy:** Do the explanations align with PrimeCredit's established lending policies and regulations?
*   **Transparency:** Are the reasons for decisions clear, concise, and easily understandable by non-technical stakeholders (e.g., loan officers, customers, auditors)?
*   **Consistency:** Do similar cases receive similar explanations, and are there any anomalous explanations?
*   **Actionability:** Do counterfactuals provide practical advice for applicants?

Based on these, I will formulate a summary of my findings and a recommendation.

The final, critical step for Anya is to ensure that all her validation work is reproducible and securely bundled for auditing purposes. For PrimeCredit Bank, regulatory compliance demands an immutable record of all explanation artifacts, along with the configuration and hashes that guarantee their traceability to specific model and data versions. This "audit-ready artifact bundle" acts as indisputable evidence of the model validation process. I will consolidate all generated explanations, configuration details, and an `evidence_manifest.json` containing SHA-256 hashes of each file, into a single, timestamped ZIP archive.

Recall the SHA-256 hashing process, which ensures the integrity of each file:

$$
\text{SHA-256}(\text{file\_content}) = \text{hexadecimal\_hash\_string}
$$

where $\text{SHA-256}$ is the cryptographic hash function, $\text{file\_content}$ is the input data (e.g., content of a file), and $\text{hexadecimal\_hash\_string}$ is the unique 256-bit hexadecimal output.

### b. Execution Steps

1.  **Navigate to "Summary & Audit":** In the Streamlit sidebar, select the "Summary & Audit" page.
2.  **Generate Explanation Summary Report:** Click the "Generate Explanation Summary Report" button.
    This will execute the `generate_explanation_summary` function. This function takes the global, local, and counterfactual explanation data and compiles them into a markdown-formatted summary report, which is then saved to the explanation directory. The Streamlit app displays the content of this report directly.
3.  **Export All Audit-Ready Artifacts:** Click the "Export All Audit-Ready Artifacts" button.

This final step orchestrates several functions to create a complete audit package:

```python
# From source.py or equivalent logic
def create_config_snapshot(model_hash, data_hash, random_seed, explanation_dir):
    """Creates a configuration snapshot JSON file."""
    config_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_hash_sha256": model_hash,
        "data_hash_sha256": data_hash,
        "random_seed_used": random_seed,
        "target_column": TARGET_COLUMN,
        # Add any other relevant configuration parameters
    }
    config_file_path = os.path.join(explanation_dir, 'config_snapshot.json')
    with open(config_file_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    return config_file_path

def create_evidence_manifest(explanation_dir, files_to_hash):
    """Creates an evidence manifest JSON file with hashes of all relevant output files."""
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "artifacts": []
    }
    for file_path in files_to_hash:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                file_hash = calculate_file_hash(f)
            manifest["artifacts"].append({
                "file_name": os.path.basename(file_path),
                "path_relative_to_archive": os.path.relpath(file_path, explanation_dir),
                "sha256_hash": file_hash
            })
    manifest_file_path = os.path.join(explanation_dir, 'evidence_manifest.json')
    with open(manifest_file_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    return manifest_file_path

def bundle_artifacts_to_zip(explanation_dir, run_id):
    """Bundles all generated explanation artifacts into a timestamped ZIP file."""
    zip_filename = f'PrimeCredit_CAM_v1.2_Validation_Audit_{run_id}.zip'
    zip_archive_path = os.path.join('reports', zip_filename)

    with zipfile.ZipFile(zip_archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(explanation_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to zip, preserving its relative path within explanation_dir
                zipf.write(file_path, os.path.relpath(file_path, explanation_dir))
    return zip_archive_path
```

Here's what happens:
1.  **Configuration Snapshot:** The `create_config_snapshot` function generates a `config_snapshot.json` file. This file records crucial metadata like model/data hashes, the random seed used, and the target column, ensuring that the environment and key parameters of the validation run are documented.
2.  **Evidence Manifest:** The `create_evidence_manifest` function iterates through all generated explanation files (global, local, counterfactual JSONs, summary report, and the config snapshot itself). For each file, it calculates its SHA-256 hash and stores this information in an `evidence_manifest.json`. This manifest provides cryptographic proof of the integrity and origin of every artifact.
3.  **Artifact Bundling:** The `bundle_artifacts_to_zip` function then collects all these files and compresses them into a single, timestamped ZIP archive.

After these operations, a "Download Audit ZIP File" button will appear, allowing you to download the complete audit package.

<button>
  [Download Audit ZIP File](https://www.example.com/your_audit_file.zip)
</button>

*(Note: The actual download link will be dynamic and provided by the Streamlit application itself.)*

### c. Explanation of Execution

The final stage of the validation workflow is complete. I have successfully generated a comprehensive set of explanation artifacts, including global and local SHAP analyses, counterfactual examples, and my detailed summary report. Each of these documents, along with a snapshot of the configuration (including model and data hashes) and a manifest of all files with their individual SHA-256 hashes, has been meticulously bundled into a timestamped ZIP archive: `PrimeCredit_CAM_v1.2_Validation_Audit_<run_id>.zip`.

This single, self-contained archive is PrimeCredit Bank's **audit-ready artifact bundle**. It ensures:
1.  **Reproducibility:** The `config_snapshot.json` captures all parameters needed to regenerate these explanations.
2.  **Traceability:** The `evidence_manifest.json` provides cryptographic proof of the integrity and origin of each artifact, linking them directly to the validated model and data versions.
3.  **Compliance:** All necessary documentation for internal auditors, regulators, and senior stakeholders is readily available and verifiable, significantly reducing regulatory risk and building trust in the AI system.

This completes my model validation task for CAM v1.2, providing PrimeCredit Bank with the necessary confidence to proceed with its deployment, knowing its decisions are explainable, transparent, and auditable.
