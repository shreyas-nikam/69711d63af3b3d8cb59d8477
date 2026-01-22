id: 69711d63af3b3d8cb59d8477_user_guide
summary: Lab 5: Interpretability & Explainability Control Workbench User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Model Interpretability & Explainability Workbench Codelab

## 1. Introduction: Your Mission as a Model Validator
Duration: 0:05

Welcome to the **QuLab: Model Interpretability & Explainability Control Workbench**! This codelab is designed to guide you through the process of validating and understanding a machine learning model's behavior using advanced explainability techniques.

<aside class="positive">
<b>Important Context:</b> You are Anya Sharma, the Lead Model Validator at PrimeCredit Bank. Your critical mission is to evaluate the new Credit Approval Model (CAM v1.2) for transparency, fairness, and compliance before it is deployed. This workbench is your primary tool for this task.
</aside>

This application will walk you through a structured workflow to:
1.  **Load Data & Model**: Establish a verifiable starting point.
2.  **Global Explanations**: Understand overall model behavior.
3.  **Local Explanations**: Investigate specific individual predictions.
4.  **Counterfactuals**: Provide actionable feedback for denied applications.
5.  **Validation Summary**: Compile your findings into a comprehensive report.
6.  **Export Artifacts**: Create a secure, auditable package of your validation evidence.

Let's begin by setting up your validation environment. Navigate to the "1. Data & Model Loading" section in the sidebar to start.

## 2. Setting Up Your Validation Environment: Data & Model Loading
Duration: 0:10

The first step in any robust model validation is to ensure that the exact model and data used for evaluation can be reproduced and traced. This application facilitates this by generating **cryptographic hashes (SHA-256)** for your loaded artifacts. A hash is like a unique digital fingerprint; if even a single byte of the file changes, its hash will be completely different, ensuring the integrity and authenticity of your model and data.

The "1. Data & Model Loading" page offers two primary ways to load your artifacts:

### Custom Upload

This option allows you to upload your own machine learning model (as a `.pkl` file) and its corresponding feature data (as a `.csv` file). This is ideal for validating new or updated models in your specific environment.

1.  Click **"Upload Model (.pkl)"** and select your model file.
2.  Click **"Upload Feature Data (.csv)"** and select your dataset.
3.  Click the **"Load Custom Model & Data"** button. The application will then load these files, preprocess the data as required by the model's pipeline, and calculate their SHA-256 hashes.

### Sample Environment

For this codelab, we recommend starting with the "Sample Environment". This pre-generates a dummy credit approval model (`Credit Approval Model v1.2`) and a synthetic validation dataset, allowing you to immediately dive into the explainability features without needing to prepare your own files.

1.  Click the **"Load Sample Credit Model & Data"** button.

<aside class="positive">
<b>Tip:</b> If you're running this locally, the application will create `credit_model_v1.2.pkl` and `credit_data_validation.csv` in your application directory. These are the files it uses for the sample environment.
</aside>

Once the loading process completes, you will see:

*   **Model SHA-256 Hash** and **Data SHA-256 Hash**: These unique identifiers confirm which exact versions of the model and data are being used.
*   **Feature Data Preview**: This shows the first few rows of your data *after* it has been preprocessed by the model's pipeline, ready for predictions and explanations.
*   **Raw Data Preview**: This displays the original, raw data with all features, including the target column and any categorical features, as they were before preprocessing.

<aside class="info">
The application handles the necessary data preprocessing (like converting categorical features to numerical ones) internally, mimicking how a real-world model pipeline would operate. This means you don't need to manually preprocess your data before uploading; the application's internal pipeline takes care of it.
</aside>

## 3. Understanding Global Model Behavior: Global Explanations
Duration: 0:15

After loading your model and data, the next logical step is to understand what drives the model's decisions at a high level. **Global explanations** provide an aggregated view of how different features influence the model's predictions across the entire dataset.

This application uses **SHAP (SHapley Additive exPlanations)** values for global interpretability. SHAP values attribute the prediction of an instance to each feature by calculating how much each feature contributes to pushing the prediction from the average prediction to the actual prediction.

The fundamental equation for SHAP is:
$$ \phi_0 + \sum_{i=1}^{M} \phi_i(f, x) = f(x) $$
Here, $f(x)$ is the model's prediction for an instance $x$, $\phi_0$ is the average prediction (or baseline), and $\phi_i(f, x)$ represents the contribution of feature $i$ to the prediction for that specific instance. When we look at global explanations, we aggregate these contributions.

1.  Navigate to the "2. Global Explanations" page.
2.  Click the **"Generate Global Explanations"** button. This will trigger the calculation of SHAP values for a sample of your training data (specifically the `X_train_exp` in the code, which is 80% of your preprocessed features), which is used as a background dataset for global explanations.

Once the process is complete, you will see two key components:

### Top Features by Mean |SHAP| Value

This table lists the features that have the most significant average impact on the model's output, regardless of the direction (positive or negative). A higher mean absolute SHAP value indicates a more important feature for the model's overall decision-making.

### SHAP Summary Plot

This is a powerful visualization that shows how each feature influences the model's output:
*   **Vertical Axis**: Features are ordered by their importance (from top to bottom).
*   **Horizontal Axis**: Represents the SHAP value for a given feature.
*   **Color**: Indicates the feature value (e.g., red for high values, blue for low values).
*   **Dots**: Each dot represents a single instance from the dataset.

By observing this plot, you can identify:
*   Which features are most impactful.
*   Whether high or low values of a feature tend to push the prediction higher or lower. For example, if "Income" is a top feature, and red dots (high income) are clustered on the positive SHAP side, it implies higher income generally increases the probability of approval.

<aside class="info">
The "positive class index" refers to the index in the model's output that corresponds to the "approved" class (or whatever is considered the positive outcome). For binary classification with classes 0 and 1, it's typically index 1.
</aside>

## 4. Investigating Individual Predictions: Local Explanations
Duration: 0:15

While global explanations provide an overall understanding, **local explanations** allow you to delve into the specific reasons behind an individual prediction. This is crucial for auditing specific loan applications or understanding why a particular customer was approved or denied.

On the "3. Local Explanations" page, you can select up to 3 individual instances (loan applications) from your dataset to analyze in detail. The application will pre-select some interesting cases for you, such as a denied instance, an approved instance, and a borderline case, to demonstrate the diverse insights local explanations can offer.

The SHAP values, introduced in the previous step, are even more insightful here. For a specific instance $x$, the SHAP values $\phi_i(f, x)$ quantify how much each feature $i$ contributes to the prediction $f(x)$ compared to the average prediction $\phi_0$. A positive SHAP value for a feature means it pushed the prediction higher (towards approval), while a negative value pushed it lower (towards denial).

1.  Navigate to the "3. Local Explanations" page.
2.  Use the multiselect box to **"Select up to 3 instance IDs to explain"**. You can use the default selections or choose your own.
3.  Click the **"Generate Local Explanations"** button.

For each selected instance, the application will display:

### Prediction Probability and Status

This metric clearly shows the model's predicted probability for the positive class (e.g., approval) and its corresponding decision (Approved/Denied, based on a 0.5 threshold).

### Feature Values

You'll see two tables:
*   **Feature Values (Preprocessed for Model)**: These are the numerical values of the features *after* they have gone through the model's preprocessing steps. This is what the model directly "sees."
*   **Raw Feature Values**: These are the original, human-readable values of the features for that instance. This helps you relate the preprocessed values back to the original data.

### Contribution Waterfall Plot

This powerful visualization breaks down the prediction for a single instance:
*   **Base Value**: The average prediction across the entire dataset.
*   **Feature Contributions**: Each bar shows how much a specific feature (and its value for this instance) increased or decreased the prediction from the Base Value. Red bars indicate features pushing the prediction higher (towards approval), while blue bars indicate features pushing it lower (towards denial).
*   **f(x) (Model Output)**: The final prediction for this specific instance.

By examining the waterfall plot, you can clearly see which features were the most influential in shaping the model's decision for that particular individual. For example, a low `CreditScore` (blue bar pushing left) might be a strong reason for a denial, while a high `Income` (red bar pushing right) might contribute to an approval.

## 5. Providing Actionable Feedback: Counterfactual Analysis
Duration: 0:10

One of the most valuable aspects of explainability is providing actionable advice. **Counterfactual explanations** address the question: "What is the smallest change to an instance's features that would flip the model's prediction to a desired outcome?" For a denied loan applicant, this means identifying the minimal changes that would lead to an approval.

The core idea is to find an alternative instance $x'$ that is very similar to the original instance $x$ but results in the desired prediction $y'$. Mathematically, this can be formulated as an optimization problem:

$$ \min_{x'} \text{distance}(x, x') \quad \text{s.t.} \quad f(x') = y' \quad \text{and} \quad x' \in \mathcal{X} $$

Here, $\text{distance}(x, x')$ measures how "far" the counterfactual $x'$ is from the original $x$ (representing the effort to make changes), $f(x')$ is the model's prediction for the counterfactual, $y'$ is the desired prediction (e.g., "approved"), and $x' \in \mathcal{X}$ ensures the counterfactual is a valid, realistic instance (e.g., age cannot be negative).

This application uses the **DiCE (Diverse Counterfactual Explanations)** library to generate these insights.

1.  Navigate to the "4. Counterfactuals" page.
2.  The application will automatically identify instances that were *denied* by the model. Select one of these denied instance IDs from the dropdown menu.
3.  Click the **"Generate Counterfactual Example"** button.

<aside class="negative">
<b>Warning:</b> If the `dice_ml` library is not installed in your environment, the application will provide a dummy counterfactual example. To generate real counterfactuals, ensure `dice_ml` is installed (`pip install dice_ml`).
</aside>

The results will show:

### Original Instance (Denied - Raw Features)

This table displays the raw, original feature values for the selected denied instance, giving you a clear picture of the applicant's profile.

### Counterfactual Instance (Approved - Raw Features)

This table presents the minimal changes suggested by DiCE that would have resulted in an approval. You'll see which features (e.g., `Income`, `CreditScore`, `LoanAmount`, `EmploymentType`) would need to be adjusted and to what values.

### Actionable Feedback

This concise summary highlights the specific changes required for the counterfactual. This text is crucial for providing direct, understandable guidance to applicants or for informing policy adjustments. For example, it might suggest: "To get approved, consider: increase Income by X; decrease LoanAmount by Y; change EmploymentType from 'Unemployed' to 'Salaried'."

## 6. Synthesizing Findings: Validation Summary
Duration: 0:05

After conducting your global, local, and counterfactual analyses, it's essential to synthesize these findings into a comprehensive **Validation & Interpretability Report**. This report serves as a formal record of your audit, providing a narrative summary of the model's behavior and the insights gained.

1.  Navigate to the "5. Validation Summary" page.
2.  Click the **"Generate Explanation Summary"** button.
    <aside class="negative">
    <b>Note:</b> You must have generated both Global and Local Explanations in the previous steps for the summary to be available. Counterfactuals are optional but recommended for a complete report.
    </aside>

The application will compile a markdown-formatted report containing:

*   **Run ID, Date, Model Hash, Data Hash**: Essential metadata for traceability.
*   **Global Feature Importance**: A summary of the top features influencing the model overall, typically including a table of mean absolute SHAP values.
*   **Local Instance Explanations**: Details for each instance you selected for local analysis, including its predicted probability, decision, key feature contributions (SHAP values), and a note about the accompanying waterfall plot image.
*   **Counterfactual Analysis**: If generated, this section will detail the original denied instance, its counterfactual (approved) counterpart, and the actionable feedback.

Review the generated summary report on this page. This report is a crucial deliverable for regulatory bodies or internal stakeholders.

## 7. Securing Your Evidence: Exporting Audit Artifacts
Duration: 0:05

The final step in your validation workflow is to create a secure, auditable package of all your findings. This **Audit-Ready Bundle (.zip)** ensures that all evidence, configuration snapshots, and explanation reports are cryptographically signed and stored together for future reference and compliance checks.

The integrity of this bundle is assured by using **SHA-256 hashes**:

$$ \text{SHA-256}(\text{file\_content}) = \text{hexadecimal\_hash\_string} $$

This means that any alteration to the files within the bundle would result in a different hash, making tampering immediately detectable.

1.  Navigate to the "6. Export Artifacts" page.
2.  Click the **"Export Audit-Ready Bundle (.zip)"** button.
    <aside class="negative">
    <b>Note:</b> This button will only be enabled after you have successfully generated the "Validation Summary" in the previous step.
    </aside>

The application will perform the following actions:

*   **Create Config Snapshot**: A JSON file containing the model hash, data hash, random seed used, and a timestamp. This captures the exact environment of your validation run.
*   **Create Evidence Manifest**: A JSON file listing all generated explanation files (global, local, counterfactual JSONs, local SHAP waterfall plots, the summary report, and the config snapshot) along with their individual SHA-256 hashes. This manifest acts as a table of contents and integrity checker for your audit package.
*   **Bundle to ZIP**: All these files (explanation JSONs, plots, summary markdown, config snapshot, and the manifest) will be compressed into a single ZIP archive.

Once complete, you will see a success message indicating the manifest and ZIP archive have been created. You will then be provided with a **"Download Audit Bundle (.zip)"** button. Click this button to download your complete audit package.

This `.zip` file represents a comprehensive, traceable record of your model validation, ready for submission or archival.

Congratulations! You have successfully completed the validation workflow using the QuLab: Model Interpretability & Explainability Control Workbench.
