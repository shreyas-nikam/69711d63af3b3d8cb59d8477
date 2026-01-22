id: 69711d63af3b3d8cb59d8477_user_guide
summary: Lab 5: Interpretability & Explainability Control Workbench User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Model Interpretability & Explainability Control Workbench User Guide

## 1. Welcome to PrimeCredit Bank's Model Validation Workbench
Duration: 0:05

<aside class="positive">
This codelab is designed to walk you through the process of validating a machine learning model for transparency, fairness, and compliance using interpretability and explainability techniques. You'll take on the role of **Anya Sharma**, a Model Validator at PrimeCredit Bank, tasked with ensuring a new Credit Approval Model (CAM v1.2) is ready for deployment.
</aside>

As Anya Sharma, a dedicated Model Validator at PrimeCredit Bank, your primary responsibility is to ensure that all machine learning models used in critical business processes are transparent, fair, and compliant with internal governance and external regulatory standards.

Today, your focus is on a newly developed **Credit Approval Model (CAM v1.2)**, which will determine loan eligibility for our customers. Before this model can be deployed, you must rigorously assess its interpretability and explainability.

This workbench will guide you through:
*   **Understanding the Model's Overall Behavior**: What are the most important factors globally?
*   **Explaining Individual Decisions**: Why was a specific loan applicant approved or denied?
*   **Generating Actionable Insights**: What minimal changes would flip a denied decision to an approval?
*   **Synthesizing Findings**: Documenting interpretability gaps and recommendations.
*   **Bundling Audit-Ready Artifacts**: Ensuring reproducibility and compliance.

This application serves as your workbench to generate, analyze, and document the required explanations as audit-ready artifacts.

To begin your validation journey, navigate to the **'1. Upload & Configure'** page using the sidebar.

## 2. Setting the Stage: Environment Setup and Data Ingestion
Duration: 0:05

Your first step as Anya is to prepare your environment and load the necessary model and data for validation. Reproducibility is paramount in model validation; therefore, you will fix a random seed and compute SHA-256 hashes for both the model and dataset to ensure traceability and detect any unauthorized changes.

### Loading Model and Data

On this page, you have two options to load the Credit Approval Model and its corresponding feature dataset:

1.  **Upload Your Own Files**: If you have a `.pkl` or `.joblib` model file and a `.csv` dataset, you can upload them using the respective file uploaders. Once both are selected, click the **"Load Uploaded Files"** button.
2.  **Load Sample Data**: To quickly get started, you can use PrimeCredit Bank's provided sample model and data. Click the **"Load Sample Data"** button. This will automatically load a `sample_credit_model.pkl` and `sample_credit_data.csv`.

<aside class="positive">
For this codelab, we recommend using the **"Load Sample Data"** option to proceed quickly through the steps.
</aside>

Once the files are loaded, you will see a confirmation message and details about the loaded artifacts:

*   **Model File:** The name of your loaded model file.
*   **Model Hash (SHA-256):** A unique cryptographic hash for the model. This hash will change if the model file is altered in any way, ensuring its integrity.
*   **Data File:** The name of your loaded data file.
*   **Data Hash (SHA-256):** A unique cryptographic hash for the dataset, ensuring its integrity.
*   **Random Seed Used:** The random seed used for reproducibility in data splitting.
*   **Model type identified:** The type of machine learning model (e.g., `sklearn.ensemble.RandomForestClassifier`).
*   **Data features:** A list of all features in your dataset.

You will also see the first 5 rows of the feature data, allowing you to quickly inspect the data.

The initial setup is complete. You've successfully loaded the Credit Approval Model and the associated feature dataset. Crucially, you've generated cryptographic hashes for both artifacts. These hashes are vital for maintaining an immutable audit trail; any future change to either the model or the dataset would result in a different hash, immediately signaling a potential issue to an auditor. This step aligns with PrimeCredit Bank's stringent requirements for data and model integrity. The data has also been pre-processed, separating features from the target variable, making it ready for explanation generation.

## 3. Unveiling Overall Behavior: Global Model Explanations
Duration: 0:10

As a Model Validator, your first task is to grasp the overall behavior of the CAM v1.2. Which factors generally drive its decisions for approving or denying loans? Global explanations provide an aggregate view of feature importance, revealing which features have the most impact across all predictions. This helps you verify if the model's general logic aligns with PrimeCredit's lending policies and expert domain knowledge.

For tree-based models like our `RandomForestClassifier`, SHAP (SHapley Additive exPlanations) values are an excellent choice for this. The SHAP value $\phi_i$ for a feature $i$ represents the average marginal contribution of that feature value to the prediction across all possible coalitions of features.

The fundamental idea behind SHAP values is to attribute the prediction of an instance $x$ to its features by considering the contribution of each feature to moving the prediction from the base value (average prediction) to the current prediction. The sum of the SHAP values for all features and the base value equals the model's output for that instance:

$$ \phi_0 + \sum_{{i=1}}^{{M}} \phi_i(f, x) = f(x) $$

where $\phi_0$ is the expected model output (the base value), $M$ is the number of features, $\phi_i(f, x)$ is the SHAP value for feature $i$ for instance $x$, and $f(x)$ is the model's prediction for instance $x$.

### Generating Global Explanations

1.  Click the **"Generate Global Explanations"** button. This will calculate the SHAP values for a subset of your training data (to make the calculation efficient) and aggregate them to determine overall feature importance.
2.  Once generated, you will see two outputs:
    *   **Global Feature Importance Ranking**: A table showing features ranked by their average absolute SHAP value, indicating their overall importance.
    *   **SHAP Global Summary Plot**: A visualization (typically a bar plot) that summarizes the impact of each feature on the model's output across the dataset. The length of the bar indicates the magnitude of the feature's influence.

### Interpreting Global Explanations

The global SHAP explanation reveals the overall drivers of the Credit Approval Model. From the summary plot and the `global_importance_df`, you can clearly see which features, such as `credit_score` and `income`, are most influential in the model's decisions regarding loan approval. This high-level overview confirms that the model is largely relying on expected financial health indicators, which aligns with PrimeCredit Bank's lending criteria. This gives you initial confidence that the model's general behavior is sensible and explainable to senior stakeholders. However, global explanations only tell part of the story; you need to investigate specific individual decisions to ensure consistency and fairness.

## 4. Deep Dive into Individual Decisions: Local Explanations for Specific Loan Applications
Duration: 0:15

While global explanations are useful, they don't explain why a *specific* loan applicant was approved or denied. As a Model Validator, you frequently encounter requests to understand individual decisions, especially for denied applications or those with unusual profiles. For PrimeCredit Bank, it's crucial to provide clear, defensible reasons to customers for loan denials or approvals. You will select a few representative cases from our `sample_credit_data` to generate local explanations using SHAP. This allows you to examine the contribution of each feature to that particular prediction.

For a specific instance $x$, the SHAP values $\phi_i(f, x)$ quantify how much each feature $i$ contributes to the prediction $f(x)$ compared to the average prediction $\phi_0$. A positive SHAP value for a feature means it pushed the prediction higher (towards approval), while a negative value pushed it lower (towards denial).

### Selecting Instances and Generating Explanations

1.  **Select Instances for Local Explanation**: You will see a table of the first few rows of your dataset. Use the multi-select box to choose up to 3 `Instance IDs` from the full dataset. The application will pre-suggest some instances, often including a denied, an approved, and a borderline case if available.
2.  Click the **"Generate Local Explanations"** button. The application will calculate individual SHAP values for the selected instances.

### Interpreting Local Explanations

Once generated, you will see the **"Local Explanation Details"**. For each selected instance:

*   You can expand the section to view a JSON summary of the explanation, including the model's predicted probability for approval.
*   A **SHAP Waterfall Plot** will be displayed. This plot visually shows how each feature contributes to pushing the instance's prediction from the base value (average prediction) to its final prediction.
    *   Features shown in **red** are pushing the prediction *higher* (towards approval).
    *   Features shown in **blue** are pushing the prediction *lower* (towards denial).

For instance, analyzing the waterfall plot for a *denied* application, you can clearly see that a low `credit_score` and high `debt_to_income` ratio were the primary negative contributors, pushing the loan approval probability below the threshold. Conversely, for an *approved* application, a high `credit_score` and `income` might be the dominant positive factors. For a *borderline* case, the contributions might be more balanced.

These detailed breakdowns are invaluable for Anya. They allow her to:
1.  **Verify decision logic:** Are the model's specific reasons for a decision coherent and justifiable according to PrimeCredit's policy?
2.  **Identify potential biases:** Do certain demographic features (if present) disproportionately influence decisions in specific cases without valid business rationale? (Note: no demographic features are in this sample data, but this is what a Model Validator would look for).
3.  **Provide actionable feedback:** Understand what factors led to a denial, which is crucial for communicating with applicants.

This level of detail is exactly what PrimeCredit's internal auditors and potentially regulators would require to validate the fairness and transparency of the model.

## 5. "What If?": Understanding Counterfactuals for Actionable Insights
Duration: 0:10

For a denied loan applicant, merely knowing *why* they were denied (via local explanations) isn't always enough. As Anya, you also need to understand 'what if?' – what minimal changes to their application would have resulted in an approval? This is where counterfactual explanations come in. They identify the smallest, most actionable changes to an applicant's features that would flip the model's decision from denial to approval. This information is invaluable for PrimeCredit Bank, not only for providing constructive feedback to customers but also for potentially refining our lending criteria or identifying areas where applicants can improve their financial standing to become eligible.

The objective of generating a counterfactual example $x'$ for an original instance $x$ that results in a different prediction $y'$ is to minimize the distance between $x$ and $x'$, subject to the constraint that $x'$ belongs to the feasible input space $\mathcal{{X}}$ and the model $f$ predicts $y'$ for $x'$. This can be formalized as:

$$ \min_{{x'}} \text{{distance}}(x, x') \quad \text{{s.t.}} \quad f(x') = y' \quad \text{{and}} \quad x' \in \mathcal{{X}} $$

where $\text{{distance}}(x, x')$ is a measure of proximity (e.g., L1 or L2 norm), and $f(x')$ is the model's prediction for the counterfactual instance $x'$.

### Generating Counterfactual Examples

1.  **Select a Denied Instance for Counterfactual Generation**: The application will present a dropdown list of instances that were predicted as 'denied' (target = 0). Choose one of these instances.
2.  Click the **"Generate Counterfactual Example"** button. The application will search for a counterfactual example, aiming to change the prediction to 'approved' (target = 1) with minimal changes.

### Interpreting Counterfactual Analysis

Once a counterfactual is found, you will see the **"Counterfactual Analysis"** section, which includes:

*   **Original Instance:** The feature values of the initially denied application and its predicted approval probability.
*   **Counterfactual Instance:** The modified feature values that would lead to an approval, along with its predicted approval probability.
*   **Features Changed to Flip Prediction:** A list of features that were altered, along with their original and counterfactual values, and the magnitude of the change.

The counterfactual analysis provides invaluable 'what-if' scenarios for PrimeCredit Bank. For the selected denied loan application, the `counterfactual_result` clearly shows what minimal changes to an applicant's profile (e.g., increasing the `credit_score` by a certain amount or raising the `income` significantly) would have resulted in the loan being approved.

This empowers Anya to:
1.  **Inform customers:** Instead of just saying 'your loan was denied,' PrimeCredit can advise applicants on specific, actionable steps (e.g., 'If your credit score improved by X points, you would likely be approved').
2.  **Refine policy:** If generating counterfactuals consistently highlights specific features as critical for flipping decisions, it might indicate areas for policy review or for developing financial literacy programs for customers.
3.  **Assess model sensitivity:** It reveals how sensitive the model is to changes in specific features, which is a key part of model validation.

This concrete evidence of actionable insights is crucial for establishing trust and demonstrating the model's utility beyond just making a prediction.

## 6. Identifying Gaps: Interpretability Analysis and Validation Findings
Duration: 0:05

After reviewing the global, local, and counterfactual explanations, Anya must now synthesize her findings and identify any interpretability gaps that could prevent the CAM v1.2 from being approved for deployment. This is a critical step for PrimeCredit Bank's risk management framework. An interpretability gap might be a feature that, while statistically significant, lacks a clear business rationale, or cases where local explanations seem inconsistent. You need to document your observations, evaluate the model's transparency, and make a recommendation for its deployment or further refinement.

Your analysis will focus on:
*   **Coherence with Policy:** Do the explanations align with PrimeCredit's established lending policies and regulations?
*   **Transparency:** Are the reasons for decisions clear, concise, and easily understandable by non-technical stakeholders (e.g., loan officers, customers, auditors)?
*   **Consistency:** Do similar cases receive similar explanations, and are there any anomalous explanations?
*   **Actionability:** Do counterfactuals provide practical advice for applicants?

Based on these, you will formulate a summary of your findings and a recommendation.

### Generating the Validation Summary

1.  Click the **"Generate Validation Summary"** button. This will compile a report based on the global, local, and counterfactual explanations you've generated so far.

### Reviewing the Validation Summary Report

The `explanation_summary.md` document captures Anya's comprehensive analysis. It consolidates findings from global importance, local decision breakdowns, and counterfactual scenarios. Crucially, it identifies specific 'interpretability gaps' – areas where explanations might be less straightforward or require further context – such as opaque feature interactions or explanations for borderline cases. For each gap, Anya has provided a pragmatic recommendation for PrimeCredit Bank, demonstrating a proactive approach to risk management.

<aside class="negative">
<b>Note on Hashes in Summary:</b> Due to the design of the internal functions, the model and data hashes displayed directly within this markdown summary (`explanation_summary.md`) are derived from the initial load of `sample_credit_model.pkl` and `sample_credit_data.csv` when the `source.py` module is first imported. For the <b>actual</b> audit-ready hashes of your dynamically uploaded model and data, please refer to the `config_snapshot.json` within the exported artifact bundle (next step).
</aside>

By clearly documenting these observations and providing a recommendation (in this case, approval with caveats), Anya fulfills her role as a Model Validator. This structured summary serves as a primary artifact for review by the Internal Audit team and senior leadership, enabling them to make an informed decision on the CAM v1.2's production readiness with a full understanding of its explainability profile.

## 7. Audit Trail: Reproducibility and Artifact Bundling
Duration: 0:05

The final, critical step for Anya is to ensure that all her validation work is reproducible and securely bundled for auditing purposes. For PrimeCredit Bank, regulatory compliance demands an immutable record of all explanation artifacts, along with the configuration and hashes that guarantee their traceability to specific model and data versions. This 'audit-ready artifact bundle' acts as indisputable evidence of the model validation process. You will consolidate all generated explanations, configuration details, and an `evidence_manifest.json` containing SHA-256 hashes of each file, into a single, timestamped ZIP archive.

The `evidence_manifest.json` will list each generated file and its corresponding SHA-256 hash. The SHA-256 hash function takes an input (e.g., a file's content) and produces a fixed-size, 256-bit (32-byte) hexadecimal string. Even a minuscule change to the input will result in a completely different hash, making it an excellent tool for verifying data integrity:

$$ \text{{SHA-256}}(\text{{file\_content}}) = \text{{hexadecimal\_hash\_string}} $$

### Generating and Bundling Artifacts

1.  Click the **"Generate & Bundle All Audit Artifacts"** button. This will perform the following actions:
    *   Create a `config_snapshot.json` containing the model hash, data hash, and random seed used.
    *   Create an `evidence_manifest.json` which lists all generated explanation files (global, local, counterfactual, summary, and config snapshot) along with their individual SHA-256 hashes.
    *   Compress all these files into a single, timestamped ZIP archive.

### Downloading the Audit-Ready Artifact Bundle

Once the bundling process is complete, you will see a **"Download Audit-Ready Artifact Bundle"** button. Click this button to download the ZIP file to your local machine.

The final stage of the validation workflow is complete. You have successfully generated a comprehensive set of explanation artifacts, including global and local SHAP analyses, counterfactual examples, and your detailed summary report. Each of these documents, along with a snapshot of the configuration (including model and data hashes) and a manifest of all files with their individual SHA-256 hashes, has been meticulously bundled into a timestamped ZIP archive.

This single, self-contained archive is PrimeCredit Bank's **audit-ready artifact bundle**. It ensures:
1.  **Reproducibility:** The `config_snapshot.json` captures all parameters needed to regenerate these explanations, including the exact model and data hashes.
2.  **Traceability:** The `evidence_manifest.json` provides cryptographic proof of the integrity and origin of each artifact, linking them directly to the validated model and data versions.
3.  **Compliance:** All necessary documentation for internal auditors, regulators, and senior stakeholders is readily available and verifiable, significantly reducing regulatory risk and building trust in the AI system.

This completes your model validation task for CAM v1.2, providing PrimeCredit Bank with the necessary confidence to proceed with its deployment, knowing its decisions are explainable, transparent, and auditable.
