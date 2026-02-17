Lab 5 — Student Guide: Interpretability & Explainability Control Workbench

Case Context
ClearLend is a fast-growing fintech lending startup using a statistical credit model to approve small-business loans. Regulators and auditors require that high-impact lending decisions be explainable, reproducible, and tied to a specific model version.

Your role
You are the model validator at ClearLend. Your task is to use the Streamlit workbench to inspect model drivers, explain a single applicant’s decision, produce a basic counterfactual, and package reproducible evidence for audits.

What you will do
- Select a baseline model and decision threshold in the sidebar.
- Inspect global explanations (SHAP or permutation importance).
- Inspect local explanations (SHAP and LIME) for one applicant.
- Generate a counterfactual to flip a denial to approval.
- Export an evidence bundle with artifacts and hashes.

Step-by-step instructions
1. Open the app and set context
   1.1 Use the sidebar to pick a baseline model (Logistic Regression or Random Forest).
   1.2 Set the decision threshold (default risk cutoff).
2. Data & Model
   2.1 Go to “Data & Model” to preview dataset rows and model snapshot.
   2.2 Note model family (tree/linear/black-box) and test ROC-AUC.
3. Global Explanation
   3.1 Open “Global Explanation”. If SHAP is available you’ll see SHAP importances, otherwise permutation importance.
   3.2 View the top drivers and run the stability check (choose repeats and click to run).
4. Local Explanation (SHAP & LIME)
   4.1 For SHAP: open the SHAP page, enter an applicant row index, view predicted risk, SHAP contributions, and the waterfall plot.
   4.2 For LIME: open the LIME page, pick the same index, click “Run LIME explanation”, then review the list and HTML surrogate output.
5. Counterfactual
   5.1 Open “Counterfactual”, choose ranking strategy (SHAP preferred), pick the same applicant, and click “Generate counterfactual”.
   5.2 Inspect suggested feature changes, status (flipped/not flipped), and feasibility concerns.
6. Reproducibility & Export
   6.1 Visit “Reproducibility & Evidence” to view model/data hashes and config snapshot.
   6.2 On “Export Bundle”, generate GLOBAL / LOCAL / COUNTERFACTUAL artifacts, write a short explanation summary, and click “Export evidence bundle” to download the zip.

What this lab is really teaching
- How to interpret global model drivers and check their stability.
- How to explain individual decisions with SHAP and LIME.
- How to generate simple, constraint-aware counterfactuals for recourse.
- How to capture reproducible artifacts (model/data hashes, snapshots) for audit evidence.

Discussion
- Do the top global drivers match business intuition? If not, what investigations would you start?
- Are the counterfactual changes realistic and actionable for an applicant?
- How would you operationalize these artifacts in a production validation workflow?

Takeaway
This lab shows how explainability tools and reproducibility controls turn visualizations into auditable evidence for deployment decisions.
