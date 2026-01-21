# QuLab: Lab 5: Interpretability & Explainability Control Workbench

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

This Streamlit application, "QuLab: Lab 5: Interpretability & Explainability Control Workbench," serves as a comprehensive tool for Model Validators to assess the transparency, fairness, and compliance of machine learning models. Built around a scenario involving Anya Sharma, a Model Validator at PrimeCredit Bank, the workbench focuses on vetting a **Credit Approval Model (CAM v1.2)**.

The primary goal is to provide a robust environment for generating, analyzing, and documenting various forms of model explanations (global, local, and counterfactual) to identify interpretability gaps, ensure alignment with lending policies, and produce audit-ready artifacts. It addresses critical needs for reproducibility, traceability, and regulatory compliance in AI/ML model deployment.

## Features

The application provides a structured workflow, guided by a navigation sidebar, to conduct a thorough model validation:

1.  **Welcome & Introduction**: An introductory page setting the context for the model validation mission at PrimeCredit Bank.
2.  **Setup & Data Ingestion**:
    *   Upload custom machine learning models (e.g., `.pkl`, `.joblib`) and corresponding datasets (`.csv`).
    *   Automated loading of artifacts and calculation of SHA-256 cryptographic hashes for the model and data to ensure integrity and traceability.
    *   Dynamic resetting of session state upon new file uploads for a fresh analysis.
    *   Display of loaded artifact summaries including model type, hashes, features, and target column.
3.  **Global Explanations**:
    *   Generate aggregate SHAP (SHapley Additive exPlanations) values to understand overall feature importance and drivers across all model predictions.
    *   Visualize global feature importance using a SHAP summary bar plot.
    *   Display a table summarizing global feature contributions.
4.  **Local Explanations**:
    *   Select specific instances from the dataset for detailed, individual prediction explanations.
    *   Generate local SHAP values to explain why a particular loan applicant was approved or denied.
    *   Visualize individual prediction explanations using SHAP waterfall plots, highlighting positive and negative feature contributions.
    *   Display feature details and model prediction probabilities for selected instances.
5.  **Counterfactual Explanations**:
    *   Select a denied loan application instance to explore "what if" scenarios.
    *   Generate counterfactual explanations using the DiCE (Diverse Counterfactual Explanations) library, identifying minimal changes to an applicant's features that would flip a loan denial to an approval.
    *   Present the original instance, counterfactual instance, and the minimal feature changes required.
6.  **Summary & Audit Trail**:
    *   Synthesize findings from global, local, and counterfactual analyses into a comprehensive markdown-based summary report.
    *   **Audit-Ready Artifact Bundling**: Automatically create an immutable audit trail by:
        *   Generating a `config_snapshot.json` containing key parameters, model hash, and data hash.
        *   Creating an `evidence_manifest.json` that lists all generated explanation artifacts along with their individual SHA-256 hashes.
        *   Bundling all reports, snapshots, and the manifest into a single, timestamped ZIP archive for secure storage and regulatory compliance.
    *   Download functionality for the generated audit ZIP file.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
    *(Replace `<repository_url>` and `<repository_name>` with your actual repository details)*

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is not provided, you would create one based on the imports. Here's a suggested `requirements.txt` based on the code:)*

    ```
    streamlit>=1.0
    pandas
    numpy
    scikit-learn
    matplotlib
    shap
    dill # Often used with joblib for more complex objects
    joblib # For saving/loading models
    dice-ml # For counterfactual explanations
    ```

## Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

2.  **Navigate through the Workbench**:
    *   **Welcome**: Read the introduction to the model validation mission.
    *   **Setup & Data**:
        *   Upload your pre-trained model (`.pkl` or `.joblib`) and corresponding data (`.csv`).
        *   Click "Load and Hash Artifacts" to process them.
        *   The application will also generate sample data and a model if none exist or are uploaded, ensuring functionality out-of-the-box for demonstration.
    *   **Global Explanations**: Click "Generate Global Explanations" to see overall feature importance.
    *   **Local Explanations**: Select instances from the dropdown and click "Generate Local Explanations" to get explanations for individual predictions.
    *   **Counterfactuals**: Select a denied instance and click "Generate Counterfactuals" to find out what minimal changes could flip the prediction.
    *   **Summary & Audit**:
        *   Click "Generate Explanation Summary Report" to compile all findings.
        *   Click "Export All Audit-Ready Artifacts" to bundle all generated reports, configurations, and hashes into a downloadable ZIP file.

## Project Structure

```
.
├── app.py                     # Main Streamlit application file
├── source.py                  # Contains core functions for data loading, model generation, and explanation logic
├── requirements.txt           # List of Python dependencies
├── reports/                   # Directory to store generated explanation reports and audit bundles
│   └── session_05_validation_run_<timestamp>/
│       ├── global_explanation.json
│       ├── local_explanation.json
│       ├── counterfactual_example.json
│       ├── explanation_summary.md
│       ├── config_snapshot.json
│       ├── evidence_manifest.json
│       └── Session_05_<timestamp>.zip
├── uploaded_files_temp/       # Temporary directory for uploaded model and data files
├── sample_credit_model.pkl    # (Generated) Example pre-trained model
└── sample_credit_data.csv     # (Generated) Example dataset
```

## Technology Stack

*   **Python**: Programming language
*   **Streamlit**: For building interactive web applications
*   **Pandas**: For data manipulation and analysis
*   **NumPy**: For numerical operations
*   **Scikit-learn**: For machine learning model building (e.g., `RandomForestClassifier`), and data utilities (`train_test_split`)
*   **SHAP (SHapley Additive exPlanations)**: A game-theoretic approach to explain the output of any machine learning model.
*   **DiCE (Diverse Counterfactual Explanations)**: For generating counterfactual explanations.
*   **Matplotlib**: For creating static, interactive, and animated visualizations.
*   **`dill` / `joblib`**: For efficient serialization and deserialization of Python objects, including trained models.
*   **`os`, `datetime`, `zipfile`, `hashlib`, `json`**: Standard Python libraries for file system operations, time management, archiving, hashing, and JSON handling.

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
*(Create a `LICENSE` file in your repository with the MIT License text)*

## Contact

For any questions or inquiries, please contact:

*   **QuantUniversity**
*   **Website**: [https://www.quantuniversity.com](https://www.quantuniversity.com)
*   **Email**: info@quantuniversity.com

---
