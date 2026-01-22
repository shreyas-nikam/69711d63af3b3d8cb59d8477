# QuLab: Lab 5: Interpretability & Explainability Control Workbench

![Streamlit App Screenshot Placeholder](https://via.placeholder.com/800x400?text=Streamlit+App+Screenshot)
*(Replace with an actual screenshot of the running application)*

## Table of Contents
1. [Project Title and Description](#1-project-title-and-description)
2. [Features](#2-features)
3. [Getting Started](#3-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Usage](#4-usage)
    - [Running the Application](#running-the-application)
    - [Workflow Guide](#workflow-guide)
5. [Project Structure](#5-project-structure)
6. [Technology Stack](#6-technology-stack)
7. [Contributing](#7-contributing)
8. [License](#8-license)
9. [Contact](#9-contact)

---

## 1. Project Title and Description

This repository hosts the Streamlit-based **QuLab: Lab 5: Interpretability & Explainability Control Workbench**. This application serves as a comprehensive tool for validating and explaining machine learning models, specifically demonstrated with a Credit Approval Model (CAM v1.2).

As a Lead Model Validator (Anya Sharma in this lab's persona), users can:
*   **Load and Verify** model and data artifacts, ensuring reproducibility through cryptographic hashing (SHA-256).
*   **Analyze Global Feature Importance** using SHAP (SHapley Additive exPlanations) to understand overall model behavior.
*   **Investigate Local Predictions** with SHAP waterfall plots for specific instances, elucidating individual decision drivers.
*   **Generate Actionable Counterfactuals** using DiCE (Diverse Counterfactual Explanations) for denied cases, providing feedback on what minimal changes would lead to a different outcome.
*   **Compile a Validation Summary Report** integrating all explainability findings.
*   **Export a Cryptographically Signed Audit Bundle** containing all artifacts, reports, and evidence for regulatory compliance and auditing purposes.

The workbench aims to enhance model transparency, fairness, and trust, crucial for responsible AI deployment in regulated industries.

## 2. Features

The application provides the following key functionalities:

*   **Secure Artifact Loading**:
    *   Upload custom `.pkl` model and `.csv` data files.
    *   Option to load a pre-configured sample credit model and data for easy demonstration.
    *   Automatic SHA-256 hash generation for both model and data to ensure version control and integrity.
    *   Preview of raw and preprocessed data.
*   **Global Model Interpretability**:
    *   Calculation and visualization of global feature importance using Mean Absolute SHAP values.
    *   Interactive SHAP summary plots to illustrate overall feature impact and distribution.
*   **Local Instance Explanations**:
    *   Selection of specific data instances for detailed analysis.
    *   Generation of SHAP waterfall plots for individual predictions, breaking down feature contributions.
    *   Display of both raw and preprocessed feature values for chosen instances.
*   **Counterfactual Analysis**:
    *   Identification and selection of "denied" instances.
    *   Generation of counterfactual explanations using DiCE, showing minimal changes needed for a desired outcome (e.g., approval).
    *   Provision of actionable feedback based on counterfactuals.
*   **Comprehensive Validation Summary**:
    *   Automatic generation of a markdown report summarizing global, local, and counterfactual findings.
    *   Includes model and data hashes, run ID, and timestamp for traceability.
*   **Audit-Ready Artifact Export**:
    *   Bundling of all generated explanations (JSON, PNG plots), configuration snapshots, and the summary report into a single ZIP archive.
    *   Creation of an `evidence_manifest.json` file within the bundle, listing all included files and their SHA-256 hashes, ensuring the integrity and completeness of the audit package.
    *   Downloadable ZIP archive for easy sharing and archival.
*   **Interactive User Interface**:
    *   Built with Streamlit for an intuitive and step-by-step validation workflow.
    *   Clear navigation through different stages of the validation process.

## 3. Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python**: Version 3.8 or higher.
*   **pip**: Python package installer (usually comes with Python).
*   **git**: Version control system (optional, but recommended for cloning the repository).

### Installation

1.  **Clone the Repository (Optional, if you have the files already):**
    ```bash
    git clone https://github.com/your_username/qu_lab5_explainability.git
    cd qu_lab5_explainability
    ```
    *(If you just have the `app.py` file, place it in a new directory.)*

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    Create a `requirements.txt` file in the same directory as `app.py` with the following content:
    ```
    streamlit
    pandas
    numpy
    joblib
    scikit-learn
    matplotlib
    shap
    dice-ml
    ```
    Then, install the packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `dice-ml` is specified with a hyphen as per PyPI package name, while `import dice_ml` uses an underscore.)*

## 4. Usage

### Running the Application

Once the dependencies are installed and your virtual environment is active, navigate to the directory containing `app.py` and run the Streamlit application:

```bash
streamlit run app.py
```

This command will open the Streamlit application in your default web browser. If it doesn't open automatically, look for a URL (usually `http://localhost:8501`) in your terminal.

### Workflow Guide

The application guides you through a step-by-step validation process using a sidebar navigation. Follow the numbered pages:

1.  **Home**: Overview of the project and workflow.
2.  **1. Data & Model Loading**:
    *   Choose to **Upload Custom Model & Data** (provide a `.pkl` model and a `.csv` feature file).
    *   Alternatively, click **Load Sample Credit Model & Data** to generate and load a synthetic dataset and a pre-trained `RandomForestClassifier` pipeline.
    *   Verify the SHA-256 hashes of the loaded artifacts.
3.  **2. Global Explanations**:
    *   Click "Generate Global Explanations" to compute overall feature importance using SHAP values.
    *   View the top features and a SHAP summary plot.
4.  **3. Local Explanations**:
    *   Select up to 3 instance IDs from the loaded dataset. The app provides default suggestions for denied, approved, and borderline cases.
    *   Click "Generate Local Explanations" to produce SHAP waterfall plots for each selected instance.
    *   Analyze how individual features contribute to each specific prediction.
5.  **4. Counterfactuals**:
    *   Select a "denied" instance ID from the dropdown.
    *   Click "Generate Counterfactual Example" to find what minimal changes to the input features would flip the prediction to "approved".
    *   Review the original and counterfactual instance details, along with actionable feedback.
6.  **5. Validation Summary**:
    *   Click "Generate Explanation Summary" to compile all findings (global, local, counterfactual) into a single Markdown report.
    *   This summary provides a comprehensive overview of your validation session.
7.  **6. Export Artifacts**:
    *   Click "Export Audit-Ready Bundle (.zip)" to package all generated reports, plots, configuration snapshots, and a manifest file into a downloadable ZIP archive.
    *   The manifest includes SHA-256 hashes for all bundled files, ensuring integrity for auditing.

## 5. Project Structure

```
.
├── app.py                      # Main Streamlit application script
├── requirements.txt            # Python dependencies for the project
├── temp_uploads/               # Directory to temporarily store uploaded model/data files
│   └── (uploaded_files.pkl)
│   └── (uploaded_files.csv)
└── reports/                    # Directory for storing validation session reports and audit bundles
    ├── validation_run_<timestamp>/   # Subdirectory for each validation session
    │   ├── config_snapshot.json    # Snapshot of model/data hashes and random seed
    │   ├── global_explanation.json # Global SHAP importance data
    │   ├── local_explanation.json  # Local SHAP data for selected instances
    │   ├── local_explanation_instance_<id>.png # SHAP waterfall plots for local instances
    │   ├── counterfactual_example.json # DiCE counterfactual result
    │   ├── explanation_summary.md  # Markdown summary of all explanations
    │   └── evidence_manifest.json  # Manifest with hashes of all files in this directory
    └── validation_run_<timestamp>_audit_bundle.zip # The final downloadable audit package
```

*(Note: `credit_model_v1.2.pkl` and `credit_data_validation.csv` might be generated directly in the root or `temp_uploads` if using the sample data/model functionality.)*

## 6. Technology Stack

*   **Application Framework**: [Streamlit](https://streamlit.io/)
*   **Data Manipulation**: [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (RandomForestClassifier, LabelEncoder, ColumnTransformer, Pipeline), [joblib](https://joblib.readthedocs.io/en/latest/)
*   **Explainable AI (XAI)**:
    *   [SHAP (SHapley Additive exPlanations)](https://github.com/shap/shap)
    *   [DiCE (Diverse Counterfactual Explanations)](https://github.com/microsoft/DiCE)
*   **Plotting**: [Matplotlib](https://matplotlib.org/)
*   **Utilities**: `os`, `hashlib`, `json`, `zipfile`, `datetime`, `shutil`, `tempfile` (standard Python libraries)

## 7. Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

Please ensure your code adheres to good coding practices and includes relevant tests.

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Create a LICENSE file in your repository if you don't have one already)*

## 9. Contact

For any questions or inquiries, please contact:

*   **QuantUniversity** - [info@quantuniversity.com](mailto:info@quantuniversity.com)
*   **Project Link**: [https://github.com/your_username/qu_lab5_explainability](https://github.com/your_username/qu_lab5_explainability) *(Replace with your actual repo link)*