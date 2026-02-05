# Lab 5 Usage Guide

## Quick Start

### 1. Installation
```bash
# Navigate to project directory
cd lab5_interpretability

# Run setup script (Linux/Mac)
./setup.sh

# Or manually install
pip install -r requirements.txt
```

### 2. Launch Application
```bash
streamlit run app.py
```

### 3. Access Interface
Open browser to: `http://localhost:8501`

## Detailed Walkthrough

### Step-by-Step Tutorial

#### Step 0: Overview (5 minutes)
**Goal**: Understand lab objectives and enterprise context

1. Navigate to "🏠 Overview" in sidebar
2. Review the three main personas
3. Explore the canonical use cases (tabs)
4. Understand what makes explanations "audit-ready"

**Key Takeaway**: This is about enterprise controls, not just visualization.

---

#### Step 1: Upload Model & Data (3 minutes)
**Goal**: Load model and dataset with version tracking

**Using Sample Data** (Recommended for first run):
1. Navigate to "📤 Step 1: Upload Model & Data"
2. Check "Use sample credit model" ✓
3. Click "Load Sample Model"
4. Check "Use sample credit data" ✓
5. Click "Load Sample Data"
6. Note the hashes displayed (SHA-256)

**Using Your Own Data**:
1. Uncheck sample options
2. Upload your .pkl or .joblib model file
3. Upload your .csv dataset
4. Select target column from dropdown
5. Verify data preview table

**What's Happening**:
- System computes cryptographic hash of model
- System computes hash of dataset
- These hashes enable reproducibility verification

**Success Indicator**: Green message "Ready for configuration! Proceed to Step 2."

---

#### Step 2: Configure Explanation (5 minutes)
**Goal**: Select appropriate explanation methods

1. Navigate to "⚙️ Step 2: Configure Explanation"
2. Review auto-detected model type
3. Select risk tier:
   - **Tier 1 (High Risk)**: Full explanations + counterfactuals required
   - **Tier 2 (Medium Risk)**: Global + sample local explanations
   - **Tier 3 (Low Risk)**: Global sufficient
4. Choose explanation method:
   - **"Auto (Recommended)"**: Let system decide (best for most cases)
   - **"SHAP"**: Force SHAP if available
   - **"Permutation Importance"**: Simpler, model-agnostic
   - **"Coefficients"**: For linear models

**Decision Guide**:
- Tree models → SHAP TreeExplainer (fast, accurate)
- Linear models → Coefficients first, then SHAP
- Black-box models → Permutation importance
- When in doubt → Use "Auto"

**Success Indicator**: Green message "Configuration complete! Proceed to Step 3."

---

#### Step 3: Global Explanation (5 minutes)
**Goal**: Understand overall feature importance

1. Navigate to "🌍 Step 3: Global Explanation"
2. Click "🔍 Generate Global Explanation" button
3. Wait for processing (10-30 seconds depending on data size)
4. Review results:
   - **Importance table**: Features ranked by impact
   - **Bar chart**: Visual comparison
   - **Method used**: Confirms chosen approach

**Interpretation Tips**:
- Top 3-5 features typically drive most predictions
- Look for unexpected high-importance features
- Validate against domain knowledge
- Check for data leakage if importance seems "too good"

**Red Flags**:
- ❌ ID fields having high importance → data leakage
- ❌ Features that shouldn't matter ranking high → model issue
- ❌ All features having similar importance → model not learning

**What to Report**:
- Top 5 features and their importance scores
- Method used for generation
- Any validation concerns

**Success Indicator**: Green message "Global explanation complete! Proceed to Step 4."

---

#### Step 4: Local Explanation (7 minutes)
**Goal**: Explain individual predictions

1. Navigate to "🎯 Step 4: Local Explanation"
2. Use slider to select instance (0 to N-1)
3. Review instance details in table
4. Check current model prediction
5. Click "🔍 Generate Local Explanation"
6. Review feature contributions:
   - **Positive contributions**: Push prediction higher
   - **Negative contributions**: Push prediction lower
   - **Magnitude**: Strength of influence

**Example Analysis** (Credit Approval):
```
Instance #23: Loan Denied
- Base prediction: 0.45 (average approval rate)
- credit_score (+0.15): Good score helps
- income (+0.08): Adequate income helps
- debt_ratio (-0.35): High debt hurts significantly
- employment_years (+0.02): Stable employment helps slightly
→ Final prediction: 0.35 (Denied)
```

**Key Questions to Answer**:
1. What's the base value (average prediction)?
2. Which features contributed most?
3. Do contributions align with domain expertise?
4. Are there surprises?

**For Auditors**:
- Contributions are additive from base value
- SHAP values satisfy mathematical consistency
- Same instance always gives same explanation (reproducibility)

**Success Indicator**: Green message "Local explanation complete! Proceed to Step 5."

---

#### Step 5: Counterfactual Analysis (8 minutes)
**Goal**: Identify changes needed for different outcomes

1. Navigate to "🔄 Step 5: Counterfactual Analysis"
2. Current instance shown (from Step 4)
3. Set target prediction (e.g., 1 for "approved")
4. Click "🔄 Generate Counterfactual"
5. Review results:
   - Success status
   - Required changes
   - Minimal modifications identified

**Example Counterfactual** (Credit Approval):
```
Original Prediction: Denied (0)
Target Prediction: Approved (1)
Changes Required:
- Increase credit_score by 50 points (650 → 700)
- Decrease debt_ratio by 0.15 (0.45 → 0.30)
Result: APPROVED ✓
```

**Business Value**:
- **Actionable insights**: "To get approved, improve credit score to 700"
- **Fairness check**: Are changes realistic and non-discriminatory?
- **Decision boundaries**: Understand model sensitivity

**Regulatory Use**:
- Adverse action notices: "Here's what to improve"
- Right to explanation: Clear, actionable guidance
- Bias detection: Check if changes are discriminatory

**Limitations**:
- Current implementation uses greedy search
- May not find global minimum change
- Constraints not enforced (e.g., age can't decrease)

**Success Indicator**: Message showing whether counterfactual was found successfully.

---

#### Step 6: Export Artifacts (5 minutes)
**Goal**: Create audit-ready documentation bundle

1. Navigate to "📦 Step 6: Export Artifacts"
2. Review artifact checklist (required vs optional)
3. Optional: Check "📄 Preview Explanation Summary"
   - Review markdown document
   - Check completeness
4. Click "📦 Generate Export Bundle"
5. Click "⬇️ Download Session_05_[timestamp].zip"

**Bundle Contents**:
```
Session_05_20260204_143022.zip
├── global_explanation.json
│   └── Feature importance scores
├── local_explanation.json
│   └── Per-feature contributions
├── counterfactual_example.json
│   └── Required changes for different outcome
├── explanation_summary.md
│   └── Human-readable report
├── config_snapshot.json
│   └── Risk tier, method, model type
└── evidence_manifest.json
    └── SHA-256 hashes of all files
```

**Using the Artifacts**:
- **Auditors**: Verify hashes, review explanations
- **Regulators**: Provide explanation_summary.md
- **Stakeholders**: Share summary for transparency
- **Technical reviewers**: Inspect JSON for details

**Reproducibility Verification**:
1. Re-run analysis with same model + data
2. Generate new bundle
3. Compare evidence_manifest.json hashes
4. Should be identical (deterministic)

**Success Indicator**: ZIP file downloaded successfully.

---

#### Step 7: Validation Review (10 minutes)
**Goal**: Confirm enterprise approval readiness

1. Navigate to "📊 Step 7: Validation Review"
2. Review acceptance criteria checklist:
   - ✅ Explanations generated successfully
   - ✅ Method selection logic correct
   - ✅ Outputs reproducible across runs
   - ✅ All artifacts exported with hashes
   - ✅ Explanation ties to model version

3. **If all criteria pass**:
   - See success message with balloons 🎉
   - Lab complete, artifacts audit-ready

4. **If criteria fail**:
   - Review which steps need completion
   - Go back and complete missing items
   - Return to Step 7 for re-validation

**Integration Checklist**:
- [ ] Upstream: Have Labs 1-4 outputs available?
- [ ] Downstream: Ready to feed Lab 6 (robustness)?
- [ ] Documentation: Saved to shared drive?
- [ ] Versioning: Model hash recorded in registry?

**Final Deliverables**:
- ZIP bundle uploaded to artifact repository
- Summary report shared with stakeholders
- Validation checklist completed in project tracker
- Integration with validation documentation (Lab 10)

---

## Advanced Features

### Custom Models
To use your own model:
1. Save model: `joblib.dump(model, 'my_model.joblib')`
2. Ensure scikit-learn compatibility
3. Upload in Step 1

### Custom Data
Your CSV should have:
- Feature columns (numeric preferred)
- Target column (for global explanations)
- No missing values (clean beforehand)
- Reasonable size (<10,000 rows for SHAP)

### Reproducibility Testing
To verify reproducibility:
1. Complete full workflow
2. Download bundle #1
3. Restart application (close and reopen)
4. Re-load same model and data
5. Complete workflow again
6. Download bundle #2
7. Compare `evidence_manifest.json` files
8. Hashes should match exactly

---

## Troubleshooting

### "SHAP is not available"
**Cause**: SHAP not installed or import failed

**Fix**:
```bash
pip install shap --upgrade
# If still fails:
pip install shap --no-cache-dir
```

**Workaround**: Use "Permutation Importance" method instead

---

### "Model loading failed"
**Cause**: Model not compatible with current environment

**Fixes**:
1. Check Python version matches training environment
2. Install required libraries: `pip install scikit-learn`
3. Verify model was saved with pickle or joblib
4. Try re-training model in current environment

---

### "Explanation generation taking too long"
**Cause**: Large dataset or complex model

**Solutions**:
1. Use smaller sample of data (first 1000 rows)
2. Switch to "Permutation Importance" (faster)
3. For local explanations, don't generate on all instances
4. Use TreeExplainer for tree models (much faster than KernelExplainer)

---

### "Counterfactual not found"
**Cause**: Simple greedy search may not always succeed

**This is OK**: Not finding a counterfactual is informative
- May indicate prediction is very stable
- May indicate decision boundary is far away
- Document as "no simple counterfactual found"

**Alternative**: Use local explanation to identify most influential features

---

## Best Practices

### For Model Validators
1. Always start with global explanation
2. Spot-check 3-5 local explanations
3. Look for consistency across instances
4. Validate against domain knowledge
5. Document any concerns immediately

### For Internal Auditors
1. Verify hashes before review
2. Check timestamp and run_id
3. Ensure all required artifacts present
4. Validate reproducibility on sample
5. Archive bundle with unique identifier

### For ML Engineers
1. Run explanations during development
2. Use to debug unexpected behavior
3. Document methodology in model card
4. Automate explanation generation in CI/CD
5. Include bundle in model package

---

## FAQ

**Q: Can I skip counterfactual analysis?**
A: For Tier 3 models, yes. For Tier 1, it's mandatory. For Tier 2, recommended but not required.

**Q: How long are artifacts valid?**
A: As long as model version doesn't change. Re-generate if model is retrained or data changes.

**Q: Can explanations change between runs?**
A: No (with fixed random seed). If they do, there's a reproducibility issue.

**Q: What if SHAP is too slow?**
A: Use permutation importance, or use SHAP on sample of data (e.g., first 1000 rows).

**Q: Are these explanations legally binding?**
A: No. They're technical explanations, not legal interpretations. Consult legal team for compliance.

**Q: How do I explain this to non-technical stakeholders?**
A: Use the `explanation_summary.md` file—it's written for business audience.

---

## Support Resources

- **In-app help**: Each page has contextual guidance
- **Sample data**: Built-in examples for learning
- **README.md**: Technical documentation
- **This guide**: Step-by-step instructions
- **Course materials**: Lab 5 lecture notes
- **Office hours**: Contact course instructors

---

## Version History

**v2.0** (2026-02-17)
- Full enterprise-grade implementation
- 7-step workflow with persona guidance
- Comprehensive artifact export
- Validation review checklist

**v1.0** (2026-01-15)
- Initial release
- Basic SHAP integration

---

**Remember**: The goal is audit-defensible explanations, not perfect visualizations. Prioritize reproducibility and documentation quality.