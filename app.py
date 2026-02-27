from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import json
import os
import random
import string
import zipfile
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")


# Optional-but-required-by-lab libs
try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    _HAS_LIME = True
except Exception:
    LimeTabularExplainer = None
    _HAS_LIME = False


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="QuLab: Lab 5: Interpretability & Explainability Control Workbench",
    layout="wide",
)

# ---------------------------
# Constants
# ---------------------------
DEFAULT_SEED = 42
REPORTS_ROOT = os.path.join("reports", "session05")
APP_VERSION = "2.0"
COURSE = "AI Design & Deployment Risks (Spring 2026)"
SESSION = "5 — Interpretability & Explainability Controls"

PERSONA_NAME = "Maya Patel"
PERSONA_ROLE = "Model Validator"
ORG_NAME = "QuantaBank (Fictional)"


# ---------------------------
# Utilities
# ---------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _random_run_id(prefix: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = "".join(random.choice(string.ascii_lowercase + string.digits)
                  for _ in range(6))
    return f"{prefix}_{stamp}_{suf}"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def df_sha256(df: pd.DataFrame) -> str:
    b = df.to_csv(index=False).encode("utf-8")
    return sha256_bytes(b)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def model_family(model: Any) -> str:
    """Coarse model family for method selection logic."""
    if isinstance(model, RandomForestClassifier):
        return "blackbox"
    if isinstance(model, LogisticRegression):
        return "linear"
    # Fallback heuristic for other sklearn estimators
    name = model.__class__.__name__.lower()
    if "forest" in name or "tree" in name or "gb" in name:
        return "tree"
    if "logistic" in name or "linear" in name:
        return "linear"
    return "blackbox"


def decision_label(default_prob: float, threshold: float) -> str:
    return "DENY" if default_prob >= threshold else "APPROVE"


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# ---------------------------
# Sample data + baseline models (backend-loaded)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_sample_credit_data(seed: int = DEFAULT_SEED, n: int = 2500) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Deterministic, backend-generated credit dataset.
    Target y=1 means 'default risk' (bad outcome). Lower predicted risk => better.
    """
    rng = np.random.default_rng(seed)

    # Core applicant attributes
    age = rng.integers(21, 70, size=n)
    income = rng.lognormal(mean=10.5, sigma=0.35, size=n) / \
        1000.0  # ~ (20k..200k) in $k
    employment_years = np.clip(rng.normal(
        loc=(age - 18) / 6.0, scale=2.0, size=n), 0, 40)

    credit_score = np.clip(rng.normal(loc=680, scale=55, size=n), 300, 850)
    num_delinquencies = np.clip(rng.poisson(lam=0.6, size=n), 0, 12)

    # Loan attributes
    loan_amount = rng.lognormal(
        mean=10.1, sigma=0.55, size=n) / 1000.0  # in $k
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n, p=[
                                  0.12, 0.18, 0.30, 0.20, 0.20])
    interest_rate = np.clip(rng.normal(
        loc=12.0, scale=4.0, size=n), 3.0, 30.0)  # %
    dti = np.clip(rng.normal(loc=0.28, scale=0.12, size=n),
                  0.0, 1.5)  # debt-to-income

    # Simple engineered features (still interpretable)
    monthly_payment_proxy = (loan_amount * 1000.0) * \
        (interest_rate / 100.0) / np.maximum(loan_term_months, 1)
    payment_to_income = monthly_payment_proxy / \
        np.maximum(income * 1000.0 / 12.0, 1.0)

    X = pd.DataFrame({
        "age": age.astype(int),
        "income_k": income,
        "employment_years": employment_years,
        "credit_score": credit_score,
        "num_delinquencies": num_delinquencies.astype(int),
        "loan_amount_k": loan_amount,
        "loan_term_months": loan_term_months.astype(int),
        "interest_rate_pct": interest_rate,
        "dti": dti,
        "payment_to_income": payment_to_income,
    })

    # Ground-truth-ish logistic risk function (adjusted for more balanced classes)
    z = (
        -1.5  # Adjusted intercept for ~25% default rate
        + 0.008 * (700 - credit_score)  # Stronger credit score impact
        + 3.5 * dti  # Stronger DTI impact
        + 2.0 * payment_to_income  # Stronger payment burden impact
        + 0.25 * num_delinquencies  # Stronger delinquency impact
        + 0.05 * (interest_rate - 10.0)  # Stronger interest rate impact
        + 0.002 * (loan_amount * 1000.0 - 15000.0) /
        1000.0  # Stronger loan amount impact
        - 0.06 * employment_years  # Stronger employment stability impact
        - 0.01 * (income - 60.0)  # Stronger income impact
    )
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p).astype(int)

    return X, pd.Series(y, name="default_risk")


@st.cache_resource(show_spinner=False)
def train_baseline_models(X: pd.DataFrame, y: pd.Series, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    set_seeds(seed)
    models: Dict[str, Any] = {}

    lr = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=seed,
    )
    lr.fit(X, y)
    models["Interpretable (Logistic Regression)"] = lr

    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    models["Black-box (Random Forest)"] = rf

    return models


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y, pred).tolist()
    report = classification_report(y, pred, output_dict=True)
    return {"roc_auc": float(auc), "confusion_matrix": cm, "classification_report": report}


# ---------------------------
# Explainability engines
# ---------------------------
def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_repeats: int = 5,
) -> pd.DataFrame:
    set_seeds(seed)
    r = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="roc_auc",
        n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return imp


def _shap_to_2d_positive_class(sv) -> np.ndarray:
    """
    Normalize SHAP outputs across versions into a 2D array: (n_samples, n_features),
    selecting the positive class for binary classification when needed.
    """
    # Case 1: list of arrays (one per class)
    if isinstance(sv, list):
        arr = sv[1] if len(sv) > 1 else sv[0]
        return np.asarray(arr)

    # Case 2: shap.Explanation object
    if hasattr(sv, "values"):
        sv = sv.values

    arr = np.asarray(sv)

    # Case 3: 3D array (n_samples, n_features, n_outputs)
    if arr.ndim == 3:
        k = 1 if arr.shape[2] > 1 else 0
        arr = arr[:, :, k]

    return arr


def compute_shap_global(
    model: Any,
    X_bg: pd.DataFrame,
    X_sample: pd.DataFrame,
    family: str,
) -> Tuple[pd.DataFrame, Any]:
    if not _HAS_SHAP:
        raise RuntimeError("SHAP is not installed.")

    if family == "tree":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        sv2d = _shap_to_2d_positive_class(sv)
        mean_abs = np.abs(sv2d).mean(axis=0).ravel()

        imp = pd.DataFrame({
            "feature": list(X_sample.columns),
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        return imp, {"explainer": "TreeExplainer"}

    if family == "linear":
        explainer = shap.LinearExplainer(
            model, X_bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_sample)
        sv2d = _shap_to_2d_positive_class(sv)
        mean_abs = np.abs(sv2d).mean(axis=0).ravel()

        imp = pd.DataFrame({
            "feature": list(X_sample.columns),
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        return imp, {"explainer": "LinearExplainer"}

    raise RuntimeError(
        "SHAP global explanations are enabled only for tree/linear baseline models in this lab.")


def compute_shap_local(
    model: Any,
    X_bg: pd.DataFrame,
    x_row: pd.DataFrame,
    family: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not _HAS_SHAP:
        raise RuntimeError("SHAP is not installed.")

    if family == "tree":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x_row)
        sv2d = _shap_to_2d_positive_class(sv)
        contrib = sv2d[0]

        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[1] if len(base) > 1 else base[0]

    elif family == "linear":
        explainer = shap.LinearExplainer(
            model, X_bg, feature_perturbation="interventional")
        sv = explainer.shap_values(x_row)
        sv2d = _shap_to_2d_positive_class(sv)
        contrib = sv2d[0]

        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[1] if len(base) > 1 else base[0]

    else:
        raise RuntimeError(
            "SHAP local explanations are enabled only for tree/linear baseline models in this lab.")

    df = pd.DataFrame({
        "feature": list(x_row.columns),
        "value": x_row.iloc[0].values,
        "shap_contribution": contrib,
        "abs_contribution": np.abs(contrib),
    }).sort_values("abs_contribution", ascending=False).reset_index(drop=True)

    meta = {"base_value": safe_float(
        base), "explainer": explainer.__class__.__name__}
    return df, meta


def shap_summary_plot(imp_df: pd.DataFrame, title: str) -> plt.Figure:
    fig = plt.figure(figsize=(7, 4))
    plt.barh(imp_df["feature"][::-1], imp_df.iloc[::-1, 1])
    plt.title(title)
    plt.xlabel(imp_df.columns[1])
    plt.tight_layout()
    return fig


def shap_local_waterfall_plot(local_df: pd.DataFrame, base_value: float, title: str) -> plt.Figure:
    if not _HAS_SHAP:
        raise RuntimeError("SHAP is not installed.")
    values = local_df["shap_contribution"].values
    data = local_df["value"].values
    feature_names = local_df["feature"].tolist()

    exp = shap.Explanation(values=values, base_values=base_value,
                           data=data, feature_names=feature_names)
    fig = plt.figure(figsize=(9, 4))
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.title(title)
    plt.tight_layout()
    return fig


def lime_local_explanation(
    model: Any,
    X_train: pd.DataFrame,
    x_row: pd.DataFrame,
    class_names: List[str],
    seed: int,
    num_features: int = 10,
) -> Dict[str, Any]:
    if not _HAS_LIME:
        raise RuntimeError("LIME is not installed.")
    set_seeds(seed)

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )

    exp = explainer.explain_instance(
        data_row=x_row.iloc[0].values,
        predict_fn=model.predict_proba,
        num_features=num_features,
        top_labels=2,   # <-- ensure both labels are available when binary
    )

    # --- Choose a label safely ---
    available_labels = sorted(list(exp.local_exp.keys()))
    # Prefer "default" label index 1 if it exists, else fall back to first available
    label_to_use = 1 if 1 in exp.local_exp else available_labels[0]

    return {
        "label_used": int(label_to_use),
        "available_labels": available_labels,
        "as_list": exp.as_list(label=label_to_use),
        "as_html": exp.as_html(),
        "score": safe_float(getattr(exp, "score", np.nan)),
        "local_pred": exp.local_pred.tolist() if hasattr(exp, "local_pred") else None,
        "intercept": (
            safe_float(exp.intercept[label_to_use])
            if hasattr(exp, "intercept") and isinstance(exp.intercept, (list, np.ndarray, dict))
            else None
        ),
    }


# ---------------------------
# Counterfactual (basic)
# ---------------------------
CF_PREFERENCES = {
    "credit_score": "increase",
    "income_k": "increase",
    "employment_years": "increase",
    "dti": "decrease",
    "payment_to_income": "decrease",
    "num_delinquencies": "decrease",
    "loan_amount_k": "decrease",
    "interest_rate_pct": "decrease",
}


def generate_counterfactual_greedy(
    model: Any,
    X_ref: pd.DataFrame,
    x0: pd.DataFrame,
    threshold: float,
    local_ranked_features: List[str],
    max_steps: int = 6,
    candidate_quantiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> Dict[str, Any]:
    """
    Greedy, constraint-aware counterfactual:
    - iteratively tweaks one feature at a time to reduce predicted default risk,
      until it crosses the approval threshold (risk < threshold) or max_steps reached.
    """
    x = x0.copy()
    p0 = model.predict_proba(x)[:, 1][0]
    if p0 < threshold:
        return {
            "status": "already_approved",
            "p_default_start": float(p0),
            "p_default_end": float(p0),
            "changes": {},
            "counterfactual_row": x.to_dict(orient="records")[0],
        }

    changes = {}
    best_p = p0

    for _step in range(max_steps):
        best_candidate = None

        for feat in local_ranked_features[:8]:
            if feat not in X_ref.columns:
                continue

            pref = CF_PREFERENCES.get(feat, None)
            qs = np.quantile(X_ref[feat].values, candidate_quantiles)

            current = float(x.iloc[0][feat])
            candidates = []
            for v in qs:
                v = float(v)
                if pref == "increase" and v <= current:
                    continue
                if pref == "decrease" and v >= current:
                    continue
                candidates.append(v)

            if not candidates:
                candidates = [float(qs[0]), float(qs[-1])]

            for v in candidates:
                x_try = x.copy()
                x_try.iloc[0, x_try.columns.get_loc(feat)] = v
                p_try = float(model.predict_proba(x_try)[:, 1][0])
                if p_try < best_p - 1e-6:
                    delta = abs(v - current)
                    score = (best_p - p_try) / (delta + 1e-6)
                    cand = (score, p_try, feat, v, current)
                    if (best_candidate is None) or (cand[0] > best_candidate[0]):
                        best_candidate = cand

        if best_candidate is None:
            break

        _, p_new, feat, v_new, v_old = best_candidate
        x.iloc[0, x.columns.get_loc(feat)] = v_new
        changes[feat] = {"from": float(v_old), "to": float(v_new)}
        best_p = p_new

        if best_p < threshold:
            return {
                "status": "flipped",
                "p_default_start": float(p0),
                "p_default_end": float(best_p),
                "changes": changes,
                "counterfactual_row": x.to_dict(orient="records")[0],
            }

    return {
        "status": "not_flipped",
        "p_default_start": float(p0),
        "p_default_end": float(best_p),
        "changes": changes,
        "counterfactual_row": x.to_dict(orient="records")[0],
    }


# ---------------------------
# Reproducibility & artifact export
# ---------------------------
@dataclass
class RunContext:
    run_id: str
    seed: int
    created_utc: str
    risk_tier: str
    model_name: str
    model_family: str
    threshold: float


def get_run_context(models: Dict[str, Any]) -> RunContext:
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = _random_run_id("session05")
    if "seed" not in st.session_state:
        st.session_state["seed"] = DEFAULT_SEED
    if "risk_tier" not in st.session_state:
        st.session_state["risk_tier"] = "Tier 2 (Moderate)"
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = list(models.keys())[0]
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = 0.35

    model_name = st.session_state["model_name"]
    fam = model_family(models[model_name])

    return RunContext(
        run_id=st.session_state["run_id"],
        seed=int(st.session_state["seed"]),
        created_utc=st.session_state.get("created_utc", _utc_now_iso()),
        risk_tier=st.session_state["risk_tier"],
        model_name=model_name,
        model_family=fam,
        threshold=float(st.session_state["threshold"]),
    )


def compute_model_hash(model: Any) -> str:
    import joblib
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return sha256_bytes(buf.getvalue())


def export_bundle(
    ctx: RunContext,
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    global_obj: Optional[Dict[str, Any]],
    local_obj: Optional[Dict[str, Any]],
    counterfactual_obj: Optional[Dict[str, Any]],
    notes_md: str,
) -> Tuple[str, str]:
    ensure_dir(REPORTS_ROOT)
    out_dir = os.path.join(REPORTS_ROOT, ctx.run_id)
    ensure_dir(out_dir)

    dataset_hash = df_sha256(pd.concat([X, y], axis=1))
    model_hash = compute_model_hash(model)

    config_snapshot = {
        "app_version": APP_VERSION,
        "course": COURSE,
        "session": SESSION,
        "created_utc": ctx.created_utc,
        "run_id": ctx.run_id,
        "seed": ctx.seed,
        "risk_tier": ctx.risk_tier,
        "model_name": ctx.model_name,
        "model_family": ctx.model_family,
        "decision_threshold_default_risk": ctx.threshold,
        "model_sha256": model_hash,
        "dataset_sha256": dataset_hash,
    }

    paths = {}

    def _write_json(name: str, obj: Dict[str, Any]) -> None:
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        paths[name] = p

    def _write_text(name: str, s: str) -> None:
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(s)
        paths[name] = p

    if global_obj is not None:
        _write_json("global_explanation.json", global_obj)
    if local_obj is not None:
        _write_json("local_explanation.json", local_obj)
    if counterfactual_obj is not None:
        _write_json("counterfactual_example.json", counterfactual_obj)

    _write_text("explanation_summary.md", notes_md)
    _write_json("config_snapshot.json", config_snapshot)

    evidence = {
        "run_id": ctx.run_id,
        "created_utc": ctx.created_utc,
        "model_sha256": model_hash,
        "dataset_sha256": dataset_hash,
        "artifacts": {},
        "hash_algorithm": "sha256",
    }
    for name, p in paths.items():
        evidence["artifacts"][name] = {
            "path": p.replace("\\", "/"),
            "sha256": sha256_file(p),
            "bytes": os.path.getsize(p),
        }

    _write_json("evidence_manifest.json", evidence)

    zip_name = f"Session_05_{ctx.run_id}.zip"
    zip_path = os.path.join(out_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, p in paths.items():
            z.write(p, arcname=name)
        z.write(os.path.join(out_dir, "evidence_manifest.json"),
                arcname="evidence_manifest.json")

    return out_dir, zip_path


# ---------------------------
# UI Components
# ---------------------------
def sidebar_controls(models: Dict[str, Any]):
    # Sidebar (starter)
    st.sidebar.image(
        "https://www.quantuniversity.com/assets/img/logo5.jpg",
        width='stretch'
    )
    st.sidebar.divider()

    # Page navigation
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "0) Mission Brief",
            "1) Persona Story",
            "2) Data & Model ",
            "3) Global Explanation (SHAP / Permutation)",
            "4) Local Explanation (SHAP)",
            "5) Local Explanation (LIME)",
            "6) Counterfactual (Basic)",
            "7) Reproducibility & Evidence",
            "8) Export Bundle",
        ],
        key="page_select",
    )

    st.session_state["run_id"] = _random_run_id("session05")
    st.session_state["created_utc"] = _utc_now_iso()

    st.session_state.seed = 42

    # st.sidebar.selectbox(
    #     "Risk tier",
    #     ["Tier 1 (High)", "Tier 2 (Moderate)", "Tier 3 (Low)"],
    #     key="risk_tier",
    # )
    st.sidebar.selectbox("Baseline model", list(
        models.keys()), key="model_name")
    st.sidebar.slider(
        "Decision threshold (default risk)",
        min_value=0.05, max_value=0.80,
        value=float(st.session_state.get("threshold", 0.35)),
        step=0.01, key="threshold",
        help="If predicted default risk ≥ threshold → DENY, else APPROVE.",
    )

    ctx = get_run_context(models)

    return page, ctx


st.title("QuLab: Lab 5: Interpretability & Explainability Control Workbench")
st.divider()


def mission_brief_page():
    st.markdown(
        """
This Streamlit application operationalizes **model interpretability as an enterprise control artifact**.

You will walk through an *audit-style* workflow: choose a baseline model, generate global and local explanations,
run a basic counterfactual, and export an evidence bundle that is:
- Tied to a specific **model version**,
- Reproducible with a fixed **seed**,
- Backed by a **SHA-256 evidence manifest**.

> **Enterprise question:** *Can this model’s decisions be explained to auditors, regulators, and internal stakeholders in a reproducible and defensible way?*
"""
    )
    st.info(
        "Use the sidebar navigation to follow the persona-driven story arc end-to-end.")

    st.markdown("### Quick start checklist")
    st.markdown(
        """
1. Use the **sidebar** to pick a baseline model and decision threshold.  
2. Review **Global Explanation** for drivers and stability.  
3. Review **Local Explanation** for a single case (SHAP + LIME).  
4. Generate a **Counterfactual** (basic) and record constraints.  
5. Export the **evidence bundle**.
"""
    )


def persona_story_page():
    st.header("1) Persona Story — From Analysis to Audit Evidence")
    st.markdown(
        f"""
### You are **{PERSONA_NAME}**, {PERSONA_ROLE} at **{ORG_NAME}**

A new credit decisioning model is scheduled for deployment. Before it can go live, you must deliver a *validation-ready* explanation package.

Your stakeholders:
- **Internal Auditor**: wants traceability and reproducibility.
- **ML Engineer**: wants actionable feedback if explanations reveal issues.
- **Business Owner**: wants clarity on top drivers of approvals/denials.

---

### Your mission (what “good” looks like)

You will produce explanation artifacts that are:
1. **Reproducible** — repeated runs with the same configuration produce consistent rankings.
2. **Traceable** — artifacts are tied to a specific model version hash and dataset hash.
3. **Decision-relevant** — local explanations clearly connect to approve/deny outcomes.
4. **Exportable** — packaged into a single evidence bundle for review.

---

### The story arc

**Scene A — Triage:**  
Pick the baseline model (interpretable vs black-box) and set the decision threshold.

**Scene B — Global understanding:**  
Identify which features most influence default risk *overall*.

**Scene C — Case file deep-dive:**  
Explain *one* decision with local explanations (SHAP + LIME), using consistent language.

**Scene D — Remediation signal:**  
Generate a basic counterfactual: “what minimal change flips the decision?”

**Scene E — Audit bundle:**  
Export all artifacts, with SHA-256 hashes.

> This transforms interpretability from “pretty charts” into **enterprise control evidence**.
"""
    )


def data_model_page(X: pd.DataFrame, y: pd.Series, models: Dict[str, Any], ctx: RunContext):
    st.header("2) Data & Model ")
    st.markdown(
        """
This lab uses **backend-loaded sample data and baseline models**. No uploads are required.

- The dataset is deterministically generated from a fixed seed.
- Two baseline models are trained on the backend:
  - **Interpretable**: Logistic Regression
  - **Black-box**: Random Forest

Target definition:
- `default_risk = 1` → higher risk (bad)
- `default_risk = 0` → lower risk (good)

Decision rule used throughout the app:
- If predicted default risk **≥ threshold** → **DENY**
- Else → **APPROVE**
"""
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Dataset preview")
        st.dataframe(pd.concat([X.head(20), y.head(20)],
                     axis=1), width='stretch')
        st.caption(
            f"Rows: {len(X):,} • Features: {X.shape[1]} • Dataset SHA-256 is computed at export time.")
    with col2:
        st.subheader("Model snapshot")
        m = models[ctx.model_name]
        st.code(repr(m), language="text")
        st.write(f"Detected model family: **{ctx.model_family}**")
        st.caption(
            "Model SHA-256 is computed at export time (by serializing the model).")

    st.subheader("Performance sanity check (not a full validation)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=ctx.seed, stratify=y)
    m = models[ctx.model_name]
    metrics = evaluate_model(m, X_test, y_test)
    st.metric("ROC-AUC (test)", f"{metrics['roc_auc']:.3f}")
    st.markdown("**Confusion matrix (threshold=0.5)**")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm,
                         columns=["Predicted: No Default",
                                  "Predicted: Default"],
                         index=["Actual: No Default", "Actual: Default"])
    st.dataframe(cm_df, width='stretch')

    st.markdown("""
    **📊 Understanding the Confusion Matrix:**
    - **Top-left (True Negative)**: Correctly predicted as no default
    - **Top-right (False Positive)**: Incorrectly predicted as default (Type I error)
    - **Bottom-left (False Negative)**: Incorrectly predicted as no default (Type II error)
    - **Bottom-right (True Positive)**: Correctly predicted as default
    
    In credit risk: False Negatives are costly (lending to defaulters), while False Positives mean lost business opportunities.
    """)

    with st.expander("Classification report"):
        report = metrics["classification_report"]
        report_data = []
        for label, values in report.items():
            if isinstance(values, dict):
                report_data.append({
                    "Class": label,
                    "Precision": f"{values.get('precision', 0):.3f}",
                    "Recall": f"{values.get('recall', 0):.3f}",
                    "F1-Score": f"{values.get('f1-score', 0):.3f}",
                    "Support": int(values.get('support', 0))
                })
        st.dataframe(pd.DataFrame(report_data), width='stretch')

        st.markdown("""
        **📊 Understanding the Classification Report:**
        - **Precision**: Of all predicted defaults, what % were actual defaults? (Relevance)
        - **Recall**: Of all actual defaults, what % did we catch? (Sensitivity)
        - **F1-Score**: Harmonic mean of precision and recall (balanced metric)
        - **Support**: Number of actual occurrences in the test set
        - **Accuracy**: Overall correct predictions
        - **Macro avg**: Unweighted mean (treats all classes equally)
        - **Weighted avg**: Weighted by support (accounts for class imbalance)
        """)


def global_explanation_page(X: pd.DataFrame, y: pd.Series, models: Dict[str, Any], ctx: RunContext):
    st.header("3) Global Explanation (SHAP / Permutation)")
    st.markdown(
        """
Global explanations answer: **“What drives the model overall?”**

In this lab:
- For **tree** and **linear** baselines, we prefer **SHAP**.
- For **black-box** models (or if SHAP is unavailable), we use **Permutation Importance** (ROC-AUC).
"""
    )

    m = models[ctx.model_name]
    fam = ctx.model_family

    X_bg = X.sample(n=min(400, len(X)), random_state=ctx.seed)
    X_sample = X.sample(n=min(800, len(X)), random_state=ctx.seed + 1)

    if fam in ("tree", "linear") and _HAS_SHAP:
        imp_df, meta = compute_shap_global(
            m, X_bg=X_bg, X_sample=X_sample, family=fam)
        st.success(f"Method used: **SHAP ({meta['explainer']})**")
        st.caption(f"Sample size: {len(X_sample)}")
        st.dataframe(imp_df.head(15), width='stretch')
        fig = shap_summary_plot(imp_df.head(
            15), title="Mean |SHAP| feature importance (top 15)")
        st.pyplot(fig, clear_figure=True)

        st.markdown("""
        **📊 Understanding Global SHAP Values:**
        - **Mean |SHAP|**: Average absolute impact of each feature across all predictions
        - **Higher values** = feature has stronger influence on model predictions
        - SHAP values are **additive**: sum of all contributions + base value = predicted risk
        - **TreeExplainer**: Fast, exact method for tree-based models
        - **LinearExplainer**: Efficient method for linear models with feature interactions
        
        💡 **Validation insight**: Top features should align with domain knowledge. Unexpected drivers warrant investigation.
        """)
    else:
        imp_df = compute_permutation_importance(
            m, X_sample, y.loc[X_sample.index], seed=ctx.seed, n_repeats=7)
        st.warning(
            "Using fallback: **Permutation importance** (expected for black-box models or when SHAP is missing).")
        st.dataframe(imp_df.head(15), width='stretch')

        st.markdown("""
        **📊 Understanding Permutation Importance:**
        - **Importance mean**: Average drop in model performance (ROC-AUC) when feature is randomly shuffled
        - **Importance std**: Variability across multiple shuffles (lower = more stable)
        - **Higher values** = feature is more critical to model accuracy
        - Works for **any model** but can be slower than SHAP for tree models
        - Measures **global importance** but doesn't show direction of effect
        
        💡 **Validation insight**: High standard deviation suggests feature importance may be unstable or context-dependent.
        """)

    st.subheader("Stability check (repeatability)")
    st.markdown(
        """
An enterprise control must be **reproducible**.  
Here we re-run the global importance computation several times and measure rank stability.

- SHAP should be stable with fixed seeds and deterministic models.
- Permutation importance can vary, so we report rank variation.
"""
    )

    repeats = st.slider("Number of repeats", 2, 8, 3,
                        help="More repeats → more confidence, slower runtime.")
    if st.button("Run stability check"):
        ranks = []
        for i in range(repeats):
            seed_i = ctx.seed + i * 7
            X_s = X.sample(n=min(700, len(X)), random_state=seed_i)
            if fam in ("tree", "linear") and _HAS_SHAP:
                imp_i, _ = compute_shap_global(
                    m, X_bg=X_bg, X_sample=X_s, family=fam)
                score_series = imp_i.set_index("feature")["mean_abs_shap"]
            else:
                imp_i = compute_permutation_importance(
                    m, X_s, y.loc[X_s.index], seed=seed_i, n_repeats=5)
                score_series = imp_i.set_index("feature")["importance_mean"]
            ranks.append(score_series.rank(ascending=False))

        rank_df = pd.concat(ranks, axis=1)
        rank_df.columns = [f"run_{i+1}" for i in range(repeats)]
        rank_df["rank_std"] = rank_df.std(axis=1)
        rank_df = rank_df.sort_values("rank_std").reset_index().rename(
            columns={"index": "feature"})
        st.dataframe(rank_df.head(20), width='stretch')
        st.caption("Lower rank_std indicates higher stability.")


def local_explanation_shap_page(X: pd.DataFrame, y: pd.Series, models: Dict[str, Any], ctx: RunContext):
    st.header("4) Local Explanation (SHAP)")
    st.markdown(
        """
Local explanations answer: **“Why this decision for this applicant?”**

Note: In this app, SHAP local explanations are enabled for the **tree** and **linear** baseline models.
"""
    )

    if not _HAS_SHAP:
        st.error(
            "SHAP is not installed. Install with: `pip install shap` and restart.")
        st.stop()

    m = models[ctx.model_name]
    fam = ctx.model_family
    if fam not in ("tree", "linear"):
        st.warning(
            "Selected model is black-box. Use the LIME page for local explanations, or switch to a tree/linear baseline.")
        st.stop()

    idx = st.number_input("Applicant row index", min_value=0,
                          max_value=len(X)-1, value=0, step=1)
    x_row = X.iloc[[int(idx)]]
    p_default = float(m.predict_proba(x_row)[:, 1][0])
    decision = decision_label(p_default, ctx.threshold)

    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric("Predicted default risk", f"{p_default:.3f}")
    col2.metric("Threshold", f"{ctx.threshold:.2f}")
    col3.metric("Decision", decision)

    st.markdown("#### Applicant features")
    st.dataframe(x_row.T.rename(
        columns={x_row.index[0]: "value"}), width='stretch')

    X_bg = X.sample(n=min(500, len(X)), random_state=ctx.seed)
    local_df, meta = compute_shap_local(m, X_bg=X_bg, x_row=x_row, family=fam)

    st.markdown("#### Top contributors (SHAP)")
    st.dataframe(local_df.head(12), width='stretch')

    fig = shap_local_waterfall_plot(
        local_df=local_df.head(12),
        base_value=float(meta["base_value"]),
        title="Local SHAP waterfall (top 12 contributions)",
    )
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
#### 📊 Understanding Local SHAP Explanations

**How to read this (auditor-friendly):**
- **Base value**: Model's average prediction across all cases (starting point)
- **Positive contribution** (+): Increases predicted default risk → pushes toward DENY
- **Negative contribution** (−): Decreases predicted default risk → pushes toward APPROVE
- **Final prediction**: Base value + sum of all contributions
- **Waterfall plot**: Shows cumulative effect as each feature is added

💡 **Validation insight**: Check if top contributors align with business logic. For example:
- Lower credit_score should push toward DENY (positive contribution)
- Higher income should push toward APPROVE (negative contribution)
- High DTI (debt-to-income) should push toward DENY

🔍 **Explainability check**: Can you explain this decision to a loan applicant or regulator?"""
    )


def local_explanation_lime_page(X: pd.DataFrame, y: pd.Series, models: Dict[str, Any], ctx: RunContext):
    st.header("5) Local Explanation (LIME)")
    st.markdown(
        """
LIME provides a **local surrogate model** around one instance.

Why it matters in an enterprise workbench:
- Works for **any** classifier with `predict_proba`.
- Provides a cross-check for SHAP or a route for black-box explainability.
"""
    )

    if not _HAS_LIME:
        st.error(
            "LIME is not installed. Install with: `pip install lime` and restart.")
        st.stop()

    m = models[ctx.model_name]
    idx = st.number_input("Applicant row index", min_value=0, max_value=len(
        X)-1, value=0, step=1, key="lime_idx")
    x_row = X.iloc[[int(idx)]]
    p_default = float(m.predict_proba(x_row)[:, 1][0])
    decision = decision_label(p_default, ctx.threshold)

    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric("Predicted default risk", f"{p_default:.3f}")
    col2.metric("Threshold", f"{ctx.threshold:.2f}")
    col3.metric("Decision", decision)

    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.25, random_state=ctx.seed, stratify=y)

    if st.button("Run LIME explanation"):
        with st.spinner("Computing LIME explanation..."):
            lime_obj = lime_local_explanation(
                model=m,
                X_train=X_train,
                x_row=x_row,
                class_names=["no_default", "default"],
                seed=ctx.seed,
                num_features=10,
            )
        st.success("LIME explanation generated.")

        st.markdown("#### Explanation (list form)")
        lime_df = pd.DataFrame(lime_obj["as_list"], columns=[
            "condition", "weight"]).sort_values("weight", ascending=False)
        st.dataframe(lime_df, width='stretch')

        st.markdown("#### Explanation")
        st.components.v1.html(lime_obj["as_html"], height=500, scrolling=True)

        st.markdown("""
        **📊 Understanding LIME Explanations:**
        - **LIME** = Local Interpretable Model-agnostic Explanations
        - Creates a **local surrogate** (simple) model around this specific prediction
        - **Condition**: Feature value range (LIME discretizes continuous features)
        - **Weight**: Impact on prediction
          - **Positive weight** → pushes toward 'default' (class 1)
          - **Negative weight** → pushes toward 'no default' (class 0)
        - **Model-agnostic**: Works with any black-box classifier
        
        💡 **LIME vs SHAP:**
        - LIME: Approximates locally, faster for complex models, may vary between runs
        - SHAP: Theoretically grounded, consistent, but can be slower
        - Use both as **cross-validation** for critical decisions
        
        ⚠️ **Validation note**: LIME uses random sampling, so explanations may vary slightly between runs even with fixed seed.
        """)


def counterfactual_page(X: pd.DataFrame, y: pd.Series, models: Dict[str, Any], ctx: RunContext):
    st.header("6) Counterfactual (Basic)")
    st.markdown(
        """
Counterfactuals answer: **“What is the smallest change that flips the decision?”**

This is a **basic, constraint-aware greedy** search (control demonstration).  
It prefers “risk-reducing” directions (e.g., increase credit score, decrease DTI).
"""
    )

    m = models[ctx.model_name]
    idx = st.number_input("Applicant row index", min_value=0,
                          max_value=len(X)-1, value=0, step=1, key="cf_idx")
    x_row = X.iloc[[int(idx)]]
    p_default = float(m.predict_proba(x_row)[:, 1][0])
    decision = decision_label(p_default, ctx.threshold)
    st.write(
        f"Current predicted default risk: **{p_default:.3f}** → **{decision}**")

    strategy = st.radio(
        "Ranking strategy",
        ["Use SHAP (if available + supported model)",
         "Use global importance (fallback)"],
        index=0,
    )

    ranked_features: List[str] = list(X.columns)

    if strategy.startswith("Use SHAP") and _HAS_SHAP and ctx.model_family in ("tree", "linear"):
        X_bg = X.sample(n=min(500, len(X)), random_state=ctx.seed)
        local_df, _ = compute_shap_local(
            m, X_bg=X_bg, x_row=x_row, family=ctx.model_family)
        ranked_features = local_df["feature"].tolist()
        st.caption("Ranking features by |local SHAP|.")
    else:
        imp = compute_permutation_importance(
            m,
            X.sample(n=min(900, len(X)), random_state=ctx.seed + 99),
            y.sample(n=min(900, len(y)), random_state=ctx.seed + 99),
            seed=ctx.seed,
            n_repeats=5,
        )
        ranked_features = imp["feature"].tolist()
        st.caption("Ranking features by permutation importance.")

    if st.button("Generate counterfactual"):
        with st.spinner("Searching for a counterfactual..."):
            cf = generate_counterfactual_greedy(
                model=m,
                X_ref=X,
                x0=x_row,
                threshold=ctx.threshold,
                local_ranked_features=ranked_features,
            )

        st.subheader("Counterfactual result")

        # Display status and risk change prominently
        col1, col2, col3 = st.columns([1, 1, 1])
        col1.metric("Status", cf["status"].replace("_", " ").title())
        col2.metric("Initial Risk", f"{cf['p_default_start']:.3f}")
        col3.metric("Final Risk", f"{cf['p_default_end']:.3f}",
                    delta=f"{cf['p_default_end'] - cf['p_default_start']:.3f}")

        st.markdown("**Changes applied**")
        if cf["changes"]:
            changes_df = pd.DataFrame(
                [{"Feature": k, "Original Value": f"{v['from']:.2f}", "Counterfactual Value": f"{v['to']:.2f}",
                  "Change": f"{v['to'] - v['from']:.2f}"}
                    for k, v in cf["changes"].items()]
            )
            st.dataframe(changes_df, width='stretch')
        else:
            st.info("No feature changes needed or found.")

        # Show full counterfactual row
        with st.expander("View complete counterfactual profile"):
            cf_df = pd.DataFrame([cf["counterfactual_row"]])
            st.dataframe(cf_df.T.rename(
                columns={0: "value"}), width='stretch')

        st.markdown("""
        **📊 Understanding Counterfactual Explanations:**
        - **Purpose**: Shows the **minimal changes** needed to flip the decision
        - **Status meanings**:
          - `already_approved`: Current risk below threshold, no changes needed
          - `flipped`: Successfully found changes that reduce risk below threshold
          - `not_flipped`: Could not find changes within constraints/max steps
        
        💡 **Business value**:
        - **Actionable feedback** for applicants: "If you increase your credit score by 50 points..."
        - **Recourse**: Shows if decision can be changed with realistic actions
        - **Fairness check**: Are the required changes reasonable and achievable?
        
        ⚠️ **Validation considerations**:
        - Are suggested changes **realistic**? (e.g., can't instantly change employment years)
        - Are changes **consistent with constraints**? (credit score increases, DTI decreases)
        - Is the minimal change **economically feasible** for the applicant?
        - Does the counterfactual respect **causal relationships**? (income affects payment capacity)
        
        🔍 **Algorithmic note**: This uses a greedy search with feature ranking from SHAP/permutation importance.
        More sophisticated methods (e.g., DICE, Wachter) can find diverse or optimal counterfactuals.
        """)


def reproducibility_page(X: pd.DataFrame, y: pd.Series, models: Dict[str, Any], ctx: RunContext):
    st.header("7) Reproducibility & Evidence")
    st.markdown(
        """
Enterprise interpretability requires **reproducibility controls**.

This page previews:
- Fixed seed usage
- Model SHA-256 (serialized model hash)
- Dataset SHA-256 (data + labels hash)
- What will be captured in `config_snapshot.json`
"""
    )

    m = models[ctx.model_name]
    model_hash = compute_model_hash(m)
    dataset_hash = df_sha256(pd.concat([X, y], axis=1))

    col1, col2 = st.columns([1, 1])
    col1.metric("Model SHA-256 (version hash)", model_hash[:16] + "…")
    col2.metric("Dataset SHA-256", dataset_hash[:16] + "…")

    st.markdown("#### Config snapshot preview")
    config_preview = {
        "app_version": APP_VERSION,
        "created_utc": ctx.created_utc,
        "run_id": ctx.run_id,
        "seed": ctx.seed,
        "risk_tier": ctx.risk_tier,
        "model_name": ctx.model_name,
        "model_family": ctx.model_family,
        "decision_threshold_default_risk": ctx.threshold,
        "model_sha256": model_hash,
        "dataset_sha256": dataset_hash,
    }
    config_df = pd.DataFrame([
        {"Configuration Item": k, "Value": str(v)}
        for k, v in config_preview.items()
    ])
    st.dataframe(config_df, width='stretch')

    with st.expander("View as JSON"):
        st.code(pretty_json(config_preview), language="json")

    st.markdown("""
    **📊 Understanding Reproducibility Controls:**
    - **run_id**: Unique identifier for this explanation session
    - **seed**: Random seed ensuring deterministic results (same inputs → same outputs)
    - **model_sha256**: Cryptographic hash of model parameters (detects any model changes)
    - **dataset_sha256**: Hash of training data (ensures data integrity)
    - **threshold**: Decision boundary used for approve/deny
    
    💡 **Why this matters for enterprise AI:**
    - **Auditability**: Can reproduce exact explanations months/years later
    - **Version control**: Detect if model or data changed between runs
    - **Regulatory compliance**: Demonstrate consistent, documented decision process
    - **Debugging**: Isolate whether issues stem from model, data, or configuration changes
    """)


def export_page(X: pd.DataFrame, y: pd.Series, models: Dict[str, Any], ctx: RunContext):
    st.header("8) Export Bundle")
    st.markdown(
        """
Tip: Use the helper buttons below to populate artifacts into the session for export.
"""
    )

    if "global_obj" not in st.session_state:
        st.session_state["global_obj"] = None
    if "local_obj" not in st.session_state:
        st.session_state["local_obj"] = None
    if "counterfactual_obj" not in st.session_state:
        st.session_state["counterfactual_obj"] = None

    col1, col2, col3 = st.columns([1, 1, 1])
    m = models[ctx.model_name]

    with col1:
        if st.button("Generate GLOBAL artifact", width='stretch'):
            try:
                X_bg = X.sample(n=min(400, len(X)), random_state=ctx.seed)
                X_sample = X.sample(n=min(800, len(X)),
                                    random_state=ctx.seed + 1)
                fam = ctx.model_family
                if fam in ("tree", "linear") and _HAS_SHAP:
                    imp_df, meta = compute_shap_global(
                        m, X_bg=X_bg, X_sample=X_sample, family=fam)
                    method = f"SHAP/{meta['explainer']}"
                else:
                    imp_df = compute_permutation_importance(
                        m, X_sample, y.loc[X_sample.index], seed=ctx.seed, n_repeats=7)
                    method = "PermutationImportance"

                st.session_state["global_obj"] = {
                    "run_id": ctx.run_id,
                    "created_utc": ctx.created_utc,
                    "model_name": ctx.model_name,
                    "model_family": ctx.model_family,
                    "method": method,
                    "top_features": imp_df.head(25).to_dict(orient="records"),
                }
                st.success("Global artifact stored.")
            except Exception as e:
                st.error(f"Global artifact failed: {e}")

    with col2:
        if st.button("Generate LOCAL artifact ", width='stretch'):
            try:
                idx = 0
                x_row = X.iloc[[idx]]
                p_default = float(m.predict_proba(x_row)[:, 1][0])
                decision = decision_label(p_default, ctx.threshold)

                local = None
                if ctx.model_family in ("tree", "linear") and _HAS_SHAP:
                    X_bg = X.sample(n=min(500, len(X)), random_state=ctx.seed)
                    local_df, meta = compute_shap_local(
                        m, X_bg=X_bg, x_row=x_row, family=ctx.model_family)
                    local = {
                        "type": "SHAP",
                        "explainer": meta["explainer"],
                        "base_value": meta["base_value"],
                        "contributions": local_df.head(25).to_dict(orient="records"),
                    }
                elif _HAS_LIME:
                    X_train, _, _, _ = train_test_split(
                        X, y, test_size=0.25, random_state=ctx.seed, stratify=y)
                    lime_obj = lime_local_explanation(
                        model=m,
                        X_train=X_train,
                        x_row=x_row,
                        class_names=["no_default", "default"],
                        seed=ctx.seed,
                        num_features=10,
                    )
                    local = {
                        "type": "LIME",
                        "explanation_list": lime_obj["as_list"],
                    }
                else:
                    raise RuntimeError(
                        "Neither SHAP (supported model) nor LIME is available for local artifact.")

                st.session_state["local_obj"] = {
                    "run_id": ctx.run_id,
                    "created_utc": ctx.created_utc,
                    "row_index": idx,
                    "predicted_default_risk": p_default,
                    "threshold": ctx.threshold,
                    "decision": decision,
                    "local_explanation": local,
                }
                st.success("Local artifact stored.")
            except Exception as e:
                st.error(f"Local artifact failed: {e}")

    with col3:
        if st.button("Generate COUNTERFACTUAL artifact ", width='stretch'):
            try:
                idx = 0
                x_row = X.iloc[[idx]]
                ranked = list(X.columns)
                if _HAS_SHAP and ctx.model_family in ("tree", "linear"):
                    X_bg = X.sample(n=min(500, len(X)), random_state=ctx.seed)
                    local_df, _ = compute_shap_local(
                        m, X_bg=X_bg, x_row=x_row, family=ctx.model_family)
                    ranked = local_df["feature"].tolist()

                cf = generate_counterfactual_greedy(
                    model=m,
                    X_ref=X,
                    x0=x_row,
                    threshold=ctx.threshold,
                    local_ranked_features=ranked,
                )
                st.session_state["counterfactual_obj"] = cf
                st.success("Counterfactual artifact stored.")
            except Exception as e:
                st.error(f"Counterfactual artifact failed: {e}")

    st.divider()
    # Use session_state to persist notes_md
    default_notes = f"""# Lab 5 — Explanation Summary

**Persona:** {PERSONA_NAME} ({PERSONA_ROLE})
**Organization:** {ORG_NAME}
**Run:** {ctx.run_id}
**Created (UTC):** {ctx.created_utc}
**Model:** {ctx.model_name} ({ctx.model_family})
**Decision threshold:** default risk ≥ {ctx.threshold:.2f} ⇒ DENY

## Findings
- *(Write your validation notes here.)*

## Validator notes
- Plausibility of top drivers:
- Any surprising drivers:
- Stability concerns:
- Recommended follow-ups:
"""
    if "notes_md" not in st.session_state:
        st.session_state["notes_md"] = default_notes
    notes_md = st.text_area(
        "Explanation summary (saved as explanation_summary.md)",
        value=st.session_state["notes_md"],
        height=260,
    )

    if st.button("📦 Export evidence bundle"):
        out_dir, zip_path = export_bundle(
            ctx=ctx,
            X=X,
            y=y,
            model=m,
            global_obj=st.session_state.get("global_obj"),
            local_obj=st.session_state.get("local_obj"),
            counterfactual_obj=st.session_state.get("counterfactual_obj"),
            notes_md=notes_md,
        )
        st.success(f"Export complete: `{out_dir}`")
        st.caption(f"Zip bundle: `{zip_path}`")

        with open(zip_path, "rb") as f:
            st.download_button(
                label="Download Session_05 bundle (zip)",
                data=f,
                file_name=os.path.basename(zip_path),
                mime="application/zip",
            )


def appendix_page():
    st.header("9) Appendix — Troubleshooting & Notes")
    st.markdown(
        """
### Required libraries

This lab requires:
- `streamlit`
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`
- `shap`
- `lime`

If SHAP or LIME are missing, install and restart:

```bash
pip install shap lime
````

### Method selection logic (as implemented)

* Tree model → SHAP TreeExplainer
* Linear model → SHAP LinearExplainer
* Black-box model → permutation importance (global) + LIME (local)

### Why the dataset is generated

The spec calls for sample data + sample models that ship with the application.
Here we generate a deterministic “credit-like” dataset and train baseline models at runtime
to preserve backend loading + reproducibility.
"""
    )


def footer():

    st.divider()
    st.write("© 2025 QuantUniversity. All Rights Reserved.")
    st.caption(
        "The purpose of this demonstration is solely for educational use and illustration. "
        "Any reproduction of this demonstration requires prior written consent from QuantUniversity."
    )
    st.caption(
        "This lab was generated using the QuCreate platform. QuCreate relies on AI models for generating code, "
        "which may contain inaccuracies or errors."
    )

# ---------------------------

# Main

# ---------------------------


def main():

    set_seeds(int(st.session_state.get("seed", DEFAULT_SEED)))

    X, y = load_sample_credit_data(
        seed=int(st.session_state.get("seed", DEFAULT_SEED)))
    models = train_baseline_models(X, y, seed=int(
        st.session_state.get("seed", DEFAULT_SEED)))

    page, ctx = sidebar_controls(models)

    if "created_utc" not in st.session_state:
        st.session_state["created_utc"] = _utc_now_iso()

    if page.startswith("0)"):
        mission_brief_page()
    elif page.startswith("1)"):
        persona_story_page()
    elif page.startswith("2)"):
        data_model_page(X, y, models, ctx)
    elif page.startswith("3)"):
        global_explanation_page(X, y, models, ctx)
    elif page.startswith("4)"):
        local_explanation_shap_page(X, y, models, ctx)
    elif page.startswith("5)"):
        local_explanation_lime_page(X, y, models, ctx)
    elif page.startswith("6)"):
        counterfactual_page(X, y, models, ctx)
    elif page.startswith("7)"):
        reproducibility_page(X, y, models, ctx)
    elif page.startswith("8)"):
        export_page(X, y, models, ctx)

    footer()


if __name__ == "__main__":
    main()
