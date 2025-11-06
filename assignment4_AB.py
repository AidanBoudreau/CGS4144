# assignment4_AB.py
# CGS4144 — Assignment 4
# Train/evaluate a Random Forest on the top-N most variable genes.
# Includes robust label handling, proper AUCs, and per-sample predictions.

import argparse
import sys
import os
import numpy as np
import pandas as pd

# Force headless plotting before importing pyplot
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# ---------- Helpers ----------

def get_most_variable_genes(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return df subset with the n highest-variance rows (genes)."""
    var = df.var(axis=1, numeric_only=True)
    top = var.nlargest(n).index
    return df.loc[top]

def collapse_rare_labels(y: pd.Series, min_count: int = 3) -> pd.Series:
    """Replace labels with count < min_count by 'Other' (to stabilize CV)."""
    vc = y.value_counts(dropna=False)
    rare = vc[vc < min_count].index
    return y.replace(dict.fromkeys(rare, "Other"))

def plot_confusion(cm: np.ndarray, classes: list[str], title: str, out_path: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

# ---------- Main workflow ----------

def main():
    ap = argparse.ArgumentParser(
        description="Subset to top-N variable genes and train a Random Forest classifier."
    )
    ap.add_argument("--expr", required=True,
                    help="Expression TSV (rows=genes, cols=samples). Must have a 'Gene' index column.")
    ap.add_argument("--meta", required=True,
                    help="Metadata TSV with sample + group columns.")
    ap.add_argument("--sample-col", default="refinebio_accession_code",
                    help="Column in metadata for sample IDs (must match expression column names).")
    ap.add_argument("--group-col", default="Group",
                    help="Column in metadata for target labels.")
    ap.add_argument("--n-genes", type=int, default=5000,
                    help="Number of most-variable genes to keep (default: 5000).")
    ap.add_argument("--test-size", type=float, default=0.20,
                    help="Held-out test fraction (default: 0.20).")
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="StratifiedKFold folds (default: 5).")
    ap.add_argument("--random-state", type=int, default=0,
                    help="Random seed.")
    ap.add_argument("--out-prefix", default="assignment4_RF",
                    help="Prefix for output files.")
    args = ap.parse_args()

    # -- Load expression
    expr = pd.read_csv(args.expr, sep="\t")
    if "Gene" not in expr.columns:
        print("ERROR: Expression file must include a 'Gene' column for the index.", file=sys.stderr)
        sys.exit(1)
    expr = expr.set_index("Gene")
    expr.columns = expr.columns.astype(str).str.strip()

    # -- Load metadata
    meta = pd.read_csv(args.meta, sep="\t")
    if args.sample_col not in meta.columns or args.group_col not in meta.columns:
        print(f"ERROR: Metadata must contain '{args.sample_col}' and '{args.group_col}'.", file=sys.stderr)
        sys.exit(1)
    meta[args.sample_col] = meta[args.sample_col].astype(str).str.strip()
    meta[args.group_col]  = meta[args.group_col].astype(str).str.strip().replace({"nan": "NA", "None": "NA"})

    # -- Align samples (intersection keeps expression column order)
    common = expr.columns.intersection(meta[args.sample_col])
    if len(common) == 0:
        print("ERROR: No overlapping sample IDs between expression and metadata.", file=sys.stderr)
        sys.exit(1)
    expr = expr.loc[:, common]
    meta = meta.set_index(args.sample_col).loc[common].copy()

    # -- Build labels, collapse super-rare ones to 'Other'
    y_raw = meta[args.group_col].fillna("NA")
    y_raw = collapse_rare_labels(y_raw, min_count=3)

    # -- Drop labels with <2 samples (stratify requirement)
    vc = y_raw.value_counts(dropna=False)
    keep_labels = vc[vc >= 2].index
    keep_mask = y_raw.isin(keep_labels)

    dropped = (~keep_mask).sum()
    if dropped > 0:
        print(f"⚠ Dropping {dropped} sample(s) with labels having <2 members: {vc[vc < 2].to_dict()}")

    expr = expr.loc[:, keep_mask.index[keep_mask]]
    meta = meta.loc[keep_mask].copy()
    y_raw = y_raw.loc[keep_mask]

    # -- Feature selection: top N variable genes (on filtered set)
    expr_sub = get_most_variable_genes(expr, args.n_genes)  # genes x samples
    X_all = expr_sub.T.values  # samples x genes

    # -- Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw.values)
    class_names = list(le.classes_)

    # -- Show class counts after filtering
    counts_after = pd.Series(y).value_counts().sort_index()
    print("Class counts after filtering:", {class_names[i]: int(counts_after.get(i, 0)) for i in range(len(class_names))})

    # -- Robust train/test split (fall back if stratify impossible)
    stratify_vec = y if counts_after.min() >= 2 else None
    if stratify_vec is None:
        print("⚠ Not enough samples per class for stratified split; using unstratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=args.test_size, stratify=stratify_vec, random_state=args.random_state
    )

    # -- Model
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.random_state,
    )

    # -- Cross-validation (TRAIN only)
    cv = StratifiedKFold(n_splits=min(args.cv_folds, len(np.unique(y_train))), shuffle=True,
                         random_state=args.random_state)
    cv_res = cross_validate(
        rf, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1, return_estimator=False
    )
    cv_df = pd.DataFrame({
        "fold": np.arange(1, len(cv_res["test_accuracy"]) + 1),
        "accuracy": cv_res["test_accuracy"],
        "f1_macro": cv_res["test_f1_macro"],
    })
    cv_df.to_csv(f"{args.out_prefix}_cv.csv", index=False)

    # -- Fit on TRAIN, evaluate on TEST
    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred_test)
    f1m = f1_score(y_test, y_pred_test, average="macro")

    # --- Proper AUC handling (binary vs multiclass), also report per-class AUCs
    per_class_auc = {}
    try:
        y_proba_test = rf.predict_proba(X_test)
        n_classes_test = len(np.unique(y_test))
        if n_classes_test < 2:
            auc_macro = np.nan
        else:
            if len(class_names) == 2:
                # Binary: compute AUC per class as positive
                for cname in class_names:
                    c_idx = le.transform([cname])[0]
                    y_true_bin = (y_test == c_idx).astype(int)
                    per_class_auc[cname] = roc_auc_score(y_true_bin, y_proba_test[:, c_idx])
                auc_macro = max(per_class_auc.values())
            else:
                # Multiclass: macro OVR AUC
                auc_macro = roc_auc_score(y_test, y_proba_test, multi_class="ovr", average="macro")
    except Exception:
        auc_macro = np.nan

    # -- Save summary
    summary = pd.DataFrame([{
        "n_genes": args.n_genes,
        "test_accuracy": acc,
        "test_f1_macro": f1m,
        "test_auc_macro_ovr": auc_macro,
        "n_classes": len(class_names),
        "classes": "|".join(class_names),
        **{f"auc_{k}": v for k, v in per_class_auc.items()},
    }])
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    # -- Classification report (TEST)
    with open(f"{args.out_prefix}_classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred_test, target_names=class_names))

    # -- Confusion matrix (TEST)
    cm = confusion_matrix(y_test, y_pred_test)
    np.savetxt(f"{args.out_prefix}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, class_names, "Random Forest — Confusion Matrix", f"{args.out_prefix}_cm.png")

    # -- Feature importances (map back to gene names)
    if hasattr(rf, "feature_importances_"):
        imp = pd.DataFrame({
            "gene": expr_sub.index.values,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
        imp.to_csv(f"{args.out_prefix}_feature_importances.csv", index=False)

    # -- Save the exact gene list used
    pd.Series(expr_sub.index.values, name="gene").to_csv(
        f"{args.out_prefix}_top{args.n_genes}_genes.csv", index=False
    )

    # -- Save per-sample predictions (for Part 3) - predict on ALL filtered samples
    all_pred = rf.predict(X_all)
    pred_df = pd.DataFrame({
        "SampleID": expr_sub.columns,
        "TrueLabel": y_raw.loc[expr_sub.columns].values,
        "PredictedLabel": le.inverse_transform(all_pred)
    })
    pred_df.to_csv(f"{args.out_prefix}_predictions.csv", index=False)

    # -- Console summary
    print("--- Training complete ---")
    print(f"CV (n={len(cv_df)}) accuracy: {cv_df['accuracy'].mean():.3f} ± {cv_df['accuracy'].std():.3f}")
    print(f"CV (n={len(cv_df)}) f1_macro: {cv_df['f1_macro'].mean():.3f} ± {cv_df['f1_macro'].std():.3f}")
    print(f"TEST accuracy: {acc:.3f}, TEST macro-F1: {f1m:.3f}, TEST macro-OVR AUC: {auc_macro:.3f}")
    print(f"Outputs written with prefix: {args.out_prefix}*")

if __name__ == "__main__":
    main()
