# CGS4144 A4 – KNN (self-contained, single file) — uses your real ERP105501 data

# ==== PATHS (edit if your paths differ) =========================
EXPR_PATH    = "code/ERP105501.tsv"
META_PATH    = "code/metadata_ERP105501.tsv"
GROUPS_PATH  = "code/groups.csv"     # must have columns: sample, group (values: OA / Control)
# ===============================================================

# ==== CONFIG ====================================================
A1_COL       = "group_A1"            # we will create this from groups.csv
CLUSTER_COL  = "cluster_A3_nhi"      # if missing, we’ll stub a single cluster to keep pipeline intact
SAMPLE_COL   = "refinebio_accession_code"
TAG          = "knn"

TOPK_FOR_BASE = 5000
KNN_K         = 5
CV_SPLITS     = 5
RANDOM_STATE  = 42
GENE_SWEEP    = [10, 100, 1000, 10000]
# ===============================================================

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

# Quiet tiny-fold warnings (harmless on small splits)
warnings.filterwarnings(
    "ignore",
    message="Number of classes in training fold",
    category=RuntimeWarning,
)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def _read_any_table(path, index_col=None):
    # Try TSV first; fallback to any whitespace-delimited
    try:
        df = pd.read_csv(path, sep="\t", index_col=index_col)
        if df.shape[1] > 1 or index_col is None:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+", engine="python", index_col=index_col)

def load_data(expr_path, meta_path, sample_col):
    X = _read_any_table(expr_path, index_col=0)
    meta = _read_any_table(meta_path, index_col=None)

    # clean headers
    X.columns    = [str(c).strip().replace("\ufeff","") for c in X.columns]
    meta.columns = [str(c).strip().replace("\ufeff","") for c in meta.columns]

    if sample_col not in meta.columns:
        # fallback to first column if needed
        sample_col = meta.columns[0]

    meta[sample_col] = meta[sample_col].astype(str).str.strip()
    X.columns        = [str(c).strip() for c in X.columns]

    # align samples
    common = [c for c in X.columns if c in set(meta[sample_col])]
    if not common:
        raise ValueError(
            "No overlapping sample names between expression columns and metadata.\n"
            f"Expr columns (first 5): {X.columns[:5].tolist()}\n"
            f"Metadata '{sample_col}' (first 5): {meta[sample_col].head().tolist()}"
        )
    X = X[common]
    meta = meta[meta[sample_col].isin(common)].copy()
    meta = meta.set_index(sample_col).loc[X.columns].reset_index()
    meta = meta.rename(columns={meta.columns[0]: "sample"})  # normalize to 'sample'
    return X, meta

def attach_groups_as_A1(meta, groups_path, new_col="group_A1"):
    """
    Merge groups.csv (sample, group) to meta and convert to binary A1 label:
      OA -> 1 (positive)
      Control -> 0 (negative)
    """
    if not os.path.exists(groups_path):
        raise FileNotFoundError(f"Missing groups file at {groups_path} (needs columns: sample, group)")
    g = pd.read_csv(groups_path)
    # clean columns
    g.columns = [str(c).strip() for c in g.columns]
    if "sample" not in g.columns or "group" not in g.columns:
        raise KeyError("groups.csv must have columns: 'sample' and 'group'.")

    g["sample"] = g["sample"].astype(str).str.strip()
    g["group"]  = g["group"].astype(str).str.strip()

    out = meta.merge(g, on="sample", how="left")
    if out["group"].isna().any():
        missing = out.loc[out["group"].isna(), "sample"].tolist()[:10]
        raise ValueError(f"groups.csv missing labels for some samples, e.g. {missing} ...")

    # Map OA as positive (1), Control as negative (0)
    out[new_col] = out["group"].map(lambda x: 1 if x.upper() == "OA" else 0)
    return out

def top_variable_genes(X, n):
    v = X.var(axis=1, ddof=1)
    return X.loc[v.sort_values(ascending=False).head(min(n, X.shape[0])).index]

def encode_binary(y_series):
    classes = sorted(pd.unique(y_series.astype(str)))
    ybin = (y_series.astype(str) == classes[1]).astype(int).values
    return ybin, classes

def zscore_rows(df):
    arr = df.sub(df.mean(axis=1), axis=0)
    arr = arr.div(df.std(axis=1, ddof=1).replace(0, np.nan), axis=0)
    return arr.fillna(0.0)

def run_knn_cv_proba(Xg, ybin):
    Xt = Xg.T.values
    n = Xt.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to run cross-validation.")
    splits = max(2, min(CV_SPLITS, n))

    binc = np.bincount(ybin.astype(int))
    min_class = binc.min() if len(binc) > 1 else 0
    can_stratify = (min_class >= splits) and (len(binc) == 2)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE) if can_stratify \
         else KFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)

    # cap k by train fold size
    n_train = max(1, int(n * (splits - 1) / splits))
    k_eff = max(1, min(KNN_K, n_train))

    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_eff))
    proba = cross_val_predict(pipe, Xt, ybin, cv=cv, method="predict_proba")[:, 1]
    return proba

def main():
    for p in [EXPR_PATH, META_PATH, GROUPS_PATH]:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            sys.exit(f"ERROR: '{p}' is missing or empty.")

    out_metrics = os.path.join("results", "metrics")
    out_preds   = os.path.join("results", "preds_by_model")
    out_genes   = os.path.join("results", "genes")
    out_figs    = os.path.join("results", "figs")
    ensure_dirs(out_metrics, out_preds, out_genes, out_figs)

    # Load data and align by sample ID
    X, meta = load_data(EXPR_PATH, META_PATH, SAMPLE_COL)

    # Attach A1 labels from groups.csv (OA vs Control)
    meta = attach_groups_as_A1(meta, GROUPS_PATH, new_col=A1_COL)

    # If you don't have clusters yet, stub a single cluster so files still write
    if CLUSTER_COL not in meta.columns:
        meta[CLUSTER_COL] = "C1"

    samples = meta["sample"]
    Xv = top_variable_genes(X, TOPK_FOR_BASE)

    # ===== A1 (binary) =====
    yA1 = meta[A1_COL].astype(int)
    # build classes list as strings for pred label output
    classes = ["Control", "OA"]
    ybin = yA1.values
    probaA1 = run_knn_cv_proba(Xv, ybin)
    aucA1 = roc_auc_score(ybin, probaA1)

    pd.Series({f"AUC_{TAG}_A1": aucA1}).to_csv(f"{out_metrics}/{TAG}_A1_auc.csv")
    pd.DataFrame({"sample": samples, "true": [classes[i] for i in ybin], "proba_pos": probaA1}) \
        .to_csv(f"{out_preds}/{TAG}_A1_preds.csv", index=False)

    predA1 = (probaA1 >= 0.5).astype(int)
    predA1_label = np.where(predA1 == 1, classes[1], classes[0])
    pd.DataFrame({"sample": samples, "pred_group": predA1_label}) \
        .to_csv(f"{out_preds}/{TAG}_A1_predlabels.csv", index=False)

    # ===== A3 clusters (OVR) =====
    clusters = meta[CLUSTER_COL].astype(str).values
    unique_clusters = sorted(pd.unique(clusters))
    Xv.index.to_series().to_csv(f"{out_genes}/{TAG}_predictive_genes.txt", index=False)
    rows = []
    for c in unique_clusters:
        ybin_c = (clusters == c).astype(int)
        if ybin_c.sum() == 0 or ybin_c.sum() == len(ybin_c):
            rows.append({"cluster": c, "AUC": np.nan})
            continue
        proba_c = run_knn_cv_proba(Xv, ybin_c)
        auc_c = roc_auc_score(ybin_c, proba_c)
        rows.append({"cluster": c, "AUC": auc_c})
        pd.DataFrame({"sample": samples, "cluster": c, "proba_pos": proba_c}) \
            .to_csv(f"{out_preds}/{TAG}_OVR_{c}.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{out_metrics}/{TAG}_clusters_auc.csv", index=False)

    # ===== gene-count sweep (A1) =====
    sweep = []
    for k in GENE_SWEEP:
        k2 = min(k, X.shape[0])
        Xk = top_variable_genes(X, k2)
        proba_k = run_knn_cv_proba(Xk, ybin)
        auc_k = roc_auc_score(ybin, proba_k)
        sweep.append({"genes": k2, "AUC": auc_k})
    pd.DataFrame(sweep).to_csv(f"{out_metrics}/{TAG}_gene_sweep_A1.csv", index=False)

    # ===== basic heatmap (matplotlib only) =====
    genes = [g for g in Xv.index.tolist() if g in X.index]
    G = X.loc[genes]
    Z = zscore_rows(G)
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(Z.values, aspect="auto", cmap="RdBu_r")
    ax.set_title("Basic Heatmap (Top Variable Genes)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Genes")
    plt.colorbar(cax, ax=ax, label="Z-score Expression")
    plt.tight_layout()
    plt.savefig(f"{out_figs}/basic_heatmap_{TAG}.png", dpi=300)
    plt.close()

    print(f"[{TAG}] Done. A1 AUC = {aucA1:.4f}")
    print("Outputs saved to results/")

if __name__ == "__main__":
    main()
