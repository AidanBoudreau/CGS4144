import os, argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.cluster import SpectralClustering
from scipy.stats import spearmanr

from utils_a3 import read_table, select_top_variable, clustermap_with_annotations

EXPR_PATH = "code/ERP105501.tsv"
META_PATH = "code/metadata_ERP105501.tsv"
GROUPS_PATH = "code/groups.csv"
RESULTS_TABLES = "results/tables"
RESULTS_PLOTS = "results/plots"

def ensure_dirs():
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    os.makedirs(RESULTS_PLOTS, exist_ok=True)

# Step 1: log2(x+1) transform with numeric coercion
def log2p1(df):
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    return np.log2(df_numeric + 1)

# Step 2: standardize samples (z-score across genes)
def standardize(expr_df):
    return ((expr_df.T - expr_df.T.mean()) / expr_df.T.std().replace(0, np.nan)).fillna(0.0)

# Step 3: train logistic regression with cross-validation
def train_logistic(X, y, folds=5, rs=42, class_weight=None):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", class_weight=class_weight, max_iter=5000, random_state=rs))
    ])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rs)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    pipe.fit(X, y)
    return proba, pipe.named_steps["clf"].coef_.ravel()

def main():
    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=5000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sweep", nargs="+", type=int, default=[10,100,1000,10000])
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    ensure_dirs()

    # Load expression and metadata
    expr_raw = read_table(EXPR_PATH)
    meta = read_table(META_PATH)
    groups = pd.read_csv(GROUPS_PATH)
    meta = meta.merge(groups, left_on="refinebio_accession_code", right_on="sample", how="left")
    sample_ids = meta["refinebio_accession_code"].tolist()

    # Apply log2(x+1) transform
    expr_log = log2p1(expr_raw)[sample_ids]

    # Select top N most variable genes
    expr_top = select_top_variable(expr_log, n=args.topn)
    X5k = standardize(expr_top)

    # Encode group labels
    le_group = LabelEncoder()
    y_group = le_group.fit_transform(meta.set_index("refinebio_accession_code").loc[sample_ids, "group"].astype(str))

    # Spectral clustering k=2 (Assignment 3)
    sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", random_state=args.random_state)
    y_cluster = sc.fit_predict(X5k.values)
    cluster_series = pd.Series(y_cluster, index=sample_ids).astype(str)

    # Logistic regression for group prediction
    p_group, coef_group = train_logistic(X5k.values, y_group, folds=args.folds, rs=args.random_state)
    auc_group = roc_auc_score(y_group, p_group[:,1])
    print(f"Group AUC (5k genes): {auc_group:.4f}")

    pd.Series(coef_group, index=X5k.columns).abs().sort_values(ascending=False).to_csv(
        f"{RESULTS_TABLES}/logistic_group_coef_top{args.topn}.csv"
    )

    # One vs rest logistic regression for each cluster
    cluster_aucs = {}
    for cl in np.unique(y_cluster):
        y_bin = (y_cluster == cl).astype(int)
        p_bin, coef_bin = train_logistic(X5k.values, y_bin, folds=args.folds, rs=args.random_state, class_weight="balanced")
        auc_bin = roc_auc_score(y_bin, p_bin[:,1])
        cluster_aucs[cl] = auc_bin
        print(f"Cluster {cl} AUC (one vs rest): {auc_bin:.4f}")

        pd.Series(coef_bin, index=X5k.columns).abs().sort_values(ascending=False).head(200).to_csv(
            f"{RESULTS_TABLES}/logistic_cluster_{cl}_top200_coef.csv"
        )

    # Retrain across varying gene counts
    pred_matrix = pd.DataFrame(index=sample_ids)
    sweep_records = []
    model_signatures = {}

    for n in args.sweep:
        expr_n = select_top_variable(expr_log, n=n)
        Xn = standardize(expr_n)

        p_g, _ = train_logistic(Xn.values, y_group, folds=args.folds, rs=args.random_state)
        auc_g = roc_auc_score(y_group, p_g[:,1])
        pred_matrix[f"group_pred_{n}"] = le_group.inverse_transform((p_g[:,1] >= 0.5).astype(int))

        proba_all = np.zeros((Xn.shape[0], 2))
        for i in [0,1]:
            y_bin = (y_cluster == i).astype(int)
            p_i, _ = train_logistic(Xn.values, y_bin, folds=args.folds, rs=args.random_state, class_weight="balanced")
            proba_all[:, i] = p_i[:,1]
        pred_matrix[f"cluster_pred_{n}"] = pd.Series(np.argmax(proba_all, axis=1).astype(str), index=sample_ids)

        sweep_records.append({"n_genes": n, "auc_group": auc_g, "auc_cluster_mean_ovr": np.mean(list(cluster_aucs.values()))})
        model_signatures[n] = list(expr_n.index)

        pd.DataFrame({
            "sample_id": sample_ids,
            f"group_pred_{n}": pred_matrix[f"group_pred_{n}"],
            f"cluster_pred_{n}": pred_matrix[f"cluster_pred_{n}"]
        }).to_csv(f"{RESULTS_TABLES}/logistic_preds_{n}genes.csv", index=False)

    pd.DataFrame(sweep_records).to_csv(f"{RESULTS_TABLES}/logistic_auc_by_gene_count.csv", index=False)
    pred_matrix.to_csv(f"{RESULTS_TABLES}/logistic_sample_by_model_predictions.csv")

    # Stability scores
    group_cols = [c for c in pred_matrix.columns if c.startswith("group_pred_")]
    cluster_cols = [c for c in pred_matrix.columns if c.startswith("cluster_pred_")]

    def majority_frac(cols):
        return [max(np.unique(pred_matrix.loc[s, cols], return_counts=True)[1]) / len(cols) for s in sample_ids]

    stability = pd.DataFrame(index=sample_ids)
    stability["group_majority_frac"] = majority_frac(group_cols)
    stability["cluster_majority_frac"] = majority_frac(cluster_cols)
    stability["true_group"] = meta.set_index("refinebio_accession_code").loc[sample_ids, "group"].astype(str)
    stability["true_cluster"] = cluster_series
    stability.to_csv(f"{RESULTS_TABLES}/logistic_sample_stability_metrics.csv")

    # Spearmanr correlation
    rho, pval = spearmanr(stability["group_majority_frac"], stability["cluster_majority_frac"])
    with open(f"{RESULTS_TABLES}/stability_spearman.txt", "w") as f:
        f.write(f"spearman_r: {rho:.4f}\n pvalue: {pval:.4g}\n")

    # Consensus score
    pos_label = "OA"
    consensus = []
    for s in sample_ids:
        votes = [1 if str(pred_matrix.loc[s, col]) == pos_label else 0 for col in group_cols]
        consensus.append(np.mean(votes)
        )
    
    pd.Series(consensus, index=sample_ids, name="consensus_score").to_csv(f"{RESULTS_TABLES}/logistic_consensus_score.csv")
    print("Consensus AUC:", round(roc_auc_score(y_group, consensus), 4))

    # Heatmap of predictive genes
    union_genes = sorted(set().union(*model_signatures.values()))
    heat_df = expr_log.loc[union_genes]
    annot_df = pd.DataFrame({
        "group": meta.set_index("refinebio_accession_code").loc[sample_ids, "group"],
        "cluster": cluster_series
    }, index=sample_ids)

    # Clustermap cols
    group_palette = {"OA": "red", "Control": "blue"}
    group_colors = annot_df["group"].map(group_palette)
    col_colors = pd.DataFrame({"Group": group_colors})

    # Generate clustermap
    g = sns.clustermap(
        heat_df,
        row_cluster=True,
        col_cluster=True,
        cmap="vlag",
        figsize=(12, 10),
        xticklabels=False,
        yticklabels=False,
        col_colors=col_colors
    )

    # Add axis labels and title
    g.ax_heatmap.set_xlabel("Samples")
    g.ax_heatmap.set_ylabel("Predictive Genes")
    plt.title("Heatmap of Predictive Genes with Sample Group Annotation", pad=80)

    # Add legend manually to dendrogram axis
    for label in group_palette:
        g.ax_col_dendrogram.bar(0, 0, color=group_palette[label], label=label, linewidth=0)
    g.ax_col_dendrogram.legend(title="Group", loc="center", ncol=2)

    # Save figure
    g.savefig(f"{RESULTS_PLOTS}/logistic_predictive_heatmap.png")
    plt.close()

    print("Done. Outputs saved under results/tables and results/plots.")

if __name__ == "__main__":
    main()