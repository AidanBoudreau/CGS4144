from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import random 
from pandas import DataFrame 
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests

def get_most_variable_genes(df, n):
    gene_var = df.var(axis=1, numeric_only=True)
    top_genes= gene_var.nlargest(n).index
    return df.loc[top_genes]

def print_scatterplot(X_pca, pca, title):
    plt.figure(figsize=(7,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], s=40, alpha=0.8)
    plt.title(title)
    plt.xlabel(f"PCA PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PCA PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

def print_clusterplot(X_pca, pca, labels, title):
    plt.figure(figsize=(7,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10", s=50)
    plt.title(title)
    plt.xlabel(f"PCA PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PCA PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

def run_gmm_for_n_genes(df, n_genes, n_components=4, *, random_state=0,
                        covariance_type="diag", reg_covar=1e-4, n_init=10):
    """Subset to n_genes most variable, fit GMM on z-scored samples, return labels."""
    top = df.var(axis=1, numeric_only=True).nlargest(n_genes).index
    X = df.loc[top].T.values
    X_scaled = StandardScaler().fit_transform(X)
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type=covariance_type,
                          reg_covar=reg_covar,
                          n_init=n_init,
                          random_state=random_state)
    labels_out = gmm.fit_predict(X_scaled)
    return labels_out

def analyze_gmm_and_report(
    df: pd.DataFrame,
    gene_counts=(10, 100, 1000, 10000, 5000),
    k=4,
    group_labels: pd.Series | None = None,   # <— allow passing precomputed groups
    metadata_path: str | None = None,
    sample_col="SampleID",
    group_col="Group",
    heatmap_ref_genes=5000,              # which n to use for the annotated heatmap
    make_heatmap=True,
    save_prefix="gmm_stats"
):
    """
    Runs GMM for each `n` in gene_counts, builds:
      (A) pairwise chi^2 table between runs
      (B) cluster-vs-group chi^2 per run (Part 4a)
      (C) FDR-adjusted p-values across ALL tests (Part 4b)
    Optionally draws an annotated heatmap for `heatmap_ref_genes`.
    """
    samples = df.columns.astype(str).tolist()

    # 1) labels for each gene count (stable settings)
    labels_dict = {}
    for n in gene_counts:
        labels_dict[n] = run_gmm_for_n_genes(df, n_genes=n, n_components=k)

    # 2) Pairwise chi^2 between runs
    pairwise_rows = []
    for i in range(len(gene_counts)):
        for j in range(i+1, len(gene_counts)):
            nA, nB = gene_counts[i], gene_counts[j]
            a, b = labels_dict[nA], labels_dict[nB]
            cont = pd.crosstab(a, b)
            chi2, p, dof, _ = chi2_contingency(cont)
            pairwise_rows.append({"Test": f"GMM_k{k}: {nA} vs {nB} genes",
                                  "Chi2": chi2, "p_value": p, "DoF": dof})
    pairwise_tbl = pd.DataFrame(pairwise_rows)

    # 3) Cluster vs Group chi^2 per run
    if group_labels is None:
        if metadata_path is not None:
            try:
                meta = pd.read_csv(metadata_path, sep="\t").set_index(sample_col)
                meta = meta.reindex(samples)
                group_labels = meta[group_col].astype(str).fillna("NA")
            except Exception as e:
                print(f"⚠ Could not read metadata; filling Group='NA' ({e})")
                group_labels = pd.Series(["NA"] * len(samples), index=samples)
        else:
            print("⚠ No metadata provided; filling Group='NA'")
            group_labels = pd.Series(["NA"] * len(samples), index=samples)
    else:
        # ensure order & type
        group_labels = group_labels.reindex(samples).astype(str).fillna("NA")

    cluster_vs_group_rows = []
    for n in gene_counts:
        labs = pd.Series(labels_dict[n], index=samples, name=f"GMM_k{k}_n{n}")
        cont = pd.crosstab(labs, group_labels.rename("Group"))
        chi2, p, dof, _ = chi2_contingency(cont)
        cluster_vs_group_rows.append({"Test": f"GMM_k{k} @ {n} genes vs Group",
                                      "Chi2": chi2, "p_value": p, "DoF": dof})
    cluster_vs_group_tbl = pd.DataFrame(cluster_vs_group_rows)

    # 4) FDR adjust across ALL tests
    combined = pd.concat([pairwise_tbl, cluster_vs_group_tbl], ignore_index=True)
    reject, p_adj, _, _ = multipletests(combined["p_value"], method="fdr_bh")
    combined["p_adj (FDR)"] = p_adj
    combined["Significant (FDR<0.05)"] = reject

    # 5) Save tables
    pairwise_tbl.to_csv(f"{save_prefix}_pairwise.csv", index=False)
    cluster_vs_group_tbl.to_csv(f"{save_prefix}_cluster_vs_group.csv", index=False)
    combined.to_csv(f"{save_prefix}_combined_FDR.csv", index=False)
    print(f"Saved: {save_prefix}_pairwise.csv, {save_prefix}_cluster_vs_group.csv, {save_prefix}_combined_FDR.csv")

    # 6) Optional heatmap using the chosen run
    if make_heatmap:
        n = heatmap_ref_genes
        expr_n = get_most_variable_genes(df, n)
        annot_df = pd.DataFrame(index=expr_n.columns.astype(str))
        annot_df[f"GMM_k{k}_n{n}"] = pd.Series(labels_dict[n], index=expr_n.columns).astype(str)
        annot_df["Group"] = group_labels.loc[expr_n.columns].astype(str)

        def series_to_color_map(series, palette="tab20"):
            cats = pd.Categorical(series)
            colors = sns.color_palette(palette, n_colors=len(cats.categories))
            return series.map(dict(zip(cats.categories, colors)))

        col_colors = pd.concat([
            series_to_color_map(annot_df[f"GMM_k{k}_n{n}"], "tab20"),
            series_to_color_map(annot_df["Group"], "hls"),
        ], axis=1)
        col_colors.columns = [f"GMM_k{k}_n{n}", "Group"]

        sns.set(font_scale=0.9)
        cg = sns.clustermap(
            expr_n, cmap="viridis",
            metric="euclidean", method="average",
            col_colors=col_colors,
            xticklabels=False, yticklabels=False,
            figsize=(13, 10)
        )
        cg.ax_heatmap.set_xlabel("Samples")
        cg.ax_heatmap.set_ylabel("Genes")
        cg.fig.suptitle(
            f"Top {n} Most Variable Genes — Heatmap with Dendrograms\n"
            f"Annotations: GMM_k{k}_n{n} + Group", y=1.05
        )
        # legend
        handles = []
        for col in col_colors.columns:
            vals = annot_df[col].astype(str)
            cats = pd.Categorical(vals).categories
            pal = "tab20" if col != "Group" else "hls"
            colors = sns.color_palette(pal, n_colors=len(cats))
            handles.append(Patch(color="white", label=f"{col}:"))
            for lab, c in zip(cats, colors):
                handles.append(Patch(color=c, label=str(lab)))
        cg.ax_col_dendrogram.legend(handles=handles, loc="center",
                                    ncol=4, bbox_to_anchor=(0.5, 1.25),
                                    fontsize=8, frameon=False)
        out_png = f"{save_prefix}_clustermap_top{n}.png"
        cg.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.show()
        print(f"Saved: {out_png}")

    return {"pairwise": pairwise_tbl,
            "cluster_vs_group": cluster_vs_group_tbl,
            "combined_FDR": combined}

num_genes = 5000
n_components = 5

df = pd.read_csv("code/ERP105501.tsv", sep="\t").set_index("Gene")
most_var_genes = get_most_variable_genes(df, num_genes)

# 2) Build sample matrix (rows=samples, cols=genes) and z-score features
X = most_var_genes.T.values 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
# 3) Reduce to 2D with PCA (positions for the “map”) 
pca = PCA(n_components=2, random_state=0) 
X_pca = pca.fit_transform(X_scaled)

#print_scatterplot(X_pca, pca, title=f"Samples mapped by gene-expression similarity ({num_genes} genes)")

gmm = GaussianMixture(n_components) 
gmm.fit(X_scaled) 
labels = gmm.predict(X_scaled)
#print_clusterplot(X_pca, pca, labels, title=f"Gaussian Mixture Clustering (K={n_components}, {num_genes} genes)")



meta_path = "code/metadata_ERP105501.tsv"
GROUP_MODE = "disease"   # choose: "disease" | "sex" | "age_bins" | "custom"

meta_raw = pd.read_csv(meta_path, sep="\t")
meta = meta_raw.rename(columns={"refinebio_accession_code": "SampleID"})

if GROUP_MODE == "disease" and "refinebio_disease" in meta.columns:
    meta["Group"] = meta["refinebio_disease"].astype(str)
elif GROUP_MODE == "sex" and "refinebio_sex" in meta.columns:
    meta["Group"] = meta["refinebio_sex"].astype(str)
elif GROUP_MODE == "age_bins" and "MetaSRA_age" in meta.columns:
    bins = [-np.inf, 40, 60, 80, np.inf]
    AGE_BIN_LABELS = ["<40", "40-60", "60-80", "80+"]  # <- don't overwrite `labels`
    meta["Group"] = pd.cut(meta["MetaSRA_age"], bins=bins, labels=AGE_BIN_LABELS)
elif GROUP_MODE == "custom":
    d = meta.get("refinebio_disease", pd.Series(["unk"] * len(meta)))
    s = meta.get("refinebio_sex", pd.Series(["unk"] * len(meta)))
    meta["Group"] = (d.fillna("unk").astype(str) + "|" + s.fillna("unk").astype(str))
else:
    print("⚠ GROUP_MODE not available; defaulting Group='NA'")
    meta["Group"] = "NA"

meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
meta["Group"] = meta["Group"].astype(str).str.strip().replace({"nan": "NA", "None": "NA"})

expr_samples = df.columns.astype(str)
meta_aligned = meta.set_index("SampleID").reindex(expr_samples)
group_labels = meta_aligned["Group"].fillna("NA")

print("Group value counts (after align):")
print(group_labels.value_counts(dropna=False))

# ---------- Stats & Heatmap driver ----------
results = analyze_gmm_and_report(
    df,
    gene_counts=(10, 100, 1000, 10000),
    k=n_components,
    group_labels=group_labels,              # <— pass prepared groups here
    metadata_path=None,                     # (not needed since we pass group_labels)
    heatmap_ref_genes=5000,
    make_heatmap=False,
    save_prefix="ERP105501_gmm"
)

'''
#prep for chi-squared comparison of different gene counts
gene_counts = [10, 100, 1000, 10000]
labels_dict = {}

for n in gene_counts:
    labels = run_gmm_for_n_genes(df, n)
    labels_dict[n] = labels
    print(f"{n} genes → cluster sizes: {np.bincount(labels)}")

# Perform pairwise chi-squared comparisons
rows = []
for i in range(len(gene_counts)):
    for j in range(i+1, len(gene_counts)):
        a = labels_dict[gene_counts[i]]
        b = labels_dict[gene_counts[j]]
        cont = pd.crosstab(a, b)
        chi2, p, dof, exp = chi2_contingency(cont)
        rows.append({
            "Genes_A": gene_counts[i],
            "Genes_B": gene_counts[j],
            "Chi2": chi2,
            "p_value": p,
            "DoF": dof
        })

chi_table = pd.DataFrame(rows)
print("chi-squared results:")
print(chi_table.round(4))
'''
'''
# Annotation columns: your GMM labels (from the K=4 fit above) + Sample Group from metadata
annot_df = pd.DataFrame(index=samples)
annot_df["GMM_k4"] = pd.Series(labels, index=samples).astype(str)

# Try to load metadata; if not found or missing cols, fill Group as "NA"
meta_path = "code/metadata_ERP105501.tsv"
group_colname = "Group"
sample_colname = "SampleID"
try:
    meta = pd.read_csv(meta_path, sep="\t")
    if sample_colname in meta.columns and group_colname in meta.columns:
        meta = meta.set_index(sample_colname)
        meta = meta.reindex(samples)
        annot_df["Group"] = meta[group_colname].astype(str).fillna("NA")
    else:
        annot_df["Group"] = "NA"
        print(f"Note: metadata missing columns '{sample_colname}'/'{group_colname}'. Filled Group='NA'.")
except Exception as e:
    annot_df["Group"] = "NA"
    print(f"Note: couldn't read metadata at {meta_path}. Filled Group='NA'. ({e})")

# Map annotation columns to colors
def series_to_color_map(series, palette="tab20"):
    cats = pd.Categorical(series)
    colors = sns.color_palette(palette, n_colors=len(cats.categories))
    mapping = dict(zip(cats.categories, colors))
    return series.map(mapping), mapping

col_colors_parts = []
legend_maps = {}
for col in annot_df.columns:
    pal = "tab20" if col != "Group" else "hls"
    colored_col, cmap = series_to_color_map(annot_df[col], palette=pal)
    col_colors_parts.append(colored_col)
    legend_maps[col] = cmap
col_colors = pd.concat(col_colors_parts, axis=1)
col_colors.columns = annot_df.columns

# Build clustermap with row & column dendrograms
sns.set(font_scale=0.9)
cg = sns.clustermap(
    expr_5000,
    cmap="viridis",
    metric="euclidean",
    method="average",      # linkage
    col_colors=col_colors, # annotation sidebar rows
    xticklabels=False,
    yticklabels=False,
    figsize=(13, 10)
)

# Labels and title
cg.ax_heatmap.set_xlabel("Samples")
cg.ax_heatmap.set_ylabel("Genes")
cg.fig.suptitle(
    "Top 5000 Most Variable Genes — Heatmap with Row/Column Dendrograms\n"
    "Annotations: GMM_k4 + Sample Group",
    y=1.05
)

# Legend showing color ↔ label for each annotation row
handles = []
for ann_col, mapping in legend_maps.items():
    handles.append(Patch(color="white", label=f"{ann_col}:"))
    for label_txt, color in mapping.items():
        handles.append(Patch(color=color, label=str(label_txt)))

cg.ax_col_dendrogram.legend(
    handles=handles,
    loc="center",
    ncol=4,
    bbox_to_anchor=(0.5, 1.25),
    fontsize=8,
    frameon=False
)
plt.show()
'''