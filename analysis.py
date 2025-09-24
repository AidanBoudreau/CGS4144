import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# ---------------------------
# Step 0: Load the data
# ---------------------------
expr = pd.read_csv("code/ERP105501.tsv", sep="\t", index_col=0)
meta = pd.read_csv("code/metadata_ERP105501.tsv", sep="\t")
groups = pd.read_csv("code/groups.csv")

# Merge group labels into metadata
meta = meta.merge(groups, left_on="refinebio_accession_code",
                  right_on="sample", how="left")

print("Expression matrix shape:", expr.shape)
print("Metadata shape:", meta.shape)


# ---------------------------
# Step 1: Density plot
# ---------------------------
expr_log = np.log2(expr + 1)
gene_medians = expr_log.median(axis=1)

plt.figure(figsize=(8,6))
sns.kdeplot(gene_medians, fill=True, color="skyblue")
plt.xlabel("Log2 Median Expression")
plt.ylabel("Density")
plt.title("Per-Gene Expression Density (ERP105501)")
plt.tight_layout()
plt.savefig("results/plots/step1_density.png", dpi=300)


# ---------------------------
# Step 2: PCA, t-SNE, UMAP
# ---------------------------
X = expr_log.T.loc[meta["refinebio_accession_code"]]
y = meta["group"]

palette = {"OA": "tab:orange", "Control": "tab:blue"}

# PCA
pca = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(6,5))
for g, c in palette.items():
    idx = y == g
    plt.scatter(pca[idx,0], pca[idx,1], c=c, label=g)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA of ERP105501")
plt.legend(title="Group")
plt.tight_layout()
plt.savefig("results/plots/step2_pca.png", dpi=300)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)
plt.figure(figsize=(6,5))
for g, c in palette.items():
    idx = y == g
    plt.scatter(tsne[idx,0], tsne[idx,1], c=c, label=g)
plt.xlabel("t-SNE1"); plt.ylabel("t-SNE2")
plt.title("t-SNE of ERP105501")
plt.legend(title="Group")
plt.tight_layout()
plt.savefig("results/plots/step2_tsne.png", dpi=300)

# UMAP
um = umap.UMAP(random_state=42).fit_transform(X)
plt.figure(figsize=(6,5))
for g, c in palette.items():
    idx = y == g
    plt.scatter(um[idx,0], um[idx,1], c=c, label=g)
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.title("UMAP of ERP105501")
plt.legend(title="Group")
plt.tight_layout()
plt.savefig("results/plots/step2_umap.png", dpi=300)


# ---------------------------
# Step 3: Differential Expression
# ---------------------------
oa_samples = meta.loc[meta["group"] == "OA", "refinebio_accession_code"]
ctrl_samples = meta.loc[meta["group"] == "Control", "refinebio_accession_code"]

results = []
for gene in expr_log.index:
    oa_vals = expr_log.loc[gene, oa_samples]
    ctrl_vals = expr_log.loc[gene, ctrl_samples]

    tstat, pval = ttest_ind(oa_vals, ctrl_vals, equal_var=False)
    lfc = oa_vals.mean() - ctrl_vals.mean()
    results.append([gene, lfc, pval])

deg = pd.DataFrame(results, columns=["gene", "log2FC", "pval"])
deg["FDR"] = multipletests(deg["pval"], method="fdr_bh")[1]
deg.sort_values("FDR", inplace=True)
deg.to_csv("results/tables/differential_expression.tsv", sep="\t", index=False)

# Volcano plot
plt.figure(figsize=(7,6))
sns.scatterplot(data=deg, x="log2FC", y=-np.log10(deg["pval"]),
                hue=deg["FDR"] < 0.05,
                palette={True:"red", False:"gray"},
                alpha=0.7, edgecolor=None)
plt.xlabel("Log2 Fold Change (OA vs Control)")
plt.ylabel("-Log10 P-value")
plt.title("Volcano Plot of Differential Expression")
plt.legend(title="FDR < 0.05", loc="upper right")
plt.tight_layout()
plt.savefig("results/plots/step3_volcano.png", dpi=300)

deg.head(50).to_csv("results/tables/top50_DEGs.tsv", sep="\t", index=False)
print("Step 3 complete")


# ---------------------------
# Step 4: Heatmap of Top DEGs
# ---------------------------
# Select top 50 DEGs by FDR
top_genes = deg.head(50)["gene"]

# Extract their expression
heatmap_data = expr_log.loc[top_genes, meta["refinebio_accession_code"]]

# Z-score normalize per gene
heatmap_data = (heatmap_data - heatmap_data.mean(axis=1).values[:, None]) / heatmap_data.std(axis=1).values[:, None]

# Map group colors
col_colors = meta["group"].map(palette)

# Plot heatmap
plt.figure(figsize=(12,8))
sns.clustermap(heatmap_data, col_colors=col_colors, cmap="vlag",
               xticklabels=False, yticklabels=True)
plt.savefig("results/plots/step4_heatmap.png", dpi=300)

print("Step 4 complete: Heatmap saved in results/plots/")
