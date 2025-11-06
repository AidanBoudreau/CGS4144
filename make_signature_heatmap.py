# make_signature_heatmap.py
# Heatmap of union of predictive-signature genes, annotated with Assignment 1 groups.

import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # avoid any Tk issues
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

def zscore_rows(df):
    # z-score per gene (row) across samples (columns)
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)

def series_to_color_map(series, palette="Set2"):
    cats = pd.Categorical(series)
    colors = sns.color_palette(palette, n_colors=len(cats.categories))
    return series.map(dict(zip(cats.categories, colors))), dict(zip(cats.categories, colors))

def main(
    expr_path = r".\code\ERP105501.tsv",
    meta_path = r".\code\metadata_OA_vs_AMP_clean.tsv",    # Assignment 1 groups
    sample_col = "refinebio_accession_code",
    group_col  = "Group",
    # All feature-importance CSVs from your 10/100/1000/10000 runs:
    feat_glob  = r".\results\ERP105501_RF_OA_vs_AMP_*_feature_importances.csv",
    top_k_per_model = 100,                                  # take top-K per model; set None to use ALL
    out_prefix = r".\results\signature_heatmap"
):
    # --- Load expression (genes × samples) ---
    expr = pd.read_csv(expr_path, sep="\t").set_index("Gene")
    expr.columns = expr.columns.astype(str).str.strip()

    # --- Load metadata & align groups to expression sample order ---
    meta = pd.read_csv(meta_path, sep="\t")
    meta[sample_col] = meta[sample_col].astype(str).str.strip()
    meta[group_col]  = meta[group_col].astype(str).str.strip().replace({"nan":"NA","None":"NA"})
    meta = meta.set_index(sample_col).reindex(expr.columns)
    groups = meta[group_col].fillna("NA")

    # --- Collect union of signature genes from all model runs ---
    files = sorted(glob.glob(feat_glob))
    if not files:
        raise FileNotFoundError(f"No feature importance files found matching: {feat_glob}")

    sig_genes = set()
    for f in files:
        df = pd.read_csv(f)
        if "gene" not in df.columns or "importance" not in df.columns:
            continue
        df = df.sort_values("importance", ascending=False)
        if top_k_per_model is not None:
            df = df.head(top_k_per_model)
        sig_genes.update(df["gene"].astype(str))

    sig_genes = [g for g in sig_genes if g in expr.index]
    if len(sig_genes) == 0:
        raise ValueError("No signature genes found in expression matrix.")

    # Save list of genes actually used
    pd.Series(sig_genes, name="gene").to_csv(f"{out_prefix}_genes_used.csv", index=False)

    # --- Subset & z-score by gene ---
    expr_sig = expr.loc[sig_genes]
    expr_sig_z = zscore_rows(expr_sig).fillna(0.0)  # rare zero-variance rows → 0

    # --- Annotation sidebar (Assignment 1 groups) ---
    ann = pd.DataFrame(index=expr_sig_z.columns)
    ann["Group"] = groups.reindex(ann.index).astype(str)

    # map groups to colors
    col_colors_series, col_legend_map = series_to_color_map(ann["Group"], palette="Set2")
    col_colors = pd.DataFrame({"Group": col_colors_series})

    # --- Seaborn clustermap (row & column dendrograms included) ---
    sns.set(font_scale=0.9)
    cg = sns.clustermap(
        expr_sig_z,
        cmap="vlag", center=0,
        metric="euclidean", method="average",
        col_colors=col_colors,
        xticklabels=False, yticklabels=False,   # turn on yticklabels if your gene set is small
        figsize=(12, 10)
    )
    cg.ax_heatmap.set_xlabel("Samples")
    cg.ax_heatmap.set_ylabel("Signature genes")
    cg.fig.suptitle(
        "Heatmap of Predictive Signature Genes\nAnnotated by Sample Group (OA_TKR vs Amputation)",
        y=1.10,
        fontsize=13,
        fontweight="bold"
    )

    # --- Improved legend placement & readability ---
    handles = [Patch(color=c, label=lab) for lab, c in col_legend_map.items()]
    legend = cg.fig.legend(
        handles=handles,
        title="Sample Group",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        frameon=False,
        fontsize=10,
        title_fontsize=11
    )


    # --- Save ---
    out_png = f"{out_prefix}.png"
    out_pdf = f"{out_prefix}.pdf"
    cg.savefig(out_png, dpi=220, bbox_inches="tight")
    cg.savefig(out_pdf, dpi=220, bbox_inches="tight")
    plt.close("all")
    print(f"Saved heatmap: {out_png} and {out_pdf}")
    print(f"Genes used (union): {out_prefix}_genes_used.csv")
    print(f"N genes: {expr_sig_z.shape[0]} | N samples: {expr_sig_z.shape[1]}")

if __name__ == "__main__":
    main()
