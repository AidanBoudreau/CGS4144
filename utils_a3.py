import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests


def read_table(path, **kwargs):
    return pd.read_csv(path, sep="\t", **kwargs)


def log2p1(df):
    return np.log2(df + 1)


def select_top_variable(df, n=5000):
    vars_ = df.var(axis=1)
    top = vars_.sort_values(ascending=False).head(n).index
    return df.loc[top]


def chi2_compare(a, b):
    ct = pd.crosstab(a, b)
    if ct.size == 0:  # skip if empty table
        return np.nan, np.nan, np.nan, ct
    chi2, p, dof, _ = chi2_contingency(ct)
    return chi2, p, dof, ct


def adjust_table_pvals(df, pcol="p_value", method="fdr_bh"):
    _, p_adj, _, _ = multipletests(df[pcol], method=method)
    df["p_adj"] = p_adj
    return df


def clustermap_with_annotations(expr, annot, out_png, figsize=(12, 9)):
    col_colors = []

    for col in annot.columns:
        cats = pd.Series(annot[col]).astype(str)

        if isinstance(cats.index, pd.MultiIndex):
            cats.index = ['_'.join(map(str, idx)) for idx in cats.index]

        if isinstance(cats, pd.Categorical):
            cats = cats.astype(str)

        lut = dict(zip(cats.unique(), sns.color_palette("husl", len(cats.unique()))))
        col_colors.append(cats.map(lut).to_list())

    g = sns.clustermap(
        expr,
        method="average",
        metric="correlation",
        z_score=0,
        col_colors=col_colors,
        cmap="vlag",
        figsize=figsize
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
