import os
import argparse
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

from utils_a3 import (
    read_table, log2p1, select_top_variable,
    chi2_compare, adjust_table_pvals, clustermap_with_annotations
)


def ensure_dirs():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)


def load_inputs():
    df = read_table("code/ERP105501.tsv")
    first = df.columns[0]

    if (not pd.api.types.is_numeric_dtype(df[first])) and df[first].is_unique:
        expr = df.set_index(first)

    else:
        expr = df.copy()
        expr.index.name = "gene_id"

    meta = read_table("code/metadata_ERP105501.tsv")
    groups = pd.read_csv("code/groups.csv")
    meta = meta.merge(groups, left_on="refinebio_accession_code", right_on="sample", how="left")

    sample_order = meta["refinebio_accession_code"].tolist()
    expr = expr[[c for c in sample_order if c in expr.columns]]

    return expr, meta


def run_all_methods(X, ks, random_state=42):
    out = {}

    for k in ks:
        out[("kmeans", k)] = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(X)
        out[("gmm", k)] = GaussianMixture(n_components=k, random_state=random_state).fit_predict(X)
        out[("spectral", k)] = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", random_state=random_state).fit_predict(X)
        out[("agglomerative", k)] = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        out[("kmedoids", k)] = KMedoids(n_clusters=k, random_state=random_state).fit_predict(X)

    return out


def standardize_samples(expr):
    X = ((expr.T - expr.T.mean(axis=0)) / expr.T.std(axis=0).replace(0, np.nan)).fillna(0.0)

    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--topn", type=int, default=5000)
    args = ap.parse_args()

    ensure_dirs()
    expr_raw, meta = load_inputs()

    expr_log = log2p1(expr_raw)
    expr_5k = select_top_variable(expr_log, n=args.topn)
    X5k = standardize_samples(expr_5k)

    ks = list(range(args.kmin, args.kmax + 1))
    labels = run_all_methods(X5k.values, ks)

    sample_ids = expr_5k.columns.tolist()
    annot = {"assn1_group": meta.set_index("refinebio_accession_code").loc[sample_ids, "group"].astype(str)}

    for (method, k), lab in labels.items():
        pd.DataFrame({"sample_id": sample_ids, f"{method}_k{k}": lab}).to_csv(
            f"results/tables/{method}_k{k}.csv", index=False
        )

        if k == 2:
            annot[f"{method}_k2"] = pd.Series(lab, index=sample_ids).astype(str)

    annot_df = pd.DataFrame(annot)
    clustermap_with_annotations(expr_5k, annot_df, "results/plots/A3_heatmap_5k.png", figsize=(13, 9))

    series_map = {"assn1_group": annot_df["assn1_group"].reset_index(drop=True)}

    for (method, k), lab in labels.items():
        series_map[f"{method}_k{k}"] = pd.Series(lab, index=sample_ids).reset_index(drop=True).astype(str)

    recs = []
    keys = list(series_map.keys())

    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a, b = keys[i], keys[j]
            chi2, p, dof, ct = chi2_compare(series_map[a], series_map[b])
            recs.append({"compare_a": a, "compare_b": b, "chi2": chi2, "dof": dof, "p_value": p})

    chi2_df = pd.DataFrame(recs)
    chi2_df = adjust_table_pvals(chi2_df, pcol="p_value", method="fdr_bh")
    chi2_df.sort_values(["p_adj", "p_value"], inplace=True)
    chi2_df.to_csv("results/tables/chi2_all_pairs.csv", index=False)

    sweep_counts = [10, 100, 1000, 10000]
    sweep_labels = {}

    for n in sweep_counts:
        expr_n = select_top_variable(expr_log, n=n)
        Xn = standardize_samples(expr_n)
        lab_n = KMeans(n_clusters=2, random_state=42, n_init="auto").fit_predict(Xn.values)
        sweep_labels[n] = pd.Series(lab_n, index=expr_n.columns, name=f"kmeans_k2_{n}")
        pd.DataFrame({"sample_id": expr_n.columns, f"kmeans_k2_{n}": lab_n}).to_csv(
            f"results/tables/kmeans_k2_{n}genes.csv", index=False
        )

    recs2 = []
    keys2 = ["assn1_group"] + [f"kmeans_k2_{n}" for n in sweep_counts]
    series_map2 = {"assn1_group": series_map["assn1_group"]}

    for n in sweep_counts:
        s = sweep_labels[n].reindex(sample_ids)
        series_map2[f"kmeans_k2_{n}"] = s.astype("Int64").astype(str)

    keys2 = list(series_map2.keys())

    for i in range(len(keys2)):
        for j in range(i+1, len(keys2)):
            a, b = keys2[i], keys2[j]
            aa = series_map2[a]
            bb = series_map2[b]
            ok = aa.notna() & bb.notna()
            chi2, p, dof, _ = chi2_compare(aa[ok], bb[ok])
            recs2.append({"compare_a": a, "compare_b": b, "chi2": chi2, "dof": dof, "p_value": p})

    sweep_df = pd.DataFrame(recs2)
    sweep_df = adjust_table_pvals(sweep_df, pcol="p_value", method="fdr_bh")
    sweep_df.sort_values(["p_adj", "p_value"], inplace=True)
    sweep_df.to_csv("results/tables/chi2_gene_count_sweep.csv", index=False)

    print("Assignment 3 pipeline complete.")
    print("• Heatmap: results/plots/A3_heatmap_5k.png")
    print("• Chi2 (all pairs): results/tables/chi2_all_pairs.csv")
    print("• Gene-count sweep: results/tables/chi2_gene_count_sweep.csv")


if __name__ == "__main__":
    main()
