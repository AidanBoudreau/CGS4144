import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

inp  = r".\results\team_predictions_combined_with_clusters.csv"
out_counts = r".\results\sample_label_counts_FINAL.csv"
out_corr   = r".\results\team_part3_correlation_FINAL.csv"
out_pred   = r".\results\team_predictions_with_stability_FINAL.csv"

df = pd.read_csv(inp)

# Class model columns (adjust names if yours differ)
class_cols = [c for c in df.columns if c in ['Aidan_RF','Hunter_Logistic','KNN_Model']]
if not class_cols:
    raise SystemExit('No class prediction columns found.')

# Cluster columns
cluster_cols = [c for c in df.columns if c.endswith('_Cluster')]
# Per-sample class vote counts
label_counts = (
    df.melt(id_vars=['SampleID'], value_vars=class_cols, var_name='Model', value_name='Pred')
      .pivot_table(index='SampleID', columns='Pred', aggfunc='size', fill_value=0)
      .reset_index()
)
label_counts.to_csv(out_counts, index=False)

# Stability metrics
def max_vote(row):
    vc = row.value_counts(dropna=False)
    return int(vc.max()) if len(vc) else 0

df['Class_Agreement_Count']    = df[class_cols].apply(max_vote, axis=1)
df['Class_Unique_Predictions'] = df[class_cols].nunique(axis=1)

if len(cluster_cols) >= 2:
    df['Cluster_Agreement_Count']    = df[cluster_cols].apply(max_vote, axis=1)
    df['Cluster_Unique_Predictions'] = df[cluster_cols].nunique(axis=1)
    rho, p = spearmanr(df['Class_Agreement_Count'], df['Cluster_Agreement_Count'])
    reject, p_adj, _, _ = multipletests([p], method='fdr_bh')
    corr = pd.DataFrame([{
        'n_models_class': len(class_cols),
        'n_models_cluster': len(cluster_cols),
        'spearman_rho': rho,
        'p_value': p,
        'p_value_FDR': p_adj[0],
        'significant_FDR_0.05': bool(reject[0])
    }])
else:
    # Cannot correlate with only one cluster source (constant input)
    df['Cluster_Agreement_Count']    = 1
    df['Cluster_Unique_Predictions'] = 1
    corr = pd.DataFrame([{
        'n_models_class': len(class_cols),
        'n_models_cluster': len(cluster_cols),
        'spearman_rho': np.nan,
        'p_value': np.nan,
        'p_value_FDR': np.nan,
        'significant_FDR_0.05': False
    }])

df.to_csv(out_pred, index=False)
corr.to_csv(out_corr, index=False)
print('Wrote:')
print(' ', out_counts)
print(' ', out_pred)
print(' ', out_corr)
