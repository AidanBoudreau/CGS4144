import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# --------------------------
# CONFIG: choose Hunter's gene count to use (one of: 10, 100, 1000, 10000)
# --------------------------
GENE_COUNT = 10000

# --------------------------
# 1) Load your predictions (RF)
# --------------------------
aidan = pd.read_csv(r'./results/ERP105501_RF_OA_vs_AMP_predictions.csv')
# Keep only SampleID + your predicted label; name your column
aidan = aidan[['SampleID', 'PredictedLabel']].rename(columns={'PredictedLabel': 'Aidan_RF'})

# --------------------------
# 2) Load Hunter's predictions (logistic, multi-gene-count)
# --------------------------
hunter = pd.read_csv(r'./results/logistic_sample_by_model_predictions_Hunter.csv')
# Standardize SampleID column name
hunter = hunter.rename(columns={'Unnamed: 0': 'SampleID'})

# Map Hunter's class labels to your label space
label_map = {'OA': 'OA_TKR', 'Control': 'Amputation'}
gcol = f'group_pred_{GENE_COUNT}'
ccol = f'cluster_pred_{GENE_COUNT}'
if gcol not in hunter.columns or ccol not in hunter.columns:
    raise ValueError(f"Hunter file missing expected columns: {gcol} and/or {ccol}")

hunter['Hunter_Logistic'] = hunter[gcol].map(label_map)
# Make cluster predictions strings so they behave like labels (e.g., '0'/'1')
hunter['Hunter_Cluster'] = hunter[ccol].astype(str)

hunter = hunter[['SampleID', 'Hunter_Logistic', 'Hunter_Cluster']]

# --------------------------
# 3) (Optional) bring in Assignment 3 cluster assignment per sample (for context)
#    If you have it as code/metadata_clusters.tsv with columns: SampleID, Group
# --------------------------
try:
    clust_meta = pd.read_csv(r'./code/metadata_clusters.tsv', sep='\t')
    clust_meta = clust_meta.rename(columns={'refinebio_accession_code': 'SampleID', 'Group': 'Cluster_Assignment'})
    clust_meta = clust_meta[['SampleID', 'Cluster_Assignment']]
except Exception:
    clust_meta = pd.DataFrame(columns=['SampleID', 'Cluster_Assignment'])

# --------------------------
# 4) Merge into a samples × models table (class labels)
# --------------------------
merged = aidan.merge(hunter, on='SampleID', how='inner')
merged = merged.merge(clust_meta, on='SampleID', how='left')  # optional

# These are the columns that contain CLASS predictions from each model
class_model_cols = ['Aidan_RF', 'Hunter_Logistic']

# These are the columns that contain CLUSTER predictions from each model
# (Right now only Hunter provided cluster preds; add more if teammates send theirs)
cluster_model_cols = ['Hunter_Cluster']

# --------------------------
# 5) (a) For each sample: how many models predict each CLASS label?
# --------------------------
label_counts = (
    merged.melt(id_vars=['SampleID'], value_vars=class_model_cols,
                var_name='Model', value_name='Prediction')
          .pivot_table(index='SampleID', columns='Prediction', aggfunc='size', fill_value=0)
          .reset_index()
)

# --------------------------
# 6) (b) For each sample: how many models predict the SAME CLUSTER?
#     We define "cluster stability" as the max vote among cluster predictions for that sample.
# --------------------------
def max_vote_count(row):
    vc = row.value_counts()
    return int(vc.max()) if len(vc) else 0

if cluster_model_cols:
    merged['Cluster_Agreement_Count'] = merged[cluster_model_cols].apply(max_vote_count, axis=1)
    merged['Cluster_Unique_Predictions'] = merged[cluster_model_cols].nunique(axis=1)
else:
    merged['Cluster_Agreement_Count'] = 0
    merged['Cluster_Unique_Predictions'] = 0

# --------------------------
# 7) Define "class stability" analogously: max vote among class predictions
# --------------------------
merged['Class_Agreement_Count'] = merged[class_model_cols].apply(max_vote_count, axis=1)
merged['Class_Unique_Predictions'] = merged[class_model_cols].nunique(axis=1)

# --------------------------
# 8) (c) Correlate class stability vs cluster stability (Spearman), with BH/FDR
#     One correlation here; if you run multiple (e.g., per subgroup), put all p-values into the list.
# --------------------------
# Only compute if we actually have cluster predictions
if cluster_model_cols:
    rho, pval = spearmanr(merged['Class_Agreement_Count'], merged['Cluster_Agreement_Count'])
else:
    rho, pval = float('nan'), float('nan')

pvals = [pval] if pd.notna(pval) else []
if pvals:
    _, pvals_adj, _, _ = multipletests(pvals, method='fdr_bh')
    pval_adj = pvals_adj[0]
else:
    pval_adj = float('nan')

# --------------------------
# 9) Save outputs
# --------------------------
# Samples × models (class + cluster columns)
out_combined = merged[['SampleID'] + class_model_cols + cluster_model_cols + ['Cluster_Assignment',
                                                                              'Class_Agreement_Count',
                                                                              'Cluster_Agreement_Count',
                                                                              'Class_Unique_Predictions',
                                                                              'Cluster_Unique_Predictions']]
out_combined.to_csv(r'./results/team_predictions_combined.csv', index=False)

# Per-sample class label counts
label_counts.to_csv(r'./results/sample_label_counts.csv', index=False)

# Small summary CSV for the correlation
summary = pd.DataFrame([{
    'gene_count_used_for_hunter': GENE_COUNT,
    'n_models_for_class': len(class_model_cols),
    'n_models_for_cluster': len(cluster_model_cols),
    'spearman_rho_class_vs_cluster_stability': rho,
    'p_value': pval,
    'p_value_FDR': pval_adj
}])
summary.to_csv(r'./results/team_part3_correlation_summary.csv', index=False)

print("✅ Wrote:")
print("  results/team_predictions_combined.csv")
print("  results/sample_label_counts.csv")
print("  results/team_part3_correlation_summary.csv")
print(f"Spearman ρ (class stability vs cluster stability): {rho:.3f}, p={pval:.3g}, FDR={pval_adj:.3g}")
