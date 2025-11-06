import pandas as pd

# --- Load your RF predictions ---
aidan = pd.read_csv(r'./results/ERP105501_RF_OA_vs_AMP_predictions.csv')
# Normalize columns (handle either SampleID present or Unnamed: 0)
if 'SampleID' not in aidan.columns and 'Unnamed: 0' in aidan.columns:
    aidan = aidan.rename(columns={'Unnamed: 0': 'SampleID'})
aidan = aidan.rename(columns={'PredictedLabel': 'Aidan_RF'})
aidan = aidan[['SampleID', 'Aidan_RF']]

# --- Load Hunter (logistic) ---
hunter = pd.read_csv(r'./results/logistic_sample_by_model_predictions_Hunter.csv')
hunter = hunter.rename(columns={'Unnamed: 0': 'SampleID'})
# Use the 10000-genes predictions
label_map = {'OA': 'OA_TKR', 'Control': 'Amputation'}
hunter['Hunter_Logistic'] = hunter['group_pred_10000'].map(label_map)
hunter['Hunter_Cluster']  = hunter['cluster_pred_10000'].astype(str)
hunter = hunter[['SampleID', 'Hunter_Logistic', 'Hunter_Cluster']]

# --- Load KNN (third model) ---

knn = pd.read_csv(r'./results/knn_A1_predlabels.csv')
knn = knn.rename(columns={'sample': 'SampleID', 'pred_group': 'KNN_Model'})
# Map to your label space if needed (only if they used OA/Control like Hunter)
knn['KNN_Model'] = knn['KNN_Model'].replace({'OA': 'OA_TKR', 'Control': 'Amputation'})
knn = knn[['SampleID', 'KNN_Model']]

# Find the column that holds the predicted class label
for cand in ['PredictedLabel','prediction','label','y_pred','pred']:
    if cand in knn.columns:
        knn = knn.rename(columns={cand: 'KNN_Model'})
        break
else:
    # if no known name found, assume the last non-SampleID column
    non_sid = [c for c in knn.columns if c != 'SampleID']
    if not non_sid:
        raise ValueError("Could not find a prediction column in knn_A1_predlabels.csv")
    knn = knn.rename(columns={non_sid[-1]: 'KNN_Model'})
# Map to your label space if needed (only if they used OA/Control like Hunter)
knn['KNN_Model'] = knn['KNN_Model'].replace({'OA':'OA_TKR','Control':'Amputation'})
knn = knn[['SampleID','KNN_Model']]

# --- Merge all three ---
combined = (
    aidan
    .merge(hunter, on='SampleID', how='inner')
    .merge(knn, on='SampleID', how='inner')
)

combined.to_csv(r'./results/team_predictions_combined_3models.csv', index=False)
print("✅ Wrote ./results/team_predictions_combined_3models.csv")
print(combined.head(8).to_string(index=False))
