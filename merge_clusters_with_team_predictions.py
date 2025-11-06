import pandas as pd

# Load combined team predictions
team = pd.read_csv(r'./team_predictions_combined_3models.csv')

# Load your GMM cluster results
aidan = pd.read_csv(r'./results/Aidan_GMM_clusters.csv')

# Merge on SampleID
merged = team.merge(aidan, on='SampleID', how='left')

# Save to new file
merged.to_csv(r'./team_predictions_combined_with_clusters.csv', index=False)

print("✅ Merged Aidan GMM clusters into team predictions.")
print("Saved to: ./team_predictions_combined_with_clusters.csv")
