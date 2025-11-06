# Aidan_GMM_clusters.py
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# --- Load expression matrix ---
expr = pd.read_csv(r'./code/ERP105501.tsv', sep='\t')
expr = expr.set_index('Gene')
expr.columns = expr.columns.astype(str).str.strip()

# --- Subset to top variable genes (same logic as RF) ---
variances = expr.var(axis=1)
top_genes = variances.nlargest(5000).index
expr_sub = expr.loc[top_genes]

# --- Transpose: rows=samples, cols=genes ---
X = expr_sub.T.values

# --- Fit GMM with 2–6 components and pick best via BIC ---
best_bic = np.inf
best_k = None
best_gmm = None

for k in range(2, 7):
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
    gmm.fit(X)
    bic = gmm.bic(X)
    print(f'k={k}, BIC={bic:.2f}')
    if bic < best_bic:
        best_bic = bic
        best_k = k
        best_gmm = gmm

print(f'Best model: k={best_k}, BIC={best_bic:.2f}')

# --- Predict cluster assignments ---
clusters = best_gmm.predict(X)

# --- Save cluster assignments ---
out = pd.DataFrame({
    'SampleID': expr_sub.columns,
    'Aidan_Cluster': clusters
})
out.to_csv(r'./results/Aidan_GMM_clusters.csv', index=False)
print("✅ Saved Aidan's GMM clusters to ./results/Aidan_GMM_clusters.csv")
