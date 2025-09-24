# Step 5 + 6: Functional Enrichment Analysis for RNA-seq DEGs

import pandas as pd
import gseapy as gp
import mygene
import os
import matplotlib.pyplot as plt


# ---------------------------
# Load DEG results
# ---------------------------
deg = pd.read_csv("results/tables/differential_expression.tsv", sep="\t")
sig_genes = deg.loc[deg["FDR"] < 0.05, "gene"].dropna().tolist()
print(f"Loaded {len(sig_genes)} significant DEGs")


# ---------------------------
# Convert Ensembl → Gene Symbols if needed
# ---------------------------
if sig_genes and sig_genes[0].startswith("ENSG"):
    print("Detected Ensembl IDs, converting to gene symbols...")
    mg = mygene.MyGeneInfo()
    mapped = mg.querymany(sig_genes, scopes="ensembl.gene", fields="symbol", species="human")
    symbols = [entry["symbol"] for entry in mapped if "symbol" in entry]
    print(f"Converted {len(sig_genes)} Ensembl IDs into {len(symbols)} gene symbols")
    sig_genes = symbols
else:
    print("Gene IDs already appear to be gene symbols")

os.makedirs("results/enrichment", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


# ---------------------------
# Member 1: Aiden — ORA GO BP
# ---------------------------
print("\n🔹 Aiden running ORA (GO Biological Process) ...")

aiden_res = gp.enrichr(
    gene_list=sig_genes,
    gene_sets=["GO_Biological_Process_2021"],
    organism="Human",
    outdir=None,   # disable auto folder spam
    cutoff=0.25
)

if not aiden_res.results.empty:
    aiden_res.results.head(20).to_csv("results/tables/Aiden_ORA_GO_BP.tsv", sep="\t", index=False)
    print("Aiden results saved")

    # Barplot
    gp.barplot(aiden_res.res2d, title="Aiden ORA GO BP (Top 10)",
               top_term=10, ofname="results/figures/Aiden_ORA_GO_BP.png")
    print("Aiden barplot saved: results/figures/Aiden_ORA_GO_BP.png")
else:
    print("⚠Aiden: No enriched terms found")


# ---------------------------
# Member 2: Nhi — GSEA Reactome
# ---------------------------
print("\n Nhi running GSEA (Reactome pathways) ...")

nhi_rnk = deg[["gene", "log2FC"]].dropna().copy()

if nhi_rnk.iloc[0,0].startswith("ENSG"):
    mg = mygene.MyGeneInfo()
    mapped = mg.querymany(nhi_rnk["gene"].tolist(), scopes="ensembl.gene", fields="symbol", species="human")
    id_map = {entry["query"]: entry.get("symbol","") for entry in mapped if "symbol" in entry}
    nhi_rnk["gene"] = nhi_rnk["gene"].map(id_map)

nhi_rnk = nhi_rnk.dropna().drop_duplicates()
nhi_rnk["gene"] = nhi_rnk["gene"].str.upper()
nhi_rnk.to_csv("results/tables/Nhi_gene_ranking.rnk", sep="\t", index=False, header=False)

nhi_res = gp.prerank(
    rnk="results/tables/Nhi_gene_ranking.rnk",
    gene_sets="Reactome_2022",
    permutation_num=100,
    min_size=5,
    max_size=1000,
    outdir=None
)

if not nhi_res.res2d.empty:
    nhi_res.res2d.to_csv("results/tables/Nhi_GSEA_Reactome.tsv", sep="\t")
    print("Nhi results saved (table only, no barplot for GSEA)")
else:
    print("⚠ Nhi: No enriched terms found")

# ---------------------------
# Member 3: Hunter — ORA GO MF
# ---------------------------
print("\n Hunter running ORA (GO Molecular Function) ...")

hunter_res = gp.enrichr(
    gene_list=sig_genes,
    gene_sets=["GO_Molecular_Function_2021"],
    organism="Human",
    outdir=None,
    cutoff=0.25
)

if not hunter_res.results.empty:
    hunter_res.results.head(20).to_csv("results/tables/Hunter_ORA_GO_MF.tsv", sep="\t", index=False)
    print("Hunter results saved")

    # Barplot
    gp.barplot(hunter_res.res2d, title="Hunter ORA GO MF (Top 10)",
               top_term=10, ofname="results/figures/Hunter_ORA_GO_MF.png")
    print("Hunter barplot saved: results/figures/Hunter_ORA_GO_MF.png")
else:
    print("⚠ Hunter: No enriched terms found")


# ---------------------------
# Step 6: Compare
# ---------------------------
print("\n Step 6: Comparing enrichment results across members...")

try:
    aiden_tbl = pd.read_csv("results/tables/Aiden_ORA_GO_BP.tsv", sep="\t")
    nhi_tbl   = pd.read_csv("results/tables/Nhi_GSEA_Reactome.tsv", sep="\t")
    hunter_tbl= pd.read_csv("results/tables/Hunter_ORA_GO_MF.tsv", sep="\t")

    summary = pd.DataFrame({
        "Aiden_GO_BP": aiden_tbl["Term"].head(10),
        "Nhi_Reactome": nhi_tbl["Term"].head(10),
        "Hunter_GO_MF": hunter_tbl["Term"].head(10)
    })
    summary.to_csv("results/tables/team_enrichment_comparison.tsv", sep="\t", index=False)
    print("Comparison table saved: results/tables/team_enrichment_comparison.tsv")
except Exception as e:
    print(f"⚠ Could not create comparison table: {e}")

print("\n Step 5 + 6 complete! Tables + figures ready for report.")
