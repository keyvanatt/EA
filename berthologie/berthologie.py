import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') # Pour SSH / Environnements sans GUI
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
from scipy.stats import spearmanr

# --- CONFIGURATION ---
FILE_PATH = 'dbscan_clusters_top1.xlsx'
MODEL_NAME = 'bert-base-uncased'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Initialisation de BERT sur {DEVICE}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE)
model.eval()

# --- 1. CHARGEMENT DES DONNÉES ---
xl = pd.ExcelFile(FILE_PATH)
all_data = []
for sheet in xl.sheet_names:
    df = xl.parse(sheet)
    if 'marker' in df.columns and 'complexity' in df.columns:
        df = df[['marker', 'complexity']].dropna().drop_duplicates(subset=['marker'])
        df['cluster'] = sheet
        all_data.append(df)
df_final = pd.concat(all_data).reset_index(drop=True)

# --- 2. EXTRACTION DES ACTIVATIONS (13 COUCHES) ---
def get_all_layers_norms(text):
    clean_text = str(text).replace('_', ' ')
    inputs = tokenizer(clean_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        norms = []
        for layer_hidden in outputs.hidden_states:
            # Mean pooling + Norme L2
            avg_emb = layer_hidden.squeeze(0).mean(dim=0).cpu().numpy()
            norms.append(np.linalg.norm(avg_emb))
    return norms

print(f"Analyse de {len(df_final)} markers...")
all_norms = df_final['marker'].apply(get_all_layers_norms).tolist()
all_norms_np = np.array(all_norms) 

# --- 3. ANALYSE STATISTIQUE ---
results = []
for cluster_name in df_final['cluster'].unique():
    cluster_mask = df_final['cluster'] == cluster_name
    complexities = df_final.loc[cluster_mask, 'complexity'].values
    cluster_norms = all_norms_np[cluster_mask]
    
    if len(complexities) < 10: continue

    for layer_idx in range(13):
        layer_norms = cluster_norms[:, layer_idx]
        corr, pval = spearmanr(layer_norms, complexities)
        results.append({
            'Cluster': cluster_name, 'Layer': layer_idx,
            'Correlation': corr, 'p_value': pval
        })

res_df = pd.DataFrame(results)

# --- 4. PRÉPARATION DES MATRICES ET MOYENNES ---
pivot_corr = res_df.pivot(index="Cluster", columns="Layer", values="Correlation")
pivot_p = res_df.pivot(index="Cluster", columns="Layer", values="p_value")

# Ajout de la ligne de moyenne générale par couche
mean_corr_line = pivot_corr.mean(axis=0).to_frame(name='MOYENNE GÉNÉRALE').T
pivot_corr_with_mean = pd.concat([pivot_corr, mean_corr_line])

# --- 5. VISUALISATION DES HEATMAPS CLASSIQUES ---
print("Génération des heatmaps standards...")

# Heatmap 1 : Valeurs de Corrélation avec ligne de moyenne
plt.figure(figsize=(14, 10))
sns.heatmap(pivot_corr_with_mean, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
plt.axhline(y=len(pivot_corr), color='black', lw=4) # Séparation visuelle de la moyenne
plt.title("Corrélation entre Complexité et Activation (Spearman)\nLigne du bas = Tendance globale par couche")
plt.savefig('1_heatmap_correlation_complete.png')
plt.close()

# Heatmap 2 : Direction avec seuils (> 0.4 ou < -0.4)
def threshold_logic(x):
    if x > 0.4: return 1
    elif x < -0.4: return -1
    else: return 0

pivot_threshold = pivot_corr_with_mean.map(threshold_logic)

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_threshold, annot=pivot_corr_with_mean, fmt=".2f", 
            cmap=['#0571b0', '#f7f7f7', '#ca0020'], center=0)
plt.axhline(y=len(pivot_corr), color='black', lw=4)
plt.title("Direction de la Corrélation (Seuils > |0.4|)\nBleu < -0.4 | Rouge > 0.4")
plt.savefig('2_heatmap_direction_seuils.png')
plt.close()

# --- 6. ANALYSE PAR BLOCS DE COUCHES ---
print("Analyse par blocs (Initial, Intermédiaire, Final)...")

blocs_config = {
    'Initial (0-3)': [0, 1, 2, 3],
    'Intermédiaire (4-8)': [4, 5, 6, 7, 8],
    'Final (9-12)': [9, 10, 11, 12]
}

bloc_results = []
for bloc_name, layers in blocs_config.items():
    # Moyenne des corrélations du bloc pour chaque cluster
    mean_vals = pivot_corr[layers].mean(axis=1)
    for cluster, val in mean_vals.items():
        bloc_results.append({'Cluster': cluster, 'Bloc': bloc_name, 'Corr_Moyenne': val})

df_blocs = pd.DataFrame(bloc_results)
pivot_blocs = df_blocs.pivot(index="Cluster", columns="Bloc", values="Corr_Moyenne")
pivot_blocs = pivot_blocs[['Initial (0-3)', 'Intermédiaire (4-8)', 'Final (9-12)']] # Ordonner

# Heatmap des Blocs
plt.figure(figsize=(10, 10))
sns.heatmap(pivot_blocs, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
plt.title("Corrélation Moyenne par Blocs de Couches")
plt.savefig('3_heatmap_blocs.png')
plt.close()

# Boxplot de comparaison des Blocs
plt.figure(figsize=(10, 6))
sns.boxplot(x='Bloc', y='Corr_Moyenne', data=df_blocs, palette='coolwarm')
sns.stripplot(x='Bloc', y='Corr_Moyenne', data=df_blocs, color="black", size=4, alpha=0.3)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title("Distribution des Corrélations : Vers une spécialisation des couches profondes ?")
plt.ylabel("Corrélation de Spearman Moyenne")
plt.savefig('4_boxplot_blocs.png')
plt.close()

# --- 7. EXPORT DES DONNÉES ---
res_df.to_csv("stats_berthologie.csv", index=False)
print("\nTraitement terminé avec succès !")
print("Fichiers générés :")
print("- 1_heatmap_correlation_complete.png : Intensité réelle + Moyenne")
print("- 2_heatmap_direction_seuils.png    : Focus sur les corrélations fortes (>0.4)")
print("- 3_heatmap_blocs.png               : Vue simplifiée par étapes BERT")
print("- 4_boxplot_blocs.png               : Preuve visuelle de la tendance par profondeur")