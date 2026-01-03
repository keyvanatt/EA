import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
FILE_PATH = 'dbscan_clusters_top1.xlsx'
MODEL_NAME = 'bert-base-uncased'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Chargement de BERT pour analyse topologique sur {DEVICE}...")
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

# --- 2. EXTRACTION DES FEATURES LATENTES ---
def get_latent_metrics(texts):
    results = []
    for text in texts:
        clean_text = str(text).replace('_', ' ')
        inputs = tokenizer(clean_text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Couche 11 et 12 (dernières)
            h11 = outputs.hidden_states[-2].squeeze(0).mean(dim=0).cpu().numpy()
            h12 = outputs.hidden_states[-1].squeeze(0).mean(dim=0).cpu().numpy()
            
            # Magnitude énergétique
            mag = np.linalg.norm(h12)
            # Distance de transition (mouvement sémantique final)
            trans = np.linalg.norm(h12 - h11)
            
            results.append({'vec': h12, 'magnitude': mag, 'transition': trans})
    return results

# --- 3. ANALYSE PAR CLUSTER ---
print("Lancement de l'analyse topologique...")
final_stats = []

for cluster in df_final['cluster'].unique():
    sub_df = df_final[df_final['cluster'] == cluster].copy()
    if len(sub_df) < 10: continue
    
    # Extraction des vecteurs et métriques individuelles
    metrics = get_latent_metrics(sub_df['marker'].tolist())
    vectors = np.array([m['vec'] for m in metrics])
    sub_df['magnitude'] = [m['magnitude'] for m in metrics]
    sub_df['transition'] = [m['transition'] for m in metrics]
    
    # Calcul de la Densité Sémantique (Similarité Cosine intra-cluster)
    sim_matrix = cosine_similarity(vectors)
    # Proximité moyenne de chaque mot avec ses pairs du cluster
    sub_df['density'] = (sim_matrix.sum(axis=1) - 1) / (len(sub_df) - 1)
    
    # Calcul des corrélations de Spearman avec la complexité
    c_mag, _ = spearmanr(sub_df['complexity'], sub_df['magnitude'])
    c_trans, _ = spearmanr(sub_df['complexity'], sub_df['transition'])
    c_dens, _ = spearmanr(sub_df['complexity'], sub_df['density'])
    
    final_stats.append({
        'Cluster': cluster,
        'Corr_Magnitude': c_mag,
        'Corr_Transition': c_trans,
        'Corr_Densite': c_dens,
        'N': len(sub_df)
    })

df_stats = pd.DataFrame(final_stats)

# --- 4. VISUALISATION GÉNÉRALE ---
plt.figure(figsize=(14, 8))
df_plot = df_stats.melt(id_vars='Cluster', value_vars=['Corr_Magnitude', 'Corr_Transition', 'Corr_Densite'])
sns.boxplot(data=df_plot, x='variable', y='value', palette='Set3')
sns.stripplot(data=df_plot, x='variable', y='value', color='black', alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.title("Impact de la Complexité sur la Topologie Latente de BERT\n(Distribution des corrélations par cluster)")
plt.ylabel("Coefficient de Spearman")
plt.savefig('topology_impact_summary.png')

# --- 5. HEATMAP DE SYNTHÈSE ---
plt.figure(figsize=(10, 12))
df_stats_pivot = df_stats.set_index('Cluster')[['Corr_Magnitude', 'Corr_Transition', 'Corr_Densite']]
sns.heatmap(df_stats_pivot, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
plt.title("Heatmap Topologique : Complexité vs Métriques Latentes")
plt.savefig('topology_heatmap.png')

print("Analyse terminée. Fichiers générés : topology_impact_summary.png et topology_heatmap.png")