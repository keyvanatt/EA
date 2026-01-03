import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

# --- CONFIGURATION ---
FILE_PATH = 'dbscan_clusters_top1.xlsx'
MODEL_NAME = 'bert-base-uncased'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Initialisation de BERT sur {DEVICE}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME, output_attentions=True).to(DEVICE)
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

# --- 2. CALCUL DE LA SÉLECTIVITÉ (DIVERGENCE JSD) ---
def compute_jsd_selectivity(text):
    """
    Calcule la divergence moyenne entre les 12 têtes d'attention 
    de la dernière couche pour mesurer la spécialisation.
    """
    clean_text = str(text).replace('_', ' ')
    inputs = tokenizer(clean_text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # On extrait l'attention de la dernière couche (index -1)
        # Shape: (heads, seq_len, seq_len)
        last_layer_attn = outputs.attentions[-1].squeeze(0).cpu().numpy()
        
        num_heads = last_layer_attn.shape[0]
        jsd_values = []

        # Comparaison de chaque paire de têtes (i, j)
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                # On aplatit les distributions pour comparer les vecteurs de probabilité
                p = last_layer_attn[i].flatten()
                q = last_layer_attn[j].flatten()
                
                # JSD = Carré de la distance de Jensen-Shannon
                # On ajoute un epsilon pour la stabilité numérique
                jsd_val = jensenshannon(p, q, base=2) ** 2
                jsd_values.append(jsd_val)
        
        # Retourne la divergence moyenne entre toutes les têtes
        return np.mean(jsd_values)

print(f"Analyse de la sélectivité sur {len(df_final)} markers...")
df_final['selectivity_jsd'] = df_final['marker'].apply(compute_jsd_selectivity)

# --- 3. ANALYSE STATISTIQUE PAR CLUSTER ---
results = []
for cluster in df_final['cluster'].unique():
    mask = df_final['cluster'] == cluster
    sub_df = df_final[mask]
    
    if len(sub_df) < 5: continue # On ignore les trop petits clusters

    corr, pval = spearmanr(sub_df['complexity'], sub_df['selectivity_jsd'])
    results.append({
        'Cluster': cluster,
        'Correlation': corr,
        'p_value': pval,
        'Count': len(sub_df)
    })

df_res = pd.DataFrame(results).sort_values(by='Correlation', ascending=False)

# --- 4. VISUALISATION ---
plt.figure(figsize=(12, 10))
sns.set_style("whitegrid")

# Création du barplot
colors = ['#d73027' if x > 0 else '#4575b4' for x in df_res['Correlation']]
sns.barplot(data=df_res, x='Correlation', y='Cluster', palette=colors)

plt.axvline(0, color='black', linestyle='-', linewidth=1)
plt.title("Sélectivité de BERT vs Complexité des mots\n(Divergence de Jensen-Shannon entre les têtes)", fontsize=14)
plt.xlabel("Corrélation de Spearman (Positif = Spécialisation accrue)", fontsize=12)
plt.ylabel("Clusters", fontsize=12)

# Ajout des annotations de p-value
for i, row in enumerate(df_res.itertuples()):
    if row.p_value < 0.05:
        plt.text(row.Correlation, i, ' *', color='black', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('analyse_selectivite_jsd.png')

# --- 5. EXPORT ---
df_res.to_csv("resultats_selectivite_bert.csv", index=False)
print("\nTraitement terminé.")
print("- Graphique : analyse_selectivite_jsd.png")
print("- Données : resultats_selectivite_bert.csv")