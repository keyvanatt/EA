import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
from scipy.stats import spearmanr

# --- CONFIGURATION ---
FILE_PATH = 'dbscan_clusters_top1.xlsx'
MODEL_NAME = 'bert-base-uncased'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Initialisation de BERT avec sortie d'attention sur {DEVICE}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# IMPORTANT: output_attentions=True pour obtenir les matrices d'attention
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

# --- 2. CALCUL DE L'ENTROPIE DE L'ATTENTION ---
def get_attention_entropy(text):
    """
    Calcule l'entropie moyenne de l'attention pour chaque couche.
    Entropie H = -sum(p * log(p))
    """
    clean_text = str(text).replace('_', ' ')
    inputs = tokenizer(clean_text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # attention shape: (couches, batch, têtes, seq_len, seq_len)
        attentions = outputs.attentions 
        
        layer_entropies = []
        for layer_attn in attentions:
            # On moyenne sur les têtes d'attention (dim 1)
            # On prend la distribution d'attention du premier jeton [CLS] ou la moyenne
            # Ici, calcul de l'entropie sur la matrice complète seq x seq
            
            # Éviter log(0) avec un petit epsilon
            epsilon = 1e-10
            attn_dist = layer_attn.squeeze(0) # (heads, seq, seq)
            
            # H = -sum(p * log(p)) sur la dernière dimension
            entropy_heads = -torch.sum(attn_dist * torch.log(attn_dist + epsilon), dim=-1)
            # Moyenne sur les jetons et sur les têtes pour avoir un score par couche
            mean_entropy = entropy_heads.mean().item()
            layer_entropies.append(mean_entropy)
            
    return layer_entropies

print(f"Analyse de l'indécision (entropie) sur {len(df_final)} markers...")
# On récupère une liste de 12 valeurs (1 par couche) pour chaque mot
all_entropies = df_final['marker'].apply(get_attention_entropy).tolist()
all_entropies_np = np.array(all_entropies) 

# --- 3. ANALYSE STATISTIQUE ---
results = []
for cluster_name in df_final['cluster'].unique():
    cluster_mask = df_final['cluster'] == cluster_name
    complexities = df_final.loc[cluster_mask, 'complexity'].values
    cluster_ents = all_entropies_np[cluster_mask]
    
    if len(complexities) < 5: continue

    for layer_idx in range(12): # BERT base a 12 couches d'attention
        layer_ents = cluster_ents[:, layer_idx]
        # Corrélation : si positive, + de complexité = + d'indécision (entropie)
        corr, pval = spearmanr(layer_ents, complexities)
        results.append({
            'Cluster': cluster_name, 'Layer': layer_idx + 1,
            'Correlation': corr, 'p_value': pval
        })

res_df = pd.DataFrame(results)

# --- 4. VISUALISATION ---
pivot_corr = res_df.pivot(index="Cluster", columns="Layer", values="Correlation")

# Ajout de la moyenne globale
mean_corr = pivot_corr.mean(axis=0).to_frame(name='MOYENNE GLOBALE').T
pivot_final = pd.concat([pivot_corr, mean_corr])



plt.figure(figsize=(12, 8))
sns.heatmap(pivot_final, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.axhline(y=len(pivot_corr), color='black', lw=4)
plt.title("Corrélation (Spearman) : Complexité du mot vs Indécision de BERT\n(Positif = Le mot complexe rend le modèle indécis)")
plt.xlabel("Couche de BERT")
plt.ylabel("Clusters de mots")
plt.savefig('correlation_complexite_indecision.png')

# Export des résultats
res_df.to_csv("stats_indecision_bert.csv", index=False)
print("\nAnalyse terminée. Rapport généré : correlation_complexite_indecision.png")