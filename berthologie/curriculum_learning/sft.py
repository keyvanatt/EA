import torch
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import random
import os  # Ajouté pour vérifier l'existence du fichier
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from tqdm import tqdm
from scipy.stats import linregress

# --- 1. CONFIGURATION & REPRODUCTIBILITÉ ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FILE_PATH = 'dbscan_clusters_top1.xlsx'
CACHE_FILE = 'wikipedia_cache.csv' # Fichier de sauvegarde
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- 2. CHARGEMENT DU MODÈLE (RESET PARTIEL) ---
def load_selective_reset_model(num_layers_to_reset=3):
    print(f"\n[Model] Réinitialisation des {num_layers_to_reset} dernières couches...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    layers = model.model.layers
    total_layers = len(layers)
    with torch.no_grad():
        for i in range(total_layers - num_layers_to_reset, total_layers):
            layers[i].apply(model._init_weights)
            
    return model, tokenizer

# --- 3. RÉCUPÉRATION DONNÉES WIKIPEDIA ---
def get_wikipedia_definition(marker):
    search_query = str(marker).replace("_", " ")
    headers = {'User-Agent': 'ResearchProject/1.0 (chshrayane@gmail.com)'}
    
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        
        # 1. Recherche du titre exact
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "format": "json",
            "srlimit": 1
        }
        search_res = requests.get(search_url, params=search_params, headers=headers, timeout=5).json()
        search_results = search_res.get('query', {}).get('search', [])
        
        if not search_results: return "introuvable"
        correct_title = search_results[0]['title']

        # 2. Récupération de l'extrait LARGE
        # explaintext: True -> retire le HTML (indispensable pour l'entraînement)
        # exsentences: 40 -> demande beaucoup de phrases (Wikipédia tronquera au max possible)
        # exintro: False -> INDISPENSABLE pour sortir de l'introduction et avoir du corps de texte
        detail_params = {
            "action": "query",
            "prop": "extracts",
            "titles": correct_title,
            "format": "json",
            "explaintext": True,  
            "exsentences": 40,    
            "exintro": False      
        }
        
        resp = requests.get(search_url, params=detail_params, headers=headers, timeout=5).json()
        pages = resp.get('query', {}).get('pages', {})
        
        for page_id in pages:
            extract = pages[page_id].get('extract', "")
            if len(extract) < 50:  # Si l'extrait est trop court, on considère que c'est raté
                return "introuvable"
            return extract

        return "introuvable"
    except Exception:
        return "introuvable"

def format_qwen_prompt(marker, definition):
    return (f"<|im_start|>system\nYou are Wikipedia.<|im_end|>\n"
            f"<|im_start|>user\Tell me about: {marker}<|im_end|>\n"
            f"<|im_start|>assistant\n{definition}<|im_end|>")

# --- 4. PRÉPARATION DU DATASET (AVEC CACHE) ---
def prepare_data():
    # 1. Vérification si le cache existe déjà
    if os.path.exists(CACHE_FILE):
        print(f"--- Chargement des définitions depuis le cache : {CACHE_FILE} ---")
        full_df = pd.read_csv(CACHE_FILE)
    else:
        # 2. Si non, on procède au scraping complet
        print("--- Cache introuvable. Début du scraping Wikipedia ---")
        xl = pd.ExcelFile(FILE_PATH)
        all_dfs = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            if 'marker' in df.columns and 'complexity' in df.columns:
                df = df[['marker', 'complexity']].dropna().drop_duplicates(subset=['marker'])
                df['cluster_name'] = sheet
                # Normalisation intra-cluster pour l'entrelacement
                c_min, c_max = df['complexity'].min(), df['complexity'].max()
                df['norm_complexity'] = (df['complexity'] - c_min) / (c_max - c_min + 1e-6)
                print(df)
                all_dfs.append(df)

        full_df = pd.concat(all_dfs).reset_index(drop=True)
        print(f"Scraping Wikipedia pour {len(full_df)} termes...")
        tqdm.pandas()
        full_df['definition'] = full_df['marker'].progress_apply(get_wikipedia_definition)
        print(full_df)
        # Sauvegarde pour la prochaine fois
        full_df.to_csv(CACHE_FILE, index=False)
        print(f"--- Définitions sauvegardées dans : {CACHE_FILE} ---")

    # Filtrage des termes non trouvés
    return full_df[~full_df['definition'].str.contains("introuvable", na=False)].copy()

df_base = prepare_data()

# --- 5. ENTRAÎNEMENT ---
def run_training(strategy="curriculum"):
    model, tokenizer = load_selective_reset_model()
    
    if strategy == "curriculum":
        df_final = df_base.sort_values(by='norm_complexity', ascending=True)
    else:
        df_final = df_base.sample(frac=1, random_state=42).reset_index(drop=True)

    prompts = [format_qwen_prompt(m, d) for m, d in zip(df_final['marker'], df_final['definition'])]
    ds = Dataset.from_dict({"text": prompts})
    tokenized_ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=2048), batched=True)

    args = TrainingArguments(
        output_dir=f"./out_{strategy}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-5,
        logging_steps=1,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    return [log['loss'] for log in trainer.state.log_history if 'loss' in log]

# --- 6. ANALYSE STATISTIQUE & GRAPHIQUES ---
print("\nLancement des tests comparatifs...")
loss_curr = run_training("curriculum")
loss_rand = run_training("random")

def smooth(data, weight=0.8):
    last = data[0]; smoothed = []
    for p in data:
        val = last * weight + (1 - weight) * p
        smoothed.append(val); last = val
    return smoothed

# Calcul des métriques sur les 30 premiers steps
steps_zoom = 30
auc_curr = np.trapz(loss_curr[:steps_zoom])
auc_rand = np.trapz(loss_rand[:steps_zoom])
gain_auc = ((auc_rand - auc_curr) / auc_rand) * 100

slope_curr, _, _, _, _ = linregress(np.arange(steps_zoom), loss_curr[:steps_zoom])
slope_rand, _, _, _, _ = linregress(np.arange(steps_zoom), loss_rand[:steps_zoom])

# Graphique 1 : GLOBAL
plt.figure(figsize=(10, 5))
plt.plot(smooth(loss_curr), color='blue', label='Curriculum (Sémantique)')
plt.plot(smooth(loss_rand), color='red', label='Random (Baseline)')
plt.title("Convergence Globale")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("graph_global.png")

# Graphique 2 : ZOOM 0-30 (Preuve du Curriculum)
plt.figure(figsize=(10, 5))
plt.plot(loss_curr[:steps_zoom], 'o-', color='blue', label=f'Curriculum (Pente: {slope_curr:.3f})')
plt.plot(loss_rand[:steps_zoom], 'o-', color='red', label=f'Random (Pente: {slope_rand:.3f})')
plt.fill_between(range(steps_zoom), loss_curr[:steps_zoom], loss_rand[:steps_zoom], 
                 where=(np.array(loss_rand[:steps_zoom]) > np.array(loss_curr[:steps_zoom])),
                 color='green', alpha=0.2, label=f"Gain AUC: {gain_auc:.1f}%")
plt.title("Zoom Convergence Initiale (Steps 0-30)")
plt.xlabel("Steps"); plt.ylabel("Loss")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("graph_zoom_initial.png")

print(f"\nANALYSE DES 30 PREMIERS STEPS :")
print(f"-> Gain d'efficacité (AUC) : {gain_auc:.2f}%")
print(f"-> Vitesse de descente (Curriculum) : {slope_curr:.4f}")
print(f"-> Vitesse de descente (Random)     : {slope_rand:.4f}")