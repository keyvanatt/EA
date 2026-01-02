import torch
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# --- 1. CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 
FILE_PATH = 'dbscan_clusters_top1.xlsx'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_selective_reset_model(num_layers_to_reset=5):
    print(f"\nChargement du modèle et réinitialisation des {num_layers_to_reset} dernières couches...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Chargement du modèle pré-entraîné
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Accès aux couches de l'architecture Qwen
    # Dans Qwen, les couches sont dans model.model.layers
    layers = model.model.layers
    total_layers = len(layers)
    
    # Réinitialisation des poids pour les N dernières couches
    # Cela efface la "mémoire" sémantique tout en gardant la structure du langage
    with torch.no_grad():
        for i in range(total_layers - num_layers_to_reset, total_layers):
            print(f"Réinitialisation de la couche {i}...")
            # On applique la fonction d'initialisation par défaut de Qwen sur la couche
            layers[i].apply(model._init_weights)
            
    return model, tokenizer

# --- 2. RÉCUPÉRATION DES DONNÉES (WIKIPEDIA) ---
def get_wikipedia_definition(marker):
    search_query = str(marker).replace("_", " ")
    headers = {'User-Agent': 'ResearchProject/1.0 (chshrayane@gmail.com)'}
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {"action": "query", "list": "search", "srsearch": search_query, "format": "json", "srlimit": 1}
        search_res = requests.get(search_url, params=search_params, headers=headers, timeout=5).json()
        search_results = search_res.get('query', {}).get('search', [])
        if not search_results: return "introuvable"
        correct_title = search_results[0]['title']
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{correct_title.replace(' ', '_')}"
        resp = requests.get(summary_url, headers=headers, timeout=5)
        if resp.status_code == 200: return resp.json().get('extract', "introuvable")
        return "introuvable"
    except: return "erreur"

def format_qwen_prompt(marker, definition):
    return (f"<|im_start|>system\nTu es un dictionnaire précis.<|im_end|>\n"
            f"<|im_start|>user\nDefine: {marker}<|im_end|>\n"
            f"<|im_start|>assistant\n{definition}<|im_end|>")

# --- 3. PRÉPARATION DU DATASET ---
print("--- Phase de préparation des données ---")
xl = pd.ExcelFile(FILE_PATH)
all_data = []
for sheet in xl.sheet_names:
    print("Processing", sheet)
    df = xl.parse(sheet)
    if 'marker' in df.columns and 'complexity' in df.columns:
        df = df[['marker', 'complexity']].dropna().drop_duplicates(subset=['marker'])
        df['cluster_name'] = sheet
        df['definition'] = df['marker'].apply(get_wikipedia_definition)
        df = df[~df['definition'].str.contains("introuvable|erreur")].copy()
        all_data.append(df)

df_base = pd.concat(all_data).reset_index(drop=True)
print(f"Markers valides : {len(df_base)}")

# --- 4. FONCTION D'ENTRAÎNEMENT ---
def run_training(strategy="curriculum"):
    # On reset les 5 dernières couches à chaque nouveau run
    model, tokenizer = load_selective_reset_model(num_layers_to_reset=5)
    
    if strategy == "curriculum":
        print(f"Stratégie : CURRICULUM (Tri Cluster + Complexité)")
        df_final = df_base.sort_values(by=['cluster_name', 'complexity'], ascending=[True, True])
    else:
        print(f"Stratégie : RANDOM (Shuffle)")
        df_final = df_base.sample(frac=1, random_state=42).reset_index(drop=True)

    prompts = [format_qwen_prompt(m, d) for m, d in zip(df_final['marker'], df_final['definition'])]
    ds = Dataset.from_dict({"text": prompts})
    tokenized_ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=256), batched=True)

    args = TrainingArguments(
        output_dir=f"./out_{strategy}",
        per_device_train_batch_size=2,
        num_train_epochs=2,
        learning_rate=2e-5, # 
        logging_steps=1,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    return [log['loss'] for log in trainer.state.log_history if 'loss' in log]

# --- 5. EXÉCUTION ET GRAPHique ---
loss_curriculum = run_training("curriculum")
loss_random = run_training("random")

def smooth(data, weight=0.9): 
    last = data[0]
    smoothed = []
    for point in data:
        val = last * weight + (1 - weight) * point
        smoothed.append(val)
        last = val
    return smoothed

plt.figure(figsize=(12, 6))
plt.plot(loss_curriculum, color='blue', alpha=0.15)
plt.plot(smooth(loss_curriculum), color='blue', lw=2, label='Curriculum (Layers Reset)')
plt.plot(loss_random, color='red', alpha=0.15)
plt.plot(smooth(loss_random), color='red', lw=2, label='Random (Layers Reset)')
plt.title("Impact du Curriculum Learning sur un modèle partiellement réinitialisé")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("reset_comparison.png")
print("\nExpérience terminée. Graphique : reset_comparison.png")