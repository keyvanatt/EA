import pandas as pd
import logging
import ast
from pathlib import Path
import sys

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MAX_ARTICLES_PER_CLUSTER = 1

# --- FONCTIONS UTILITAIRES ---

def safe_list_eval(val):
    """Convertit de façon robuste une chaîne en liste Python."""
    if isinstance(val, list): return val
    if pd.isna(val) or val == "": return []
    
    if isinstance(val, str):
        val = val.strip()
        # Tentative de parsing liste Python standard
        if val.startswith('[') and val.endswith(']'):
            try:
                res = ast.literal_eval(val)
                return res if isinstance(res, list) else [res]
            except (ValueError, SyntaxError):
                pass 
    
    # Tentative de split par virgule
    if isinstance(val, str) and ',' in val:
        return [x.strip() for x in val.split(',')]
    
    return [val]

def process_cluster_limited(df):
    """
    Transforme le cluster en limitant arbitrairement à 100 articles max.
    """
    if df.empty: return df

    # 1. Nettoyage des éditeurs (on les garde en une seule chaîne pour info)
    if 'publishers_label' in df.columns:
        df['publishers_label'] = df['publishers_label'].apply(
            lambda x: ", ".join([str(i) for i in safe_list_eval(x)])
        )

    # 2. Conversion des colonnes cibles en listes
    sync_cols = ['article_ids_list', 'publication_dates_list', 'publication_hours_list']
    for col in sync_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_list_eval)

    # 3. Logique de limitation et synchronisation
    def limit_sync_and_zip(row):
        ids = row.get('article_ids_list', [])
        dates = row.get('publication_dates_list', [])
        hours = row.get('publication_hours_list', [])
        
        if not ids: return []

        # --- LIMITATION DRASTIQUE ICI ---
        # On coupe les listes à MAX_ARTICLES_PER_CLUSTER
        ids = ids[:MAX_ARTICLES_PER_CLUSTER]
        # On coupe aussi les dates/heures pour rester synchro
        dates = dates[:MAX_ARTICLES_PER_CLUSTER]
        hours = hours[:MAX_ARTICLES_PER_CLUSTER]
        # --------------------------------

        # On aligne les longueurs (au cas où dates/heures seraient plus courtes que IDs)
        max_l = len(ids)
        safe_dates = (dates + [None] * max_l)[:max_l]
        safe_hours = (hours + [None] * max_l)[:max_l]
        
        return list(zip(ids, safe_dates, safe_hours))

    # Application
    df['zipped_data'] = df.apply(limit_sync_and_zip, axis=1)

    # 4. Explosion (maintenant très rapide car max 100 lignes par cluster)
    df_exploded = df.explode('zipped_data')

    # 5. Extraction propre
    mask = df_exploded['zipped_data'].notna()
    if mask.any():
        unpacked = pd.DataFrame(df_exploded.loc[mask, 'zipped_data'].tolist(), index=df_exploded.loc[mask].index)
        df_exploded.loc[mask, 'article_id'] = unpacked[0]
        df_exploded.loc[mask, 'pub_date'] = unpacked[1]
        df_exploded.loc[mask, 'pub_hour'] = unpacked[2]

    # 6. Nettoyage colonnes
    cols_to_drop = sync_cols + ['zipped_data']
    df_final = df_exploded.drop(columns=[c for c in cols_to_drop if c in df_exploded.columns])
    
    return df_final

# --- EXÉCUTION ---

if __name__ == "__main__":
    try:
        from complexity_clusters import run_all 
        from cluster_analysis import (
            get_individual_clusters_markers_df, 
            get_article_ids_dates_and_hours_for_clusters
        )
    except ImportError as e:
        logging.error(f"Erreur d'importation : {e}")
        sys.exit(1)

    root_path = Path("/Data/rc/data/causalitylink_sample")
    
    logging.info("1. Calcul des clusters...")
    filtered_marker_df, selected_markers, conv, markers_journals, lift_matrix, complexities, labels = run_all(root_path)

    cluster_dataframes = get_individual_clusters_markers_df(
        lift_matrix=lift_matrix,
        selected_markers=selected_markers,
        markers_journals=markers_journals,
        complexities=complexities
    )

    logging.info("2. Enrichissement des données...")
    cluster_dataframes = get_article_ids_dates_and_hours_for_clusters(
        cluster_dataframes=cluster_dataframes,
        filtered_marker_df=filtered_marker_df
    )

    logging.info(f"3. Transformation (Limite : {MAX_ARTICLES_PER_CLUSTER} articles/cluster)...")
    final_processed_clusters = []
    
    for df in cluster_dataframes:
        try:
            processed_df = process_cluster_limited(df)
            final_processed_clusters.append(processed_df)
        except Exception as e:
            logging.error(f"Erreur cluster : {e}")

    # --- SAUVEGARDE SIMPLE ---
    # Plus besoin de chunking compliqué car les fichiers seront petits
    output_file = 'dbscan_clusters_top1.xlsx'
    
    logging.info(f"4. Sauvegarde vers {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for df in final_processed_clusters:
            if not df.empty:
                c_id = str(df['cluster_id'].iloc[0])
                sheet_name = f'Cluster_{c_id}'[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    logging.info("✅ Terminé ! Fichier généré proprement.")