import logging
from typing import Dict, List, Tuple, Any
import polars as pl

import numpy as np
import pandas as pd

# 1. Import necessary functions from the original file
# Assuming your original code file is named complexity_clusters.py
# and is accessible in the same directory or Python path.
from complexity_clusters import (
    compute_latent_and_cluster,  # Performs UMAP and DBSCAN
    markers_from_cluster,        # Helper to extract markers by label
)

logger = logging.getLogger(__name__)
# Set up logging for this file as well
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Fonction get_article_ids_dates_and_hours_for_clusters (dans cluster_analysis.py)

# Fonction get_article_ids_dates_and_hours_for_clusters (dans cluster_analysis.py)

# Fonction get_article_ids_dates_and_hours_for_clusters (dans cluster_analysis.py)

def get_article_ids_dates_and_hours_for_clusters(
    cluster_dataframes: List[pd.DataFrame],
    filtered_marker_df: pl.DataFrame
) -> List[pd.DataFrame]: 
    """
    Récupère, pour chaque cluster et pour chaque marker, la liste des IDs, la liste 
    des dates (colonne 'date') et la liste des heures (colonne 'hour') de publication 
    correspondantes, puis les ajoute dans le DataFrame du cluster.
    
    Args:
        cluster_dataframes: Liste des DataFrames de clusters.
        filtered_marker_df: Le DataFrame Polars initial contenant les colonnes 
                            'id', 'marker', 'date', et 'hour'.

    Returns:
        Une LISTE de DataFrames Pandas enrichis avec 'article_ids_list', 
        'publication_dates_list', et 'publication_hours_list'.
    """
    logger.info("Démarrage de la récupération des IDs, des dates et des heures d'articles par marqueur.")
    
    # Vérification des colonnes nécessaires
    required_cols = ["id", "marker", "date", "hour"]
    if not all(col in filtered_marker_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in filtered_marker_df.columns]
        logger.error(f"COLONNES MANQUANTES : {missing}. Les colonnes 'id', 'marker', 'date' et 'hour' sont requises.")
        raise pl.exceptions.ColumnNotFoundError(f"Les colonnes requises sont manquantes : {missing}")


    # 1. Création de la structure Marker -> Liste des (ID, Date, Heure)
    
    # Agrégation : l'ID, la Date et l'Heure sont agrégés en listes pour chaque marqueur
    article_info_by_marker = (
        filtered_marker_df
        .group_by("marker")
        .agg([
            pl.col("id").alias("article_ids"),
            pl.col("date").alias("publication_dates"), 
            pl.col("hour").alias("publication_hours")
        ])
    )

    # 2. Conversion en dictionnaire Python pour une recherche rapide
    marker_map = article_info_by_marker.to_dict()
    
    marker_lookup = {}
    for marker, ids, dates, hours in zip(marker_map['marker'], 
                                         marker_map['article_ids'], 
                                         marker_map['publication_dates'], 
                                         marker_map['publication_hours']):
        # Zippage de l'ID, de la Date et de l'Heure pour chaque article
        marker_lookup[marker] = list(zip(ids, dates, hours))
    
    logger.info("Mapping créé pour %d marqueurs uniques (IDs, Dates et Heures).", len(marker_lookup))

    enriched_cluster_dfs: List[pd.DataFrame] = []

    # 3. Itérer sur chaque cluster DataFrame et enrichir les données
    for cluster_df in cluster_dataframes:
        
        # Fonction pour obtenir et sérialiser les IDs, les Dates et les Heures
        def get_and_serialize_info(marker):
            info_triplets = marker_lookup.get(marker, [])
            
            # 1. Séparer les IDs, les Dates et les Heures
            ids = [triplet[0] for triplet in info_triplets]
            dates = [triplet[1] for triplet in info_triplets]
            hours = [triplet[2] for triplet in info_triplets]
            
            # 2. Sérialiser les trois listes en chaînes de caractères (format lisible)
            ids_str = ", ".join(ids)
            
            # Formatage : conversion des éléments en chaîne avant de joindre
            dates_str = ", ".join([str(d) for d in dates])
            hours_str = ", ".join([str(h) for h in hours])
            
            # La fonction retourne un tuple de trois chaînes
            return ids_str, dates_str, hours_str

        # --- CORRECTION DE L'ERREUR TypeERROR ---
        # 1. Appliquer la fonction à la Series 'marker'. Cela produit une Series de tuples.
        results_series = cluster_df['marker'].apply(get_and_serialize_info)
        
        # 2. Convertir la Series de tuples en DataFrame temporaire
        # On utilise tolist() pour décomposer les tuples, puis on crée un DataFrame
        # L'index est conservé pour assurer l'alignement avec cluster_df
        results_df = pd.DataFrame(results_series.tolist(), index=cluster_df.index)
        
        # 3. Assignation des colonnes du DataFrame temporaire (0, 1, 2)
        cluster_df['article_ids_list'] = results_df[0]
        cluster_df['publication_dates_list'] = results_df[1]
        cluster_df['publication_hours_list'] = results_df[2]
        
        enriched_cluster_dfs.append(cluster_df)
        logger.info("Cluster %d: colonnes IDs, Dates et Heures ajoutées.", cluster_df['cluster_id'].iloc[0])

    return enriched_cluster_dfs

def get_individual_clusters_markers_df(
    lift_matrix: np.ndarray,
    selected_markers: np.ndarray,
    markers_journals: np.ndarray,
    complexities: Dict[str, float],
    eps_dbscan: float = 0.10,
    min_samples_dbscan: int = 20,
    out_prefix: str = "cluster_output",
) -> List[pd.DataFrame]:
    """
    Performs clustering (UMAP + DBSCAN) using pre-computed inputs and returns 
    a list of pandas DataFrames, where each DataFrame contains the markers and 
    their complexities for one distinct cluster.

    Args:
        lift_matrix: The (symmetrized) lift matrix for the selected markers.
        selected_markers: An array of marker names corresponding to the lift_matrix indices.
        markers_journals: An array of publisher labels associated with the markers.
        complexities: A dictionary mapping marker names to their complexity scores.
        eps_dbscan: The DBSCAN epsilon parameter.
        min_samples_dbscan: The DBSCAN minimum samples parameter.
        out_prefix: Prefix for saving the clustering plots (passed to compute_latent_and_cluster).

    Returns:
        A list of pandas DataFrames. Each DataFrame represents a cluster and 
        contains 'cluster_id', 'marker', 'complexity', and 'publishers_label'.
        The cluster with label -1 (noise) is excluded.
    """
    logger.info("Starting latent embedding and clustering for cluster extraction.")
    
    # 1. Compute Latent Embedding and perform DBSCAN Clustering
    # This function is imported from complexity_clusters.py
    X_latent, labels = compute_latent_and_cluster(
        lift_matrix=lift_matrix,
        selected_markers=selected_markers,
        markers_journals=markers_journals,
        out_prefix=out_prefix,
        eps_dbscan=eps_dbscan,
        min_samples_dbscan=min_samples_dbscan
    )
    
    unique_labels = np.unique(labels)
    cluster_dfs = []
    
    logger.info("Extracting individual clusters from a total of %d labels (including noise).", len(unique_labels))
    
    # 2. Iterate through unique cluster labels and extract markers
    for cluster_label in unique_labels:
        # DBSCAN noise points have label -1. We usually skip them for cluster analysis.
        if cluster_label == -1:
            logger.info("Skipping DBSCAN noise cluster (-1).")
            continue
            
        # Get the markers belonging to the current cluster label
        # markers_from_cluster is imported from complexity_clusters.py
        markers = markers_from_cluster(labels, cluster_label, selected_markers)
        
        # Get corresponding publisher labels and complexities
        # We need the indices of the markers in the original selected_markers array
        indices_in_original = np.where(labels == cluster_label)[0]
        publishers_for_cluster = markers_journals[indices_in_original]
        complexities_for_cluster = [complexities.get(m, np.nan) for m in markers]
        
        # 3. Create a DataFrame for the cluster
        cluster_data = {
            'marker': markers,
            'complexity': complexities_for_cluster,
            # Note: markers_journals contains an array of arrays (or lists) of publisher labels
            'publishers_label': [p for p in publishers_for_cluster]
        }
        
        cluster_df = pd.DataFrame(cluster_data)
        
        # 4. Sort the cluster markers by complexity and add cluster ID
        cluster_df = cluster_df.sort_values(by='complexity', ascending=False).reset_index(drop=True)
        cluster_df.insert(0, 'cluster_id', int(cluster_label))
        
        cluster_dfs.append(cluster_df)
        
        print("Cluster %d extracted with %d markers.", int(cluster_label), len(markers))

    return cluster_dfs