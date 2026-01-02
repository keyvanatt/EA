import pypandoc
import os

source_file = 'tst.rtf'
output_file = 'ProQuestDocuments-2025-12-16.md'

def clean_conversion():
    try:
        # 1. On force la lecture en RTF et la sortie en Markdown 'gfm' (plus robuste pour les tableaux)
        # On utilise 'extra_args' pour s'assurer que l'encodage ne saute pas
        output = pypandoc.convert_file(
            source_file, 
            'gfm', 
            format='rtf', 
            extra_args=['--wrap=none'] # Évite de couper les lignes au milieu des phrases
        )
        
        # 2. Sauvegarde propre
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
            
        print(f"✅ Conversion réussie dans {output_file}")
        
        # Petit test : affiche les 500 premiers caractères pour vérifier
        print("\n--- Aperçu du résultat ---")
        print(output[:500])
        
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    clean_conversion()