import pandas as pd
import numpy as np
import glob
import os

# ===== CONFIGURAZIONE =====
input_folder = "path/to/taxonomy_file/"  # cartella con i file *_tassonomia.txt
output_folder = os.path.join(input_folder)  # sottocartella per i risultati
N_SUBSAMPLE = 200000  # numero totale di elementi da campionare

# crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# ===== PROCESSO =====
for filepath in glob.glob(os.path.join(input_folder, "*tassonomia.txt")):
    print(f"Processing {filepath}...")
    
    # leggi il file
    df = pd.read_csv(filepath, sep="\t", header=None, names=["tassonomia", "count"])
    
    # calcola pesi proporzionali ai count
    weights = df["count"] / df["count"].sum()
    
    # estrazione randomica ponderata
    sampled = np.random.choice(df["tassonomia"], size=N_SUBSAMPLE, p=weights, replace=True)
    
    # riconta quante volte ogni tassonomia è stata estratta
    sampled_counts = pd.Series(sampled).value_counts().reset_index()
    sampled_counts.columns = ["tassonomia", "new_count"]
    
    # salva il risultato nella sottocartella SENZA header
    filename = os.path.basename(filepath).replace("tassonomia.txt", "subsampled_taxa.txt")
    outpath = os.path.join(output_folder, filename)
    sampled_counts.to_csv(outpath, sep="\t", index=False, header=False)
    
    print(f" -> Saved subsampled data to {outpath}")

print("Tutti i file sono stati processati.")
