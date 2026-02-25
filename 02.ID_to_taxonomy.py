#!/usr/bin/env python3
import os
import glob
import argparse

# ----------------------------
# Funzioni
# ----------------------------
def pad_taxonomy_to_7(taxonomy: str, warn=None, id_ref=None, max_warn=5):
    """
    Se taxonomy ha meno di 7 livelli (separati da ';'), aggiunge 'unknown'
    fino ad arrivare a 7.
    """
    parts = [p.strip() for p in taxonomy.split(';') if p.strip() != '']
    if len(parts) < 7:
        if warn is not None:
            warn["padded"] += 1
            if warn["padded"] <= max_warn:
                print(f"⚠️ WARNING: Tassonomia con {len(parts)} livelli per ID {id_ref}. Padding con 'unknown' fino a 7.")
                print(f"   → original: {taxonomy}")
        parts = parts + ["unknown"] * (7 - len(parts))
    return parts

def genera_chiave_aggregazione(full_taxonomy: str):
    """
    Crea una chiave a livello specie: genus_species.
    Gestisce eccezioni per uncultured_*.
    """
    parts = [p.strip() for p in full_taxonomy.split(';') if p.strip() != '']
    # qui assumiamo che sia già padded a 7 fuori da questa funzione
    if len(parts) < 7:
        return None

    genus = parts[5].lower()
    species = parts[6].lower()

    if species in ['uncultured_bacterium', 'uncultured_organism']:
        return f"{genus}_{species}"

    if '_' not in species:
        return f"{genus}_{species}"
    else:
        return species

def carica_tassonomia(taxonomy_file: str):
    tax_dict = {}
    with open(taxonomy_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                id_, taxonomy = parts
                tax_dict[id_] = taxonomy
    return tax_dict

def processa_counts(counts_file: str, tax_dict: dict):
    aggregated_data = {}
    stats = {
        "lines_total": 0,
        "lines_used": 0,
        "unknown_ids": 0,
        "padded_tax": 0,
    }
    warn = {"padded": 0}

    with open(counts_file, "r", encoding="utf-8") as f:
        for line in f:
            stats["lines_total"] += 1
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            id_, count_str = parts
            try:
                count = int(count_str)
            except ValueError:
                continue

            raw_tax = tax_dict.get(id_)
            if raw_tax is None:
                stats["unknown_ids"] += 1
                full_taxonomy = "unknown;unknown;unknown;unknown;unknown;unknown;unknown"
                padded_parts = ["unknown"] * 7
            else:
                padded_parts = pad_taxonomy_to_7(raw_tax, warn=warn, id_ref=id_)
                if len(padded_parts) < 7:
                    # non dovrebbe succedere, ma teniamolo robusto
                    padded_parts = padded_parts + ["unknown"] * (7 - len(padded_parts))
                # ricostruisco una full taxonomy coerente a 7 livelli
                full_taxonomy = ";".join(padded_parts)
                if len(raw_tax.split(';')) < 7:
                    stats["padded_tax"] += 1

            key = genera_chiave_aggregazione(full_taxonomy)
            if key is None:
                # fallback super-safe
                key = "unknown_unknown"

            stats["lines_used"] += 1

            if key in aggregated_data:
                aggregated_data[key]["count"] += count
            else:
                aggregated_data[key] = {
                    "full_taxonomy": full_taxonomy,
                    "count": count
                }

    stats["padded_tax"] = warn["padded"]
    return aggregated_data, stats

# ----------------------------
# MAIN
# ----------------------------
parser = argparse.ArgumentParser(
    description="Aggrega counts (*.txt) in tassonomia, padding a 7 livelli con 'unknown' quando manca."
)
parser.add_argument("input_dir", help="Directory radice dove cercare i file .txt (ricorsivo)")
parser.add_argument("--taxonomy_file", required=True, help="File mapping ID -> tassonomia")
parser.add_argument("--pattern", default="*.txt", help="Pattern input (default *.txt)")
parser.add_argument("--skip_suffix", default="_taxonomy.txt", help="Non processare file già taxonomy (default _taxonomy.txt)")
args = parser.parse_args()

tax_dict = carica_tassonomia(args.taxonomy_file)

files = glob.glob(os.path.join(args.input_dir, "**", args.pattern), recursive=True)
files = [
    f for f in files
    if os.path.isfile(f)
    and not f.endswith(args.skip_suffix)
]

if not files:
    print("Nessun file trovato da processare.")
    raise SystemExit(0)

print(f"Tassonomie caricate: {len(tax_dict)}")
print(f"File counts trovati: {len(files)}")

for counts_file in sorted(files):
    output_file = os.path.splitext(counts_file)[0] + args.skip_suffix

    aggregated_data, stats = processa_counts(counts_file, tax_dict)

    with open(output_file, "w", encoding="utf-8") as out:
        for data in aggregated_data.values():
            out.write(f"{data['full_taxonomy']}\t{data['count']}\n")

    print(f"✅ {os.path.basename(counts_file)} -> {os.path.basename(output_file)} | "
          f"righe usate {stats['lines_used']}/{stats['lines_total']} | "
          f"unknown_id {stats['unknown_ids']} | padded_tax {stats['padded_tax']}")
