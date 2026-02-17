#!/usr/bin/env python3
import os
import glob

# Hardcoded generic paths
taxonomy_file = "path/to/seq_itgdb_taxa.txt"
input_folder = "path/to/counts_folder"

# Generate aggregation key for taxonomy
def generate_aggregation_key(taxonomy):
    parts = taxonomy.split(';')
    if len(parts) < 7:
        return None
    genus = parts[5].lower()
    species = parts[6].lower()
    if species in ['uncultured_bacterium', 'uncultured_organism']:
        return f"{genus}_{species}"
    if '_' not in species:
        return f"{genus}_{species}"
    return species

# Load taxonomy mapping: ID -> full taxonomy
tax_dict = {}
with open(taxonomy_file, 'r') as f:
    for line in f:
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            id_, taxonomy = parts
            tax_dict[id_] = taxonomy

# Process each counts file
for counts_file in glob.glob(os.path.join(input_folder, "*.txt")):
    output_file = os.path.splitext(counts_file)[0] + "_taxonomy.txt"
    aggregated_data = {}

    with open(counts_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            id_, count_str = parts
            count = int(count_str)

            full_taxonomy = tax_dict.get(id_, "unknown")
            aggregation_key = (generate_aggregation_key(full_taxonomy)
                               if full_taxonomy != "unknown" else "unknown")

            if aggregation_key:
                if aggregation_key in aggregated_data:
                    aggregated_data[aggregation_key]['count'] += count
                else:
                    aggregated_data[aggregation_key] = {
                        'full_taxonomy': full_taxonomy,
                        'count': count
                    }

    with open(output_file, 'w') as out:
        for data in aggregated_data.values():
            out.write(f"{data['full_taxonomy']}\t{data['count']}\n")

    print(f"Aggregated file saved: {output_file}")
