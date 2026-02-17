#!/usr/bin/env python3
import os
import re
import sys
#usage: python3 00.errore_filter.py input.sam filtered.sam #error

# Compute aligned length from CIGAR string
def get_aligned_length(cigar):
    ops = re.findall(r'(\d+)([MIDNSHP=X])', cigar)
    return sum(int(n) for n, op in ops if op in ('M', 'I', '=', 'X'))

# Filter a single SAM file based on error rate
def filter_sam(input_file, output_file, max_error=0.02):
    total = kept = lost = 0

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if line.startswith('@'):
                fout.write(line)
                continue

            total += 1
            fields = line.strip().split('\t')
            if len(fields) < 6:
                continue

            cigar = fields[5]
            if cigar == '*':
                lost += 1
                continue

            aln_len = get_aligned_length(cigar)
            if aln_len == 0:
                lost += 1
                continue

            nm_match = re.search(r'NM:i:(\d+)', line)
            de_match = re.search(r'de:f:([\d\.eE-]+)', line)

            if de_match:
                error_rate = float(de_match.group(1))
            elif nm_match:
                error_rate = int(nm_match.group(1)) / aln_len
            else:
                lost += 1
                continue

            if error_rate <= max_error:
                fout.write(line)
                kept += 1
            else:
                lost += 1

    return total, kept, lost

# ----------------------------
# MAIN
# ----------------------------
if len(sys.argv) < 2:
    print("Usage: python3 script.py <file.sam|folder>")
    sys.exit(1)

target_path = sys.argv[1]

# Single SAM file
if os.path.isfile(target_path) and target_path.endswith('.sam'):
    output_path = target_path.replace('.sam', '_filtered.sam')
    print(f"Processing single file: {target_path}")
    total, kept, lost = filter_sam(target_path, output_path)
    perc = (lost / total * 100) if total > 0 else 0
    print(f"Total: {total} | Kept: {kept} | Lost: {lost} ({perc:.2f}%)")

# Folder with SAM files
elif os.path.isdir(target_path):
    sam_files = [f for f in os.listdir(target_path) if f.endswith('.sam')]
    if not sam_files:
        print(f"No SAM files found in {target_path}")
        sys.exit(0)

    print(f"\n=== Processing folder: {target_path} ===")
    for f in sam_files:
        input_path = os.path.join(target_path, f)
        output_path = os.path.join(target_path, f.replace('.sam', '_filtered.sam'))
        print(f"Processing {f}...")
        total, kept, lost = filter_sam(input_path, output_path)
        perc = (lost / total * 100) if total > 0 else 0
        print(f"Total: {total} | Kept: {kept} | Lost: {lost} ({perc:.2f}%)")

else:
    print(f"Invalid path: {target_path}")
    sys.exit(1)
