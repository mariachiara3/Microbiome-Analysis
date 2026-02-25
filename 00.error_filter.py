#!/usr/bin/env python3
import os
import re
import sys
import argparse

# usage:
#   python3 filtraggio_percentuale_errore_sam.py path/to/file.sam --max_error 0.02 --min_len 1300 --max_len 1700 --min_cov 0.90
#   python3 filtraggio_percentuale_errore_sam.py path/to/folder    --max_error 0.02 --min_len 1300 --max_len 1700 --min_cov 0.90

_CIGAR_RE = re.compile(r'(\d+)([MIDNSHP=X])')

def get_aligned_bases_on_read(cigar: str) -> int:
    """
    Basi della read effettivamente coinvolte nell'allineamento sul read:
      M + I + = + X
    (NON include S; NON include D/N perché sono sul riferimento)
    """
    ops = _CIGAR_RE.findall(cigar)
    return sum(int(n) for n, op in ops if op in ('M', 'I', '=', 'X'))

def extract_error_rate(line: str, aligned_bases: int):
    """
    Ritorna error_rate oppure None se non disponibile.
    Priorità:
      - de:f:<float> (se presente) usato direttamente
      - altrimenti NM:i:<int> / aligned_bases
    """
    de_match = re.search(r'\bde:f:([\d\.eE-]+)\b', line)
    if de_match:
        return float(de_match.group(1))

    nm_match = re.search(r'\bNM:i:(\d+)\b', line)
    if nm_match:
        return int(nm_match.group(1)) / aligned_bases

    return None

def filter_sam(input_file, output_file, max_error, min_len, max_len, min_cov):
    total_alignments = kept_alignments = lost_alignments = 0

    # contatori "perché ho perso i record"
    lost_unmapped = 0
    lost_malformed = 0
    lost_no_seq = 0
    lost_len = 0
    lost_zero_aln = 0
    lost_cov = 0
    lost_no_tags = 0
    lost_error = 0

    # contatori "quante read perdo completamente"
    read_ids_seen = set()
    read_ids_kept = set()

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if line.startswith('@'):
                fout.write(line)
                continue

            total_alignments += 1
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 11:
                lost_alignments += 1
                lost_malformed += 1
                continue

            read_id = fields[0]
            read_ids_seen.add(read_id)

            cigar = fields[5]
            if cigar == '*':
                lost_alignments += 1
                lost_unmapped += 1
                continue

            seq = fields[9]
            if seq == '*' or seq == '':
                lost_alignments += 1
                lost_no_seq += 1
                continue

            read_len = len(seq)

            # filtro lunghezza read (SEQ)
            if not (min_len <= read_len <= max_len):
                lost_alignments += 1
                lost_len += 1
                continue

            aligned_bases = get_aligned_bases_on_read(cigar)
            if aligned_bases <= 0:
                lost_alignments += 1
                lost_zero_aln += 1
                continue

            # filtro coverage
            cov = aligned_bases / read_len
            if cov < min_cov:
                lost_alignments += 1
                lost_cov += 1
                continue

            # filtro error rate
            error_rate = extract_error_rate(line, aligned_bases)
            if error_rate is None:
                lost_alignments += 1
                lost_no_tags += 1
                continue

            if error_rate <= max_error:
                fout.write(line)
                kept_alignments += 1
                read_ids_kept.add(read_id)  # la read ha almeno un allineamento valido
            else:
                lost_alignments += 1
                lost_error += 1

    breakdown = {
        "lost_unmapped": lost_unmapped,
        "lost_malformed": lost_malformed,
        "lost_no_seq": lost_no_seq,
        "lost_len": lost_len,
        "lost_zero_aln": lost_zero_aln,
        "lost_cov": lost_cov,
        "lost_no_tags": lost_no_tags,
        "lost_error": lost_error,
    }

    total_reads = len(read_ids_seen)
    kept_reads = len(read_ids_kept)
    lost_reads = total_reads - kept_reads  # read "perse completamente"

    return total_alignments, kept_alignments, lost_alignments, breakdown, total_reads, kept_reads, lost_reads

def pct(num, den):
    return (num / den * 100.0) if den > 0 else 0.0

def print_summary(total_aln, kept_aln, lost_aln, breakdown, total_reads, kept_reads, lost_reads):
    print(f"Alignments -> Total: {total_aln} | Kept: {kept_aln} | Lost: {lost_aln} ({pct(lost_aln, total_aln):.2f}%)")
    print("  Lost breakdown (alignments):")
    for k, v in breakdown.items():
        if v:
            print(f"   - {k}: {v} ({pct(v, total_aln):.2f}%)")

    print(f"\nReads -> Total: {total_reads}")
    print(f"Reads with ≥1 valid alignment: {kept_reads}")
    print(f"Reads lost completely: {lost_reads} ({pct(lost_reads, total_reads):.2f}%)")

# ----------------------------
# MAIN with argparse
# ----------------------------
parser = argparse.ArgumentParser(description="Filter SAM files by error rate + read length + coverage; also report reads lost completely")
parser.add_argument("input", help="Input SAM file or folder")

parser.add_argument("--max_error", type=float, default=0.02,
                    help="Maximum error rate per alignment (default 0.02 = 2%)")
parser.add_argument("--min_len", type=int, default=1300,
                    help="Minimum read length (SEQ length). Default 1300")
parser.add_argument("--max_len", type=int, default=1800,
                    help="Maximum read length (SEQ length). Default 1800")
parser.add_argument("--min_cov", type=float, default=0.90,
                    help="Minimum coverage: aligned_bases/read_length. Default 0.90")

args = parser.parse_args()

target_path = args.input

def process_one(input_path: str):
    output_path = input_path.replace('.sam', '_filtered.sam')
    print(f"Processing: {input_path}")
    print(f"Params: max_error={args.max_error} | len={args.min_len}-{args.max_len} | min_cov={args.min_cov}")
    total_aln, kept_aln, lost_aln, breakdown, total_reads, kept_reads, lost_reads = filter_sam(
        input_path, output_path, args.max_error, args.min_len, args.max_len, args.min_cov
    )
    print(f"Output: {output_path}")
    print_summary(total_aln, kept_aln, lost_aln, breakdown, total_reads, kept_reads, lost_reads)
    print("-" * 70)

# Single SAM file
if os.path.isfile(target_path) and target_path.endswith('.sam'):
    process_one(target_path)

# Folder with SAM files
elif os.path.isdir(target_path):

    print(f"\n=== Scansione ricorsiva della cartella: {target_path} ===\n")

    found = False

    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.endswith(".sam"):
                found = True
                input_path = os.path.join(root, file)
                process_one(input_path)

    if not found:
        print("Nessun file .sam trovato nelle sottocartelle.")
