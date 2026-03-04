#!/usr/bin/env python3
"""
Subsampling WITHOUT replacement + global matrix output.

Per ogni file input che termina con --suffix (ricorsivo da input_root):
- subsample a --depth senza replacement
- salva SOLO il TXT dei counts accanto al file input:
    <base>_subsampled_taxa.txt   (tab-separated, no header)

Nella cartella input_root salva SOLO:
- subsampled_matrix_counts.csv
  righe = campioni (ordinati per trattamento secondo SUFFIX_ORDER)
  colonne = taxa
  valori = count subsampled (0 se assente)

Esempio:
python3 subsampling_matrix.py /path/to/root \
  --suffix "tassonomia.txt" --depth 135000 --seed 123 --repeat_threshold 3500000
"""

import os
import argparse
import time
import traceback
import numpy as np
import pandas as pd
import gc


# -----------------------
# Treatment ordering
# -----------------------
SUFFIX_ORDER = ["control", "ethanol", "bleach", "DNA_RNA_zap", "lyzo_bleach", "lyzo_zap"]

def _tokenize_path(sample_id: str):
    # normalizza separatori
    s = sample_id.replace("\\", "/").lower()
    # split path
    chunks = []
    for part in s.split("/"):
        if not part:
            continue
        # split per separatori comuni
        part = part.replace(".", "_").replace("-", "_")
        chunks.extend([t for t in part.split("_") if t])
    return chunks

def extract_treatment(sample_id: str, order=SUFFIX_ORDER):
    """
    Riconosce il trattamento cercando una sequenza di token.
    Priorità ai trattamenti più specifici (più token), es: lyzo_bleach prima di bleach.
    """
    tokens = _tokenize_path(sample_id)

    # prepara pattern come lista di token per ogni trattamento
    patterns = []
    for key in order:
        pat = key.lower().replace("-", "_").split("_")  # es: "lyzo_bleach" -> ["lyzo","bleach"]
        patterns.append((key, pat))

    # ordina per lunghezza pattern desc (più specifici prima), poi per ordine definito in SUFFIX_ORDER
    patterns.sort(key=lambda x: (-len(x[1]), order.index(x[0])))

    # cerca pattern come sottosequenza contigua nei tokens
    for key, pat in patterns:
        L = len(pat)
        for i in range(0, len(tokens) - L + 1):
            if tokens[i:i+L] == pat:
                return key

    return "unmatched"

def treatment_rank(sample_id: str, order=SUFFIX_ORDER):
    t = extract_treatment(sample_id, order=order)
    if t == "unmatched":
        return len(order) + 999
    return order.index(t)

# -----------------------
# Helpers
# -----------------------
def find_input_files(root_dir, suffix):
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(suffix):
                matches.append(os.path.join(dirpath, f))
    return sorted(matches)


def read_counts_file(filepath):
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["taxonomy", "count"],
        dtype={"taxonomy": str},
    )
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df = df[df["count"] > 0].copy()
    return df


def make_per_file_txt_path(input_filepath, in_suffix):
    base = os.path.basename(input_filepath)
    if base.endswith(in_suffix):
        base = base[: -len(in_suffix)]
    out_txt = os.path.join(os.path.dirname(input_filepath), base + "_subsampled_taxa.txt")
    return out_txt


# -----------------------
# hypergeometric sequential fallback (low RAM)
# -----------------------
def sample_counts_without_replacement_seq(counts, n, rng):
    counts = np.asarray(counts, dtype=int).copy()
    N = int(counts.sum())
    if n <= 0 or N <= 0:
        return np.zeros_like(counts)
    if n >= N:
        return counts.copy()

    sampled = np.zeros_like(counts)
    remaining_to_draw = int(n)
    remaining_total = int(N)

    for i in range(len(counts) - 1):
        c = int(counts[i])
        if c <= 0:
            continue

        ngood = c
        nbad = remaining_total - ngood

        if remaining_to_draw <= 0:
            draw = 0
        else:
            draw = rng.hypergeometric(ngood, max(0, nbad), remaining_to_draw)

        sampled[i] = int(draw)
        remaining_to_draw -= int(draw)
        remaining_total -= c

        if remaining_to_draw <= 0:
            break

    if remaining_to_draw > 0:
        sampled[-1] = min(int(counts[-1]), remaining_to_draw)

    return sampled


# -----------------------
# fast expansion + choice (fast but uses RAM proportional to N)
# -----------------------
def subsample_without_replacement_fast(df, depth, rng):
    taxa = df["taxonomy"].to_numpy(dtype=object)
    counts = df["count"].to_numpy(dtype=int)

    reads = np.repeat(taxa, counts)
    N = reads.size
    if N < depth:
        del reads
        gc.collect()
        return None, N

    sampled = rng.choice(reads, size=depth, replace=False)
    uniq, cts = np.unique(sampled, return_counts=True)
    df_sub = pd.DataFrame({"taxonomy": uniq, "count": cts.astype(int)})
    df_sub = df_sub.sort_values("count", ascending=False).reset_index(drop=True)

    del reads, sampled
    gc.collect()
    return df_sub, N


# -----------------------
# process one file
# -----------------------
def process_file(filepath, root, depth, in_suffix, rng, repeat_threshold, verbose=True):
    t_start = time.time()
    sample_id = os.path.relpath(filepath, root)  # per evitare collisioni

    try:
        df = read_counts_file(filepath)
        total_reads = int(df["count"].sum())

        if verbose:
            print("\n" + "=" * 70)
            print(f"START file: {filepath}")
            print(f"  sample_id: {sample_id}")
            print(f"  taxa={len(df):,} total_reads={total_reads:,}")

        if total_reads < depth:
            if verbose:
                print(f"  -> DISCARD (total_reads {total_reads:,} < depth {depth:,})")
                print(f"END file (time {time.time() - t_start:.2f}s)")
            return {
                "sample_id": sample_id,
                "input_file": os.path.abspath(filepath),
                "total_reads": total_reads,
                "status": "discarded_total_reads_below_depth",
                "method": None,
                "timing_seconds": time.time() - t_start,
            }, None

        use_fast = (total_reads <= repeat_threshold)

        if verbose:
            print(
                f"  chosen method: "
                f"{'FAST np.repeat + choice' if use_fast else 'SEQUENTIAL hypergeometric'} "
                f"(threshold={repeat_threshold:,})"
            )

        df_sub = None

        if use_fast:
            try:
                df_sub, _ = subsample_without_replacement_fast(df, depth, rng)
                method = "fast_repeat"
            except MemoryError:
                if verbose:
                    print("  -> MemoryError in FAST method: falling back to hypergeometric sequential")
                gc.collect()
                df_sub = None
                method = "hypergeometric_seq"
        else:
            method = "hypergeometric_seq"

        if df_sub is None:
            taxa = df["taxonomy"].to_numpy(dtype=object)
            counts = df["count"].to_numpy(dtype=int)
            sampled_counts = sample_counts_without_replacement_seq(counts, depth, rng)
            df_sub = pd.DataFrame({"taxonomy": taxa, "count": sampled_counts})
            df_sub = df_sub[df_sub["count"] > 0].copy()

        used_depth = int(df_sub["count"].sum())

        # write per-file TXT next to input
        out_txt = make_per_file_txt_path(filepath, in_suffix)
        df_sub.to_csv(out_txt, sep="\t", index=False, header=False)

        # series for global matrix
        s_counts = pd.Series(
            df_sub["count"].to_numpy(dtype=int),
            index=df_sub["taxonomy"].to_numpy(dtype=object),
            name=sample_id,
        )

        if verbose:
            print(f"  wrote per-file counts TXT: {out_txt}")
            print(f"  taxa_after={len(df_sub):,} used_depth={used_depth:,}")
            print(f"END file (time {time.time() - t_start:.2f}s)")
            print("=" * 70)

        return {
            "sample_id": sample_id,
            "input_file": os.path.abspath(filepath),
            "total_reads": total_reads,
            "status": "subsampled_without_replacement",
            "method": method,
            "used_depth": used_depth,
            "output_counts_txt": os.path.abspath(out_txt),
            "timing_seconds": time.time() - t_start,
        }, s_counts

    except Exception as e:
        if verbose:
            print("!!! Exception during processing file:")
            traceback.print_exc()
        return {
            "sample_id": sample_id,
            "input_file": os.path.abspath(filepath),
            "total_reads": None,
            "status": "error",
            "method": None,
            "timing_seconds": time.time() - t_start,
            "notes": str(e),
        }, None


# -----------------------
# main
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Subsampling + global counts matrix (ordered by treatment)")
    p.add_argument("input_root", help="root dir containing input files (recursive search)")
    p.add_argument("--suffix", default="taxonomy.txt", help="match input files ending with this suffix")
    p.add_argument("--depth", type=int, default=135000, help="subsample depth")
    p.add_argument("--seed", type=int, default=123, help="RNG seed")
    p.add_argument(
        "--repeat_threshold",
        type=int,
        default=1000,
        help="if total_reads <= repeat_threshold use fast repeat method; otherwise hypergeometric sequence",
    )
    p.add_argument(
        "--matrix_name",
        default="subsampled_matrix_counts.csv",
        help="filename of the global matrix written in input_root",
    )
    p.add_argument("--quiet", action="store_true", help="less printing")
    args = p.parse_args()

    root = os.path.abspath(args.input_root)
    if not os.path.isdir(root):
        raise SystemExit(f"Error: '{root}' is not a directory.")

    rng = np.random.default_rng(args.seed)
    verbose = not args.quiet

    if verbose:
        print(f"ROOT: {root}")
        print(f"Searching files with suffix '{args.suffix}' ...")

    files = find_input_files(root, args.suffix)

    if verbose:
        print(f"Found {len(files)} files.")
        print("Treatment order:", SUFFIX_ORDER)

    # raccogliamo (sample_id, series)
    sample_series = []
    summaries = []

    for fp in files:
        summary, s_counts = process_file(
            fp,
            root=root,
            depth=args.depth,
            in_suffix=args.suffix,
            rng=rng,
            repeat_threshold=args.repeat_threshold,
            verbose=verbose,
        )
        summaries.append(summary)
        if s_counts is not None and summary.get("status") == "subsampled_without_replacement":
            sample_series.append((summary["sample_id"], s_counts))

    # ordina per trattamento, poi alfabetico (stabile) dentro trattamento
    sample_series.sort(key=lambda x: (treatment_rank(x[0]), x[0].lower()))

    matrix_path = os.path.join(root, args.matrix_name)

    if sample_series:
        ordered_series = [s for _, s in sample_series]
        mat_counts = pd.DataFrame(ordered_series).fillna(0).astype(int)
        mat_counts.index.name = "sample"
        mat_counts.to_csv(matrix_path)

        if verbose:
            print(f"\nGlobal matrix written to: {matrix_path}")
            print(f"  shape: {mat_counts.shape[0]} samples x {mat_counts.shape[1]} taxa")

            # piccolo riepilogo di quanti campioni per trattamento
            counts_by_treat = {}
            for sid, _ in sample_series:
                key = extract_treatment(sid, SUFFIX_ORDER)
                counts_by_treat[key] = counts_by_treat.get(key, 0) + 1
            print("  samples by treatment:", counts_by_treat)
    else:
        pd.DataFrame(columns=["sample"]).to_csv(matrix_path, index=False)
        if verbose:
            print(f"\nNo valid subsampled samples. Wrote empty matrix to: {matrix_path}")


if __name__ == "__main__":
    main()
