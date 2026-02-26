#!/usr/bin/env python3
import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================
# File discovery + naming
# ================================================================
def find_taxonomy_files(root_dir: str, suffix: str, recursive: bool = True):
    pattern = os.path.join(root_dir, "**", f"*{suffix}") if recursive else os.path.join(root_dir, f"*{suffix}")
    files = glob.glob(pattern, recursive=recursive)
    files = [f for f in files if os.path.isfile(f)]
    return sorted(files)

def make_unique_sample_name(filepath: str, root_dir: str, suffix: str):
    """
    Nome campione unico basato sul path relativo:
      group/sample_taxonomy.txt -> group__sample
    """
    rel = os.path.relpath(filepath, root_dir)
    base = rel.replace(os.sep, "__")
    if base.endswith(suffix):
        base = base[:-len(suffix)]
    return base

# ================================================================
# Parsing taxonomy files
# ================================================================
def parse_taxonomy_file(filepath: str):
    """
    Atteso formato per riga:
      <full_taxonomy>\t<count>
    Ritorna dict: {taxonomy_string: count_int}
    """
    taxa_counts = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            taxonomy = parts[0].strip()
            count_str = re.sub(r"[^\d]", "", parts[1].strip())
            if count_str == "":
                continue
            count = int(count_str)
            if count <= 0:
                continue

            taxa_counts[taxonomy] = taxa_counts.get(taxonomy, 0) + count
    return taxa_counts

def build_table(files, root_dir, suffix):
    """
    Costruisce tabella campioni x taxa (counts) e mapping sample->group (sottocartella).
    """
    data = {}
    sample_to_group = {}
    for fp in files:
        sample = make_unique_sample_name(fp, root_dir, suffix)
        group = os.path.basename(os.path.dirname(fp))  # sottocartella immediata
        data[sample] = parse_taxonomy_file(fp)
        sample_to_group[sample] = group

    tabella = pd.DataFrame.from_dict(data, orient="index").fillna(0).astype(int)
    tabella.index.name = "sample"
    return tabella, sample_to_group

# ================================================================
# Rarefaction expectation
# ================================================================
def expected_species_at_x(abundances, x):
    """
    E[S(x)] = sum_i (1 - (1 - p_i)^x), p_i = abund_i / N.
    Se x > N (reads osservate), usa x_eff=N (satura a S_obs).
    """
    abund = np.asarray(abundances, dtype=float)
    abund = abund[abund > 0]
    N = abund.sum()
    if N <= 0:
        return 0.0

    x_eff = min(float(x), float(N))
    p = abund / N
    return float(np.sum(1 - (1 - p) ** x_eff))

def rarefaction_curve(abundances, num_points=800):
    abund = np.asarray(abundances, dtype=float)
    abund = abund[abund > 0]
    N = abund.sum()
    if N <= 0:
        return np.array([0.0]), np.array([0.0])
    p = abund / N
    x = np.linspace(1, N, num=num_points)
    y = np.sum(1 - (1 - p[:, None]) ** x, axis=0)
    return x, y

# ================================================================
# Plot rarefaction (group colors) + vertical line at 200k
# ================================================================
def plot_rarefaction(tabella, output_png, sample_to_group, num_points=800, subsample_depth=200000):
    # Okabe–Ito (colorblind-safe)
    PALETTE = [
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
    ]

    plt.figure(figsize=(10, 6))

    groups = sorted(set(sample_to_group.values()))
    group_to_color = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}

    groups_plotted = set()
    max_reads = 0.0
    curves_plotted = 0

    for sample in tabella.index:
        abund = tabella.loc[sample].values
        if np.sum(abund) <= 0:
            continue

        x, y = rarefaction_curve(abund, num_points=num_points)
        max_reads = max(max_reads, float(x.max()))
        curves_plotted += 1

        group = sample_to_group.get(sample, "unknown_group")
        color = group_to_color.get(group, "#000000")

        if group not in groups_plotted:
            plt.plot(x, y, color=color, alpha=0.75, linewidth=1.6, label=group)
            groups_plotted.add(group)
        else:
            plt.plot(x, y, color=color, alpha=0.35, linewidth=1.2)

    plt.axvline(
        subsample_depth,
        linestyle="--",
        linewidth=1.5,
        color="black",
        alpha=0.85,
        label=f"Subsampling ({subsample_depth:,} reads)"
    )

    plt.xlabel("Number of Reads")
    plt.ylabel("Expected Number of Unique Species/Taxa")
    plt.title("Rarefaction Curves")
    plt.grid(True, linestyle="--", alpha=0.3)

    if max_reads > 0:
        plt.xlim(0, max(max_reads * 1.02, subsample_depth * 1.05))

    plt.ticklabel_format(style="plain", axis="x")
    plt.legend(title="Groups", loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"✅ Plot rarefazione salvato: {output_png} (curve: {curves_plotted}, gruppi: {len(groups)})")

# ================================================================
# Compute "% observed species recovered at 200k"
# ================================================================
def compute_recovered_observed(tabella, sample_to_group, subsample_depth=200000):
    rows = []
    for sample in tabella.index:
        abund = tabella.loc[sample].values
        total_reads = int(np.sum(abund))
        S_obs = int(np.sum(np.asarray(abund) > 0))
        E_Sx = expected_species_at_x(abund, subsample_depth)
        perc = (E_Sx / S_obs * 100.0) if S_obs > 0 else np.nan

        rows.append({
            "sample": sample,
            "group": sample_to_group.get(sample, "unknown_group"),
            "total_reads": total_reads,
            "S_obs": S_obs,
            f"E_S_at_{subsample_depth}": E_Sx,
            "percent_observed_recovered": perc
        })

    df = pd.DataFrame(rows).sort_values(
        ["group", "percent_observed_recovered", "total_reads"],
        ascending=[True, False, False]
    )
    return df

# ================================================================
# Plot "tabellina" (table figure) with results
# ================================================================
def plot_results_table(df, output_png, subsample_depth=200000, max_rows=30):
    """
    Crea una figura con una tabella (matplotlib) per i risultati.
    Mostra le prime max_rows righe (ordinate come df).
    """
    df_show = df.copy()
    # arrotondamenti per renderla leggibile
    col_ES = f"E_S_at_{subsample_depth}"
    if col_ES in df_show.columns:
        df_show[col_ES] = df_show[col_ES].round(1)
    df_show["percent_observed_recovered"] = df_show["percent_observed_recovered"].round(2)

    # limita righe
    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)

    cols = ["group", "sample", "total_reads", "S_obs", col_ES, "percent_observed_recovered"]
    df_show = df_show[cols]

    fig_h = max(2.5, 0.35 * (len(df_show) + 1))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    title = f"% observed species recovered at {subsample_depth:,} reads (showing top {len(df_show)})"
    ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="left",
        colLoc="left"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"✅ Tabellina salvata: {output_png}")

# ================================================================
# MAIN
# ================================================================
parser = argparse.ArgumentParser(
    description="From *_taxonomy.txt: build sample×taxa table, plot rarefaction with group colors, and compute % observed species recovered at 200k."
)
parser.add_argument("input_dir", help="Directory radice dove cercare i file *_taxonomy.txt (ricorsivo)")
parser.add_argument("--suffix", default="_taxonomy.txt", help="Suffisso dei file taxonomy (default: _taxonomy.txt)")
parser.add_argument("--out_prefix", default="rarefaction", help="Prefisso output (default: rarefaction)")
parser.add_argument("--num_points", type=int, default=800, help="Punti per curva rarefazione (default 800)")
parser.add_argument("--subsample_depth", type=int, default=200000, help="Subsampling depth (default 200000)")
parser.add_argument("--table_rows", type=int, default=30, help="Numero massimo di righe nella tabellina (default 30)")
args = parser.parse_args()

root = args.input_dir
if not os.path.isdir(root):
    raise SystemExit(f"Errore: '{root}' non è una directory.")

files = find_taxonomy_files(root, args.suffix, recursive=True)
if not files:
    raise SystemExit(f"Nessun file trovato con suffisso '{args.suffix}' sotto {root}")

print(f"File taxonomy trovati: {len(files)}")

tabella, sample_to_group = build_table(files, root, args.suffix)

# Output: salvati nella directory input (root)
out_table_csv = os.path.join(root, f"{args.out_prefix}_table.csv")
out_raref_png = os.path.join(root, f"{args.out_prefix}_curves.png")
out_reco_csv  = os.path.join(root, f"{args.out_prefix}_recovered_observed_{args.subsample_depth}.csv")
out_tab_png   = os.path.join(root, f"{args.out_prefix}_recovered_observed_table_{args.subsample_depth}.png")

tabella.to_csv(out_table_csv)
print(f"✅ Tabella campioni×taxa salvata in: {out_table_csv}")

plot_rarefaction(
    tabella, out_raref_png, sample_to_group,
    num_points=args.num_points,
    subsample_depth=args.subsample_depth
)

df_rec = compute_recovered_observed(tabella, sample_to_group, subsample_depth=args.subsample_depth)
df_rec.to_csv(out_reco_csv, index=False)
print(f"✅ Risultati (% observed recovered) salvati in: {out_reco_csv}")

plot_results_table(df_rec, out_tab_png, subsample_depth=args.subsample_depth, max_rows=args.table_rows)

print("\n📊 Preview (prime 10 righe):")
print(df_rec.head(10).to_string(index=False))
