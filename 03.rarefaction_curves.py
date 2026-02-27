#!/usr/bin/env python3
"""
Rarefaction + bootstrap subsampling (WITHOUT replacement) on taxonomy-count files.
- You provide a ROOT directory, and the script searches recursively for files ending in a suffix
  (default: "_taxonomy.txt") produced by your previous pipeline.
- Groups are inferred from the *parent folder name* of each file.
  (Users can organize folders as they prefer; the legend labels will match folder names.)
Outputs are saved to an output directory you choose (default: the root directory).

Expected input file format (tab-separated, no header required):
    <taxonomy>\t<count>
"""

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D


# ================================================================
# PAPER STYLE
# ================================================================
def set_paper_style(base_font=16):
    plt.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": base_font + 4,
        "axes.labelsize": base_font + 2,
        "xtick.labelsize": base_font,
        "ytick.labelsize": base_font,
        "legend.fontsize": base_font,
        "legend.title_fontsize": base_font + 1,
        "lines.linewidth": 2.0,
        "figure.dpi": 120,
        "savefig.dpi": 400,
    })


# ================================================================
# COLORS (user-provided palette)
# ================================================================
BASE_COLORS = {
    "control":      "#0C7BDC",  # intense blue
    "ethanol":      "#D35FB7",  # magenta/violet
    "bleach":       "#E66100",  # vivid orange
    "dna_rna_zap":  "#16EB16",  # apple green
    "lyzo_bleach":  "#FEFE62",  # lemon yellow
    "lyzo_cand":    "#40B0A6",  # turquoise/cyan
}

# This determines the *preferred order* of groups in legend and CSV, by suffix.
# You can change/extend these labels to match your folder naming convention.
SUFFIX_ORDER = ["control", "ethanol", "bleach", "dna_rna_zap", "lyzo_bleach", "lyzo_cand"]


# ================================================================
# FILE DISCOVERY + NAMING
# ================================================================
def find_taxonomy_files(root_dir: str, suffix: str, recursive: bool = True):
    pattern = os.path.join(root_dir, "**", f"*{suffix}") if recursive else os.path.join(root_dir, f"*{suffix}")
    files = glob.glob(pattern, recursive=recursive)
    return sorted([f for f in files if os.path.isfile(f)])


def make_unique_sample_name(filepath: str, root_dir: str, suffix: str):
    """
    Makes a unique sample ID from the relative path (stable across runs).
    Example: subdir/file_taxonomy.txt -> subdir__file
    """
    rel = os.path.relpath(filepath, root_dir)
    base = rel.replace(os.sep, "__")
    if base.endswith(suffix):
        base = base[:-len(suffix)]
    return base


# ================================================================
# PARSING TAXONOMY FILES
# ================================================================
def parse_taxonomy_file(filepath: str):
    """
    Expected (TSV):
        taxonomy \t count
    Returns dict {taxonomy: count}
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
            if not count_str:
                continue

            count = int(count_str)
            if count <= 0:
                continue

            taxa_counts[taxonomy] = taxa_counts.get(taxonomy, 0) + count

    return taxa_counts


def build_table(files, root_dir, suffix):
    """
    Builds:
      - table: DataFrame (samples x taxa) with integer counts
      - sample_to_group: sample -> group label (parent folder name)
      - groups_found: sorted list of group labels
    """
    data = {}
    sample_to_group = {}
    groups_found = set()

    for fp in files:
        sample = make_unique_sample_name(fp, root_dir, suffix)
        group = os.path.basename(os.path.dirname(fp))  # parent folder name
        groups_found.add(group)

        data[sample] = parse_taxonomy_file(fp)
        sample_to_group[sample] = group

    table = pd.DataFrame.from_dict(data, orient="index").fillna(0).astype(int)
    table.index.name = "sample"
    return table, sample_to_group, sorted(groups_found)


# ================================================================
# GROUP ORDER + COLOR ASSIGNMENT (generic, no hardcoded folder lists)
# ================================================================
def extract_suffix_from_group(group_name: str):
    """
    Try to infer a known suffix from the group folder name.
    Example folder names that will match:
      - "adults_control"
      - "nymphs_DNA_RNA_zap"
      - "control"
      - "something_lyzo_bleach_rep1"
    Matching is case-insensitive and '_' / '-' tolerant.

    Returns a canonical suffix from SUFFIX_ORDER, or None.
    """
    g = group_name.lower().replace("-", "_")
    # pick the longest match to avoid partial collisions
    candidates = []
    for suf in SUFFIX_ORDER:
        if suf in g:
            candidates.append(suf)
    if not candidates:
        return None
    return sorted(candidates, key=len, reverse=True)[0]


def build_group_order(groups):
    """
    Order groups by:
      1) suffix order (SUFFIX_ORDER)
      2) then alphabetical among groups with same suffix
      3) groups with unknown suffix last (alphabetical)
    """
    def key(g):
        suf = extract_suffix_from_group(g)
        if suf in SUFFIX_ORDER:
            return (0, SUFFIX_ORDER.index(suf), g.lower())
        return (1, 999, g.lower())

    return sorted(groups, key=key)


def build_group_palette(groups):
    """
    Assign colors by suffix using BASE_COLORS.
    If a group doesn't match any suffix -> fallback colormap.
    """
    groups_sorted = build_group_order(groups)
    fallback = mpl.colormaps.get_cmap("tab20").resampled(max(1, len(groups_sorted)))

    palette = {}
    for i, g in enumerate(groups_sorted):
        suf = extract_suffix_from_group(g)
        if suf is not None and suf in BASE_COLORS:
            palette[g] = BASE_COLORS[suf]
        else:
            palette[g] = fallback(i)

    return palette


# ================================================================
# RAREFACTION CURVE (analytic expectation, used for plotting)
# ================================================================
def expected_species_curve_multinomial(abundances, x):
    """
    Analytical expectation under multinomial sampling (WITH replacement).
    Used here only to generate smooth rarefaction-like curves for visualization.
    """
    abund = np.asarray(abundances, dtype=float)
    abund = abund[abund > 0]
    N = abund.sum()
    if N <= 0:
        return 0.0
    x_eff = min(float(x), float(N))
    p = abund / N
    return float(np.sum(1 - (1 - p) ** x_eff))


def rarefaction_curve_visual(abundances, num_points=800):
    abund = np.asarray(abundances, dtype=float)
    abund = abund[abund > 0]
    N = abund.sum()
    if N <= 0:
        return np.array([0.0]), np.array([0.0])

    x = np.linspace(1, N, num=num_points)
    y = np.array([expected_species_curve_multinomial(abund, xx) for xx in x], dtype=float)
    return x, y


# ================================================================
# BOOTSTRAP SUBSAMPLING WITHOUT REPLACEMENT (efficient; no array expansion)
# ================================================================
def sample_counts_without_replacement_seq(counts, n, rng):
    """
    Multivariate hypergeometric sampling implemented as sequential conditional hypergeometric draws.
    Returns sampled counts per category (same order as counts), WITHOUT replacement.
    """
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
        draw = rng.hypergeometric(ngood, nbad, remaining_to_draw)

        sampled[i] = int(draw)
        remaining_to_draw -= int(draw)
        remaining_total -= c

        if remaining_to_draw <= 0:
            break

    if remaining_to_draw > 0:
        sampled[-1] = min(int(counts[-1]), remaining_to_draw)

    return sampled


def bootstrap_recovered_percent(counts_dict, depth, n_iter=50, rng=None):
    """
    Bootstrap subsampling WITHOUT replacement at a given depth.
    Returns:
      mean_percent, sd_percent, ci_low, ci_high, mean_recovered_taxa
    Notes:
      - If total reads N < depth, we do NOT upsample; we treat it as full data => 100%.
    """
    if rng is None:
        rng = np.random.default_rng()

    taxa = list(counts_dict.keys())
    counts = np.array([counts_dict[t] for t in taxa], dtype=int)

    N = int(counts.sum())
    if N <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    S_obs = int(np.sum(counts > 0))

    if N < depth:
        return 100.0, 0.0, 100.0, 100.0, float(S_obs)

    recovered = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sampled_counts = sample_counts_without_replacement_seq(counts, depth, rng)
        recovered[i] = float(np.sum(sampled_counts > 0))

    perc = (recovered / S_obs) * 100.0
    mean_p = float(np.mean(perc))
    sd_p = float(np.std(perc, ddof=1)) if n_iter > 1 else 0.0
    ci_low = float(np.percentile(perc, 2.5))
    ci_high = float(np.percentile(perc, 97.5))
    mean_S = float(np.mean(recovered))
    return mean_p, sd_p, ci_low, ci_high, mean_S


def compute_recovered_for_thresholds(table, sample_to_group, depths=(135000, 200000), n_iter=50, seed=123, group_order=None):
    rng = np.random.default_rng(seed)
    rows = []

    for idx, sample in enumerate(table.index, start=1):
        counts_series = table.loc[sample]
        counts_series = counts_series[counts_series > 0]
        counts_dict = counts_series.to_dict()

        total_reads = int(counts_series.sum())
        S_obs = int(len(counts_series))

        row = {
            "group": sample_to_group.get(sample, "unknown_group"),
            "sample": sample,
            "total_reads": total_reads,
            "S_obs": S_obs,
        }

        for d in depths:
            mean_p, sd_p, ci_low, ci_high, mean_S = bootstrap_recovered_percent(
                counts_dict, depth=int(d), n_iter=n_iter, rng=rng
            )
            row[f"percent_recovered_mean_{d}"] = mean_p
            row[f"percent_recovered_sd_{d}"] = sd_p
            row[f"percent_recovered_ci_low_{d}"] = ci_low
            row[f"percent_recovered_ci_high_{d}"] = ci_high
            row[f"recovered_taxa_mean_{d}"] = mean_S

        rows.append(row)

        # lightweight progress
        if idx % 10 == 0 or idx == len(table.index):
            print(f"  ...bootstrap progress: {idx}/{len(table.index)} samples")

    df = pd.DataFrame(rows)

    if group_order is not None:
        df["__group_order"] = df["group"].apply(lambda g: group_order.index(g) if g in group_order else 999)
        df = df.sort_values(["__group_order", "sample"], ascending=[True, True]).drop(columns=["__group_order"])
    else:
        df = df.sort_values(["group", "sample"], ascending=[True, True])

    return df


# ================================================================
# PLOTTING
# ================================================================
def plot_rarefaction(table, output_png, sample_to_group, group_order, palette,
                     num_points=800, vlines=(135000, 200000), x_max=1_000_000):
    plt.figure(figsize=(12, 7))

    # style: make ALL curves identical (same alpha/linewidth), like you requested
    alpha_all = 0.6
    lw_all = 2.2

    for sample in table.index:
        abund = table.loc[sample].values
        if int(np.sum(abund)) <= 0:
            continue

        x, y = rarefaction_curve_visual(abund, num_points=num_points)
        group = sample_to_group.get(sample, "unknown_group")
        color = palette.get(group, "#333333")
        plt.plot(x, y, color=color, alpha=alpha_all, linewidth=lw_all)

    # vertical lines (subsampling depths)
    for d in vlines:
        plt.axvline(d, linestyle="--", linewidth=2.6, color="black", alpha=0.9, zorder=10)

    plt.xlabel("Number of Reads")
    plt.ylabel("Expected Number of Unique Species/Taxa")
    plt.title("Rarefaction Curves")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.xlim(0, x_max)
    plt.ticklabel_format(style="plain", axis="x")

    # legend: ordered by group_order
    groups_present = sorted(set(sample_to_group.values()), key=lambda g: group_order.index(g) if g in group_order else 999)
    handles = [Line2D([0], [0], color=palette[g], lw=3.0, label=g) for g in groups_present if g in palette]
    for d in vlines:
        handles.append(Line2D([0], [0], color="black", lw=2.6, linestyle="--", label=f"Subsampling ({d:,} reads)"))

    # de-duplicate legend labels
    lab2h = {h.get_label(): h for h in handles}
    plt.legend(handles=list(lab2h.values()), title="Groups", loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(output_png)
    print(f"✅ Rarefaction plot saved: {output_png}")


def plot_results_table(df, output_png, depths=(135000, 200000), max_rows=80, base_font=16):
    df_show = df.copy()

    for d in depths:
        df_show[f"percent_recovered_mean_{d}"] = df_show[f"percent_recovered_mean_{d}"].round(2)
        df_show[f"percent_recovered_sd_{d}"] = df_show[f"percent_recovered_sd_{d}"].round(2)
        df_show[f"percent_recovered_ci_low_{d}"] = df_show[f"percent_recovered_ci_low_{d}"].round(2)
        df_show[f"percent_recovered_ci_high_{d}"] = df_show[f"percent_recovered_ci_high_{d}"].round(2)

    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)

    cols = ["group", "sample", "total_reads", "S_obs"]
    for d in depths:
        cols += [
            f"percent_recovered_mean_{d}",
            f"percent_recovered_sd_{d}",
            f"percent_recovered_ci_low_{d}",
            f"percent_recovered_ci_high_{d}",
        ]
    df_show = df_show[cols]

    fig_h = max(4.0, 0.42 * (len(df_show) + 1))
    fig, ax = plt.subplots(figsize=(22, fig_h))
    ax.axis("off")

    title = "Percent observed taxa recovered (bootstrap subsampling WITHOUT replacement)"
    ax.set_title(title, fontsize=base_font + 2, pad=14)

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="left",
        colLoc="left"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(base_font - 2)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_linewidth(1.2)
        else:
            cell.set_linewidth(0.4)

    plt.tight_layout()
    plt.savefig(output_png)
    print(f"✅ Table-figure saved: {output_png}")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Recursive analysis of *_taxonomy.txt files: rarefaction plot + bootstrap % taxa recovered at given depths."
    )
    parser.add_argument("input_dir", help="Root directory to search recursively for taxonomy files")
    parser.add_argument("--suffix", default="_taxonomy.txt", help="Input file suffix (default: _taxonomy.txt)")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: input_dir)")
    parser.add_argument("--out_prefix", default="rarefaction", help="Output prefix (default: rarefaction)")
    parser.add_argument("--num_points", type=int, default=800, help="Number of points for curves (default 800)")
    parser.add_argument("--depths", default="135000,200000", help="Comma-separated depths (default 135000,200000)")
    parser.add_argument("--n_iter", type=int, default=50, help="Bootstrap iterations per sample (default 50)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (default 123)")
    parser.add_argument("--base_font", type=int, default=16, help="Base font for figures (default 16)")
    parser.add_argument("--x_max", type=int, default=1_000_000, help="X-axis max for rarefaction plot (default 1,000,000)")
    parser.add_argument("--table_rows", type=int, default=80, help="Max rows shown in table-figure (default 80)")
    args = parser.parse_args()

    set_paper_style(base_font=args.base_font)

    root = os.path.abspath(args.input_dir)
    if not os.path.isdir(root):
        raise SystemExit(f"Error: '{root}' is not a directory.")

    out_dir = os.path.abspath(args.output_dir) if args.output_dir else root
    os.makedirs(out_dir, exist_ok=True)

    depths = tuple(int(x.strip()) for x in args.depths.split(",") if x.strip())
    if not depths:
        raise SystemExit("Error: --depths must contain at least one integer (e.g. 135000,200000).")

    files = find_taxonomy_files(root, args.suffix, recursive=True)
    if not files:
        raise SystemExit(f"No files found with suffix '{args.suffix}' under: {root}")

    print(f"Found taxonomy files: {len(files)}")

    table, sample_to_group, groups = build_table(files, root, args.suffix)

    # Generic ordering + palette:
    group_order = build_group_order(groups)
    palette = build_group_palette(groups)

    # Output paths (absolute)
    out_table_csv = os.path.join(out_dir, f"{args.out_prefix}_table.csv")
    out_raref_png = os.path.join(out_dir, f"{args.out_prefix}_curves.png")
    out_boot_csv  = os.path.join(out_dir, f"{args.out_prefix}_percent_recovered_bootstrap.csv")
    out_tab_png   = os.path.join(out_dir, f"{args.out_prefix}_percent_recovered_table.png")

    # Save sample×taxa table
    table.to_csv(out_table_csv)
    print(f"✅ Sample×taxa table saved: {out_table_csv}")

    # Rarefaction plot
    plot_rarefaction(
        table, out_raref_png, sample_to_group,
        group_order=group_order,
        palette=palette,
        num_points=args.num_points,
        vlines=depths,
        x_max=args.x_max
    )

    # Bootstrap recovery
    df_boot = compute_recovered_for_thresholds(
        table, sample_to_group,
        depths=depths,
        n_iter=args.n_iter,
        seed=args.seed,
        group_order=group_order
    )
    df_boot.to_csv(out_boot_csv, index=False)
    print(f"✅ Bootstrap results saved: {out_boot_csv}")

    # Table-figure
    plot_results_table(
        df_boot, out_tab_png,
        depths=depths,
        max_rows=args.table_rows,
        base_font=args.base_font
    )

    print("\nPreview (first 10 rows):")
    print(df_boot.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
