#!/usr/bin/env python3
import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

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


BASE_COLORS = {
    "control": "#0C7BDC",
    "ethanol": "#E140BB",
    "bleach": "#FB6D08",
    "dna_rna_zap": "#039703",
    "lyzo_bleach": "#E8E80D",
    "lyzo_zap": "#8EB9FC",
}

SUFFIX_ORDER = ["control", "ethanol", "bleach", "dna_rna_zap", "lyzo_bleach", "lyzo_zap"]

def infer_prefixes_from_folders(folder_names):
    """
    Detect whether folders contain adults_*, nymphs_*, or both.
    Returns sorted unique prefixes.
    """
    prefixes = set()
    for f in folder_names:
        if "_" in f:
            prefixes.add(f.split("_", 1)[0].lower())
    # keep only plausible ones
    prefixes = [p for p in prefixes if p in ("adults", "nymphs")]
    return sorted(prefixes)

def build_group_order_and_palette(prefixes):
    """
    Build GROUP_ORDER and CUSTOM_PALETTE automatically for detected prefixes.
    """
    group_order = []
    palette = {}
    for pref in prefixes:
        for suf in SUFFIX_ORDER:
            g = f"{pref}_{suf}"
            group_order.append(g)
            palette[g] = BASE_COLORS[suf]
    return group_order, palette

def find_taxonomy_files(root_dir: str, suffix: str, recursive: bool = True):
    pattern = os.path.join(root_dir, "**", f"*{suffix}") if recursive else os.path.join(root_dir, f"*{suffix}")
    files = glob.glob(pattern, recursive=recursive)
    return sorted([f for f in files if os.path.isfile(f)])

def make_unique_sample_name(filepath: str, root_dir: str, suffix: str):
    rel = os.path.relpath(filepath, root_dir)
    base = rel.replace(os.sep, "__")
    if base.endswith(suffix):
        base = base[:-len(suffix)]
    return base

def parse_taxonomy_file(filepath: str):
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
    data = {}
    sample_to_group = {}
    folder_names = set()

    for fp in files:
        sample = make_unique_sample_name(fp, root_dir, suffix)
        folder = os.path.basename(os.path.dirname(fp))
        folder_names.add(folder)

        # group label = folder name (as requested)
        group = folder.lower()
        data[sample] = parse_taxonomy_file(fp)
        sample_to_group[sample] = group

    tabella = pd.DataFrame.from_dict(data, orient="index").fillna(0).astype(int)
    tabella.index.name = "sample"
    return tabella, sample_to_group, sorted(folder_names)

def group_sort_key_factory(group_order):
    def _key(g: str):
        if g in group_order:
            return (0, group_order.index(g))
        return (1, g)
    return _key

def taxonomy_to_genus(taxonomy):
    parts = [x.strip() for x in str(taxonomy).split(";") if x.strip()]
    if len(parts) < 2:
        return None
    return parts[-2] 

def expected_species_curve_multinomial(abundances, x):
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


def expand_to_reads(taxa, counts):
    return np.repeat(np.asarray(taxa, dtype=object), np.asarray(counts, dtype=int))

def bootstrap_recovered_percent(counts_dict, depth, n_iter=50, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    taxa = np.array(list(counts_dict.keys()), dtype=object)
    counts = np.array([counts_dict[t] for t in taxa], dtype=int)
    N = int(counts.sum())
    if N <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    S_obs = int(np.sum(counts > 0))

    genera_obs_set = {
        taxonomy_to_genus(t)
        for t in taxa[counts > 0]
        if taxonomy_to_genus(t) is not None
    }
    G_obs = int(len(genera_obs_set))

    if N < depth:
        return 100.0, 0.0, 100.0, 100.0, float(S_obs), 100.0, 0.0, 100.0, 100.0, float(G_obs)

    reads = expand_to_reads(taxa, counts)

    recovered_taxa = np.empty(n_iter, dtype=float)
    recovered_genera = np.empty(n_iter, dtype=float)

    for i in range(n_iter):
        idx = rng.choice(N, size=depth, replace=False)
        sampled = reads[idx]

        sampled_unique_taxa = np.unique(sampled)
        recovered_taxa[i] = float(len(sampled_unique_taxa))

        sampled_genera = {
            taxonomy_to_genus(t)
            for t in sampled_unique_taxa
            if taxonomy_to_genus(t) is not None
        }
        recovered_genera[i] = float(len(sampled_genera))

    perc_taxa = recovered_taxa / S_obs * 100.0
    mean_p = float(np.mean(perc_taxa))
    sd_p = float(np.std(perc_taxa, ddof=1)) if n_iter > 1 else 0.0
    ci_low = float(np.percentile(perc_taxa, 2.5))
    ci_high = float(np.percentile(perc_taxa, 97.5))
    mean_S = float(np.mean(recovered_taxa))

    perc_genera = recovered_genera / G_obs * 100.0 if G_obs > 0 else np.full(n_iter, np.nan)
    mean_pg = float(np.nanmean(perc_genera))
    sd_pg = float(np.nanstd(perc_genera, ddof=1)) if n_iter > 1 else 0.0
    ci_low_g = float(np.nanpercentile(perc_genera, 2.5))
    ci_high_g = float(np.nanpercentile(perc_genera, 97.5))
    mean_G = float(np.mean(recovered_genera))

    return mean_p, sd_p, ci_low, ci_high, mean_S, mean_pg, sd_pg, ci_low_g, ci_high_g, mean_G

def compute_recovered_for_thresholds(tabella, sample_to_group, depths=(135000, 200000), n_iter=50, seed=123, group_order=None):
    rng = np.random.default_rng(seed)
    rows = []

    for sample in tabella.index:
        counts_series = tabella.loc[sample]
        counts_series = counts_series[counts_series > 0]
        counts_dict = counts_series.to_dict()

        total_reads = int(counts_series.sum())
        S_obs = int(len(counts_series))

        genera_obs = {
            taxonomy_to_genus(t)
            for t in counts_series.index
            if taxonomy_to_genus(t) is not None
        }
        G_obs = int(len(genera_obs))

        row = {
            "group": sample_to_group.get(sample, "unknown_group"),
            "sample": sample,
            "total_reads": total_reads,
            "S_obs": S_obs,
            "G_obs": G_obs,
        }

        for d in depths:
            mean_p, sd_p, ci_low, ci_high, mean_S, mean_pg, sd_pg, ci_low_g, ci_high_g, mean_G = bootstrap_recovered_percent(
                counts_dict, depth=int(d), n_iter=n_iter, rng=rng
            )

            row[f"percent_recovered_mean_{d}"] = mean_p
            row[f"percent_recovered_sd_{d}"] = sd_p
            row[f"percent_recovered_ci_low_{d}"] = ci_low
            row[f"percent_recovered_ci_high_{d}"] = ci_high
            row[f"recovered_taxa_mean_{d}"] = mean_S

            row[f"percent_genera_recovered_mean_{d}"] = mean_pg
            row[f"percent_genera_recovered_sd_{d}"] = sd_pg
            row[f"percent_genera_recovered_ci_low_{d}"] = ci_low_g
            row[f"percent_genera_recovered_ci_high_{d}"] = ci_high_g
            row[f"recovered_genera_mean_{d}"] = mean_G

        rows.append(row)

    df = pd.DataFrame(rows)

    if group_order is not None:
        df["group_order"] = df["group"].apply(lambda g: group_order.index(g) if g in group_order else 999)
        df = df.sort_values(["group_order", "sample"], ascending=[True, True]).drop(columns=["group_order"])
    else:
        df = df.sort_values(["group", "sample"], ascending=[True, True])

    return df

def plot_rarefaction(tabella, output_png, sample_to_group, group_order, palette, num_points=800, vlines=(135000, 200000), x_max=1_000_000):
    plt.figure(figsize=(12, 7))

    groups = sorted(set(sample_to_group.values()), key=group_sort_key_factory(group_order))
    fallback_cmap = mpl.colormaps.get_cmap("tab10").resampled(max(1, len(groups)))

    group_to_color = {}
    for i, g in enumerate(groups):
        group_to_color[g] = palette.get(g, fallback_cmap(i))

    alpha_all = 0.6
    lw_all = 2.2

    for sample in tabella.index:
        abund = tabella.loc[sample].values
        if int(np.sum(abund)) <= 0:
            continue
        x, y = rarefaction_curve_visual(abund, num_points=num_points)
        group = sample_to_group.get(sample, "unknown_group")
        color = group_to_color.get(group, "#333333")
        plt.plot(x, y, color=color, alpha=alpha_all, linewidth=lw_all)

    for d in vlines:
        plt.axvline(d, linestyle="--", linewidth=2.6, color="black", alpha=0.9)

    plt.xlabel("Number of Reads")
    plt.ylabel("Expected Number of Unique Species/Taxa")
    plt.title("Rarefaction Curves")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.xlim(0, x_max)
    plt.ticklabel_format(style="plain", axis="x")

    ordered_groups = [g for g in group_order if g in groups] + [g for g in groups if g not in group_order]
    handles = [Line2D([0], [0], color=group_to_color[g], lw=3.0, label=g) for g in ordered_groups]
    for d in vlines:
        handles.append(Line2D([0], [0], color="black", lw=2.6, linestyle="--", label=f"Subsampling ({d:,} reads)"))

    lab2h = {h.get_label(): h for h in handles}
    plt.legend(handles=list(lab2h.values()), title="Groups", loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(output_png)
    

def plot_results_table(df, output_png, depths=(135000, 200000), max_rows=80, base_font=16):
    df_show = df.copy()
    for d in depths:
        df_show[f"percent_recovered_mean_{d}"] = df_show[f"percent_recovered_mean_{d}"].round(2)
        df_show[f"percent_recovered_sd_{d}"] = df_show[f"percent_recovered_sd_{d}"].round(2)

    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)

    cols = ["group", "sample", "total_reads", "S_obs"]
    for d in depths:
        cols += [f"percent_recovered_mean_{d}", f"percent_recovered_sd_{d}"]

    df_show = df_show[cols]

    fig_h = max(4.0, 0.42 * (len(df_show) + 1))
    fig, ax = plt.subplots(figsize=(20, fig_h))
    ax.axis("off")

    title = "Percent observed taxa recovered (mean ± SD across subsampling iterations)"
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
    


parser = argparse.ArgumentParser(
    description="From *_taxonomy.txt: plot rarefaction curves and compute % observed taxa recovered at 135k and 200k using subsampling WITHOUT replacement (bootstrap). Works for adults or nymphs automatically."
)
parser.add_argument("input_dir", help="Root directory where *_taxonomy.txt files are searched recursively")
parser.add_argument("--suffix", default="_taxonomy.txt", help="Taxonomy file suffix (default: _taxonomy.txt)")
parser.add_argument("--out_prefix", default="rarefaction", help="Output prefix (default: rarefaction)")
parser.add_argument("--num_points", type=int, default=800, help="Number of points for rarefaction curves (default 800)")
parser.add_argument("--depths", default="135000", help="Comma-separated subsampling depths (default 135000,200000)")
parser.add_argument("--n_iter", type=int, default=50, help="Bootstrap subsampling iterations (default 50)")
parser.add_argument("--seed", type=int, default=123, help="Random seed (default 123)")
parser.add_argument("--base_font", type=int, default=16, help="Base font size for figures (default 16)")
parser.add_argument("--x_max", type=int, default=1_000_000, help="Maximum X-axis value (default 1000000)")
parser.add_argument("--table_rows", type=int, default=80, help="Maximum number of rows displayed in the summary table (default 80)")
args = parser.parse_args()

set_paper_style(base_font=args.base_font)

root = args.input_dir
if not os.path.isdir(root):
    raise SystemExit(f"Error: '{root}' is not a directory.")

depths = tuple(int(x.strip()) for x in args.depths.split(",") if x.strip())
if len(depths) == 0:
    raise SystemExit("Error: --depths must contain at least one value (e.g. 135000,200000).")

files = find_taxonomy_files(root, args.suffix, recursive=True)
if not files:
    raise SystemExit(f"No files found with suffix '{args.suffix}' under {root}")

print(f"Taxonomy files found: {len(files)}")

tabella, sample_to_group, folder_names = build_table(files, root, args.suffix)

# Infer groups (adults vs nymphs) from folder names
prefixes = infer_prefixes_from_folders([n.lower() for n in folder_names])
if not prefixes:
    # fallback: build order from observed groups (no enforced order)
    prefixes = []

GROUP_ORDER, CUSTOM_PALETTE = build_group_order_and_palette(prefixes) if prefixes else ([], {})

# If order is empty (unknown naming), use observed groups
observed_groups = sorted(set(sample_to_group.values()))
if not GROUP_ORDER:
    GROUP_ORDER = observed_groups

# Output files saved in root
out_table_csv = os.path.join(root, f"{args.out_prefix}_table.csv")
out_raref_png = os.path.join(root, f"{args.out_prefix}_curves_custom_palette.png")
out_boot_csv  = os.path.join(root, f"{args.out_prefix}_percent_recovered_bootstrap.csv")
out_tab_png   = os.path.join(root, f"{args.out_prefix}_percent_recovered_table.png")

tabella.to_csv(out_table_csv)


plot_rarefaction(
    tabella, out_raref_png, sample_to_group,
    group_order=GROUP_ORDER,
    palette=CUSTOM_PALETTE,
    num_points=args.num_points,
    vlines=depths,
    x_max=args.x_max
)

df_boot = compute_recovered_for_thresholds(
    tabella, sample_to_group,
    depths=depths,
    n_iter=args.n_iter,
    seed=args.seed,
    group_order=GROUP_ORDER
)

df_boot.to_csv(out_boot_csv, index=False)


plot_results_table(
    df_boot, out_tab_png,
    depths=depths,
    max_rows=args.table_rows,
    base_font=args.base_font
)

