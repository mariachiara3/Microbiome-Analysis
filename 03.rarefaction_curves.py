#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None
    print("⚠️ seaborn not found. KDE plots will use matplotlib.")

# ----------------------------
# GENERIC PATHS
# ----------------------------
ADULT_FOLDERS = [
    "path/to/adults_control",
    "path/to/adults_ethanol",
    "path/to/adults_bleach",
    "path/to/adults_zap",
    "path/to/adults_lyzo_zap",
    "path/to/adults_lyzo_bleach",
]

NYMPH_FOLDERS = [
    "path/to/nymphs_control",
    "path/to/nymphs_ethanol",
    "path/to/nymphs_bleach",
    "path/to/nymphs_zap",
    "path/to/nymphs_lyzo_zap",
    "path/to/nymphs_lyzo_bleach",
]

GROUP_NAME_MAP = {
    "adults_control": "Adults Control",
    "adults_ethanol": "Adults Ethanol",
    "adults_bleach": "Adults Bleach",
    "adults_zap": "Adults Zap",
    "adults_lyzo_zap": "Adults Lyzo_Zap",
    "adults_lyzo_bleach": "Adults Lyzo_Bleach",
    "nymphs_control": "Nymphs Control",
    "nymphs_ethanol": "Nymphs Ethanol",
    "nymphs_bleach": "Nymphs Bleach",
    "nymphs_zap": "Nymphs Zap",
    "nymphs_lyzo_zap": "Nymphs Lyzo_Zap",
    "nymphs_lyzo_bleach": "Nymphs Lyzo_Bleach",
}

PALETTE_GROUPS = ["#1b9e77","#1f78b4","#e31a1c","#6a3d9a","#8c510a","#ff7f00"]

# ----------------------------
# TAXONOMY PARSING
# ----------------------------
def aggregation_key(taxonomy):
    parts = taxonomy.split(';')
    if len(parts) < 7:
        return None
    genus = parts[5].lower()
    species = parts[6].lower()
    if species in ['uncultured_bacterium','uncultured_organism']:
        return f"{genus}_{species}"
    return f"{genus}_{species}" if '_' not in species else species

def parse_taxonomy_file(taxonomy_file):
    tax_dict = {}
    with open(taxonomy_file) as f:
        for line in f:
            parts = line.strip().split(None,1)
            if len(parts)==2:
                tax_dict[parts[0]] = parts[1]
    return tax_dict

def aggregate_counts(input_folder, taxonomy_file):
    tax_dict = parse_taxonomy_file(taxonomy_file)
    data = {}
    counter = {}
    for folder in input_folder:
        if not os.path.isdir(folder):
            print(f"⚠️ Folder not found: {folder}")
            continue
        folder_name = os.path.basename(folder)
        counter[folder_name] = 1
        for file in sorted(os.listdir(folder)):
            if file.endswith("taxonomy.txt"):
                filepath = os.path.join(folder,file)
                parsed=[]
                with open(filepath,'r',encoding='utf-8',errors='ignore') as f:
                    for line in f:
                        if not line.strip() or line.startswith(("Specie","Tassonomia","taxonomy")):
                            continue
                        parts = line.strip().split('\t')
                        if len(parts)>=2:
                            tax, count_str = parts[0].strip(), re.sub(r'[^\d]','',parts[1])
                            try:
                                parsed.append({'tax':tax,'count':int(count_str)})
                            except ValueError:
                                continue
                df = pd.DataFrame(parsed)
                if not df.empty:
                    df_agg = df.groupby('tax',dropna=False)['count'].sum().reset_index().set_index('tax')
                    sample_name = f"{folder_name}_{counter[folder_name]}"
                    counter[folder_name]+=1
                    data[sample_name] = df_agg['count'].to_dict()
    return pd.DataFrame.from_dict(data,orient='index').fillna(0)

# ----------------------------
# RAREFACTION & CHAO1
# ----------------------------
def rarefaction_curve(abundances,num_points=1000,extend_factor=1.05):
    abundances=np.array(abundances)
    total_reads=np.sum(abundances)
    if total_reads==0:
        return np.array([0]),np.array([0])
    p=abundances/total_reads
    max_reads=int(total_reads*extend_factor)
    x=np.linspace(1,max_reads,num=num_points)
    y=np.sum(1-(1-p[:,None])**x,axis=0)
    return x,y

def chao1(abundances):
    abundances=np.array(abundances)
    S_obs=np.sum(abundances>0)
    F1=np.sum(abundances==1)
    F2=np.sum(abundances==2)
    return S_obs + (F1*(F1-1)/2 if F2==0 else (F1**2)/(2*F2))

def reads_for_fraction_chao1(tabella,fraction=0.95,num_points=1000,extend_factor=1.05):
    results=[]
    for sample in tabella.index:
        abundances = tabella.loc[sample].values
        abundances = abundances[abundances>0]
        if abundances.size==0:
            results.append((sample,np.nan,np.nan,np.nan))
            continue
        total_species=chao1(abundances)
        target=total_species*fraction
        x,y=rarefaction_curve(abundances,num_points=num_points,extend_factor=extend_factor)
        mask=y>=target
        reads_needed=x[np.argmax(mask)] if np.any(mask) else x[-1]
        results.append((sample,total_species,target,reads_needed))
    return pd.DataFrame(results,columns=['sample','chao1_total_species','target_species','reads_needed'])

# ----------------------------
# THRESHOLD ANALYSIS
# ----------------------------
def rarefaction_thresholds(tabella,p=0.9,num_points=2000):
    results=[]
    for sample in tabella.index:
        abundances = tabella.loc[sample].values
        abundances = abundances[abundances>0]
        if abundances.size==0:
            results.append((sample,np.nan,np.nan,np.nan,0.0,True))
            continue
        total_reads=np.sum(abundances)
        x,y=rarefaction_curve(abundances,num_points=num_points)
        a_obs=y.max()
        xp=x[np.argmax(y>=p*a_obs)] if np.any(y>=p*a_obs) else x[-1]
        results.append((sample,a_obs,np.nan,xp,y.max(),True))
    return pd.DataFrame(results,columns=['sample','a','b','x_p','observed_max_species','fallback'])

def suggest_subsampling(df_xp):
    xp=df_xp['x_p'].dropna().values
    xp_sorted=np.sort(xp)
    suggestions={'min':float(xp_sorted[0]) if xp_sorted.size>0 else 0.0,
                 '25pct':float(np.percentile(xp_sorted,25)) if xp_sorted.size>0 else 0.0,
                 'median':float(np.median(xp_sorted)) if xp_sorted.size>0 else 0.0,
                 '75pct':float(np.percentile(xp_sorted,75)) if xp_sorted.size>0 else 0.0,
                 'max':float(xp_sorted[-1]) if xp_sorted.size>0 else 0.0}
    for r in [1.0,0.9,0.8,0.75,0.5]:
        suggestions[f'include_{int(r*100)}pct']=float(np.percentile(xp_sorted,r*100) if xp_sorted.size>0 else 0.0)
    return suggestions

# ----------------------------
# PLOTTING
# ----------------------------
def plot_box_reads(df_reads,tabella,title,output_file):
    req=df_reads.set_index('sample')['reads_needed']
    obs=tabella.sum(axis=1)
    df_plot=pd.DataFrame({'Observed':obs,'Required (95% Chao1)':req}).melt(var_name='Type',value_name='Reads')
    plt.figure(figsize=(8,6))
    if sns:
        sns.boxplot(x='Type',y='Reads',data=df_plot,palette=['skyblue','salmon'])
    else:
        plt.boxplot([df_plot[df_plot['Type']=='Observed']['Reads'],
                     df_plot[df_plot['Type']=='Required (95% Chao1)']['Reads']],
                     labels=['Observed','Required'])
    plt.yscale('log')
    plt.ylabel('Number of Reads (log scale)')
    plt.title(title)
    plt.grid(True,linestyle='--',alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file,dpi=300)
    plt.show()

def plot_bar_reads(df_reads,tabella,title,output_file):
    req=df_reads.set_index('sample')['reads_needed']
    obs=tabella.sum(axis=1)
    samples=obs.index
    x=range(len(samples))
    plt.figure(figsize=(max(12,len(samples)*0.5),6))
    plt.bar([i-0.2 for i in x],obs.loc[samples],width=0.4,label='Observed',color='skyblue')
    plt.bar([i+0.2 for i in x],req.loc[samples],width=0.4,label='Required (95% Chao1)',color='salmon')
    plt.xticks(x,samples,rotation=90)
    plt.ylabel('Number of Reads')
    plt.title(title)
    plt.legend()
    plt.grid(True,axis='y',linestyle='--',alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file,dpi=300)
    plt.show()

def plot_rarefaction_subplot(tab_adults,tab_nymphs,group_map,output_file,subsample_depth=200000):
    fig, axes = plt.subplots(1,2,figsize=(14,6),sharey=True)
    datasets={"Adults":(tab_adults,axes[0]),"Nymphs":(tab_nymphs,axes[1])}
    for title,(tab,ax) in datasets.items():
        sample_to_group={s:'_'.join(s.split('_')[:-1]) for s in tab.index}
        groups=sorted(set(sample_to_group.values()))
        for i,group in enumerate(groups):
            color=PALETTE_GROUPS[i%len(PALETTE_GROUPS)]
            for sample in tab.index:
                if sample_to_group[sample]!=group: continue
                abundances=tab.loc[sample].values
                abundances=abundances[abundances>0]
                if abundances.size==0: continue
                x,y=rarefaction_curve(abundances,num_points=800)
                ax.plot(x,y,color=color,alpha=0.8,linewidth=1.6,label=group_map.get(group,group))
        ax.set_title(title)
        ax.set_xlabel("Sequencing depth (reads)")
        ax.axvline(subsample_depth,linestyle="--",linewidth=1.5,color="black",alpha=0.8,label=f"Subsampling depth ({subsample_depth:,} reads)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Expected number of taxa")
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label=dict(zip(labels,handles))
        ax.legend(by_label.values(),by_label.keys(),title="Groups",fontsize="small")
    plt.tight_layout()
    plt.savefig(output_file,dpi=300)
    plt.show()

# ----------------------------
# MAIN EXECUTION
# ----------------------------
TAX_FILE = "path/to/taxonomy_file.txt"

# Aggregate counts
tab_adults = aggregate_counts(ADULT_FOLDERS,TAX_FILE)
tab_nymphs = aggregate_counts(NYMPH_FOLDERS,TAX_FILE)

# Rarefaction + threshold
df_reads_adults=reads_for_fraction_chao1(tab_adults, fraction=0.95)
df_reads_nymphs=reads_for_fraction_chao1(tab_nymphs, fraction=0.95)

plot_box_reads(df_reads_adults,tab_adults,"Adults: Observed vs Required Reads (95% Chao1)","path/to/output/boxplot_reads_adults.png")
plot_bar_reads(df_reads_adults,tab_adults,"Adults: Reads per Sample (95% Chao1)","path/to/output/barplot_reads_adults.png")
plot_box_reads(df_reads_nymphs,tab_nymphs,"Nymphs: Observed vs Required Reads (95% Chao1)","path/to/output/boxplot_reads_nymphs.png")
plot_bar_reads(df_reads_nymphs,tab_nymphs,"Nymphs: Reads per Sample (95% Chao1)","path/to/output/barplot_reads_nymphs.png")

plot_rarefaction_subplot(tab_adults,tab_nymphs,GROUP_NAME_MAP,"path/to/output/rarefaction_subplot.png")

