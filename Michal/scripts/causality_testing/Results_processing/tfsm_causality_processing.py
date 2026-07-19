import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore empty plot warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_DIR = r"neurofondationmodels\Michal\results\causality_results" 
OUTPUT_DIR = r"neurofondationmodels\Michal\scripts\causality_testing\Results_processing"
ALPHA = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visual settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=0.9)

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print(f"Searching for *causality_results.csv files in: {os.getcwd()}...")
search_path = os.path.join(RESULTS_DIR, "**", "*causality_results.csv")
csv_files = glob.glob(search_path, recursive=True)

if not csv_files:
    print("Error: No *causality_results.csv files found.")
    exit()

df_list = []
for file in csv_files:
    if "granger_classical" in file: 
        continue 
        
    base_name = os.path.basename(file)
    model_name = base_name.split('_')[0].capitalize()
        
    temp_df = pd.read_csv(file)
    temp_df['model_name'] = model_name
    df_list.append(temp_df)

df_all = pd.concat(df_list, ignore_index=True)
df_all['group'] = df_all['group'].fillna('Unknown')
df_all['Pair'] = df_all['covariate_X'] + " -> " + df_all['target_Y']
df_all['is_significant'] = (df_all['p_value_residual'] < ALPHA).astype(int)

models_found = list(df_all['model_name'].unique())
print(f"Found models: {models_found}. Generating separate reports...")

# ==========================================
# 2. MAIN LOOP - REPORT FOR EACH MODEL
# ==========================================
for model in models_found:
    print(f"\n[{model}] Processing started...")
    df = df_all[df_all['model_name'] == model].copy()
    
    # Prefix for output files of the current model
    PREFIX = f"{model}_"
        
    PLOT_LAG_PROFILE = os.path.join(OUTPUT_DIR, f"{PREFIX}plot_lag_profile.png")
    PLOT_HEATMAP = os.path.join(OUTPUT_DIR, f"{PREFIX}plot_causality_heatmap.png")
    PLOT_GROUP_HEATMAPS = os.path.join(OUTPUT_DIR, f"{PREFIX}plot_group_magnitude_heatmaps.png")
    PLOT_GROUP_SIG_HEATMAPS = os.path.join(OUTPUT_DIR, f"{PREFIX}plot_group_significance_heatmaps.png")

    # --- CALCULATE STATISTICS ---
    lag_stats = df.groupby(['Pair', 'lag_ms']).agg(
        Median_F=('f_stat_residual', 'median'),
        Sig_Pct=('is_significant', lambda x: x.mean() * 100)
    ).reset_index()

    idx_optimal = lag_stats.groupby('Pair')['Sig_Pct'].idxmax()
    optimal_lags_df = lag_stats.loc[idx_optimal].copy()
    optimal_lags_df.rename(columns={'lag_ms': 'Optimal_Lag_ms'}, inplace=True)

    table1 = optimal_lags_df.sort_values(by='Sig_Pct', ascending=False).round(2)
    table1.to_csv(os.path.join(OUTPUT_DIR, f"{PREFIX}Table1_Overall.csv"), index=False)

    df_optimal_only = pd.merge(df, optimal_lags_df[['Pair', 'Optimal_Lag_ms']], 
                               left_on=['Pair', 'lag_ms'], 
                               right_on=['Pair', 'Optimal_Lag_ms'])

    group_stats = df_optimal_only.groupby(['Pair', 'group']).agg(
        Sig_Pct=('is_significant', lambda x: x.mean() * 100)
    ).reset_index()

    table2 = group_stats.pivot(index='Pair', columns='group', values='Sig_Pct').reset_index()
    table2.columns.name = None 
    table2 = table2.round(2)
    table2.to_csv(os.path.join(OUTPUT_DIR, f"{PREFIX}Table2_Group.csv"), index=False)

    # --- PLOT A: LAG PROFILE ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=lag_stats, x='lag_ms', y='Sig_Pct', hue='Pair', marker='o', linewidth=2.5, markersize=8)
    plt.title(f"[{model}] Lag Profile: Temporal Evolution of Causality", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Look-back Lag (ms)", fontsize=14)
    plt.ylabel("Significant Windows (%)", fontsize=14)
    plt.legend(title="Electrode Direction", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig(PLOT_LAG_PROFILE, dpi=300)
    plt.close()

    # --- PLOT B: OVERALL HEATMAP ---
    heatmap_data = lag_stats.pivot(index='Pair', columns='lag_ms', values='Median_F')
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt=".2f", cbar_kws={'label': 'Median F-Statistic'})
    plt.title(f"[{model}] Overall Causality Strength Heatmap", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Lag (ms)", fontsize=14)
    plt.ylabel("Electrode Direction", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOT_HEATMAP, dpi=300)
    plt.close()

    # --- PLOTS C & D: GROUP HEATMAPS ---
    valid_groups = sorted([g for g in df['group'].unique() if pd.notna(g) and g != "Unknown"])
    n_groups = len(valid_groups)

    if n_groups > 0:
        group_lag_stats = df.groupby(['group', 'Pair', 'lag_ms']).agg(
            Median_F=('f_stat_residual', 'median'),
            Sig_Pct=('is_significant', lambda x: x.mean() * 100)
        ).reset_index()

        # F-STAT HEATMAP
        fig_f, axes_f = plt.subplots(1, n_groups, figsize=(5.5 * n_groups, 6), sharey=True)
        if n_groups == 1: axes_f = [axes_f]

        vmin_f = group_lag_stats['Median_F'].min()
        vmax_f = group_lag_stats['Median_F'].max()

        for idx, grp in enumerate(valid_groups):
            grp_data = group_lag_stats[group_lag_stats['group'] == grp]
            h_data_f = grp_data.pivot(index='Pair', columns='lag_ms', values='Median_F')
            show_cbar = (idx == n_groups - 1)
            
            sns.heatmap(h_data_f, annot=True, cmap='YlOrRd', fmt=".2f", 
                        vmin=vmin_f, vmax=vmax_f,
                        cbar=show_cbar, cbar_kws={'label': 'Median F-Stat'} if show_cbar else None,
                        ax=axes_f[idx])
            
            axes_f[idx].set_title(f"Group: {grp}", fontsize=14, fontweight='bold')
            axes_f[idx].set_xlabel("Lag (ms)", fontsize=12)
            axes_f[idx].set_ylabel("Electrode Direction" if idx == 0 else "", fontsize=12)

        plt.suptitle(f"[{model}] Causality Strength by Group (Median F-Stat)", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(PLOT_GROUP_HEATMAPS, dpi=300, bbox_inches='tight')
        plt.close(fig_f)

        # SIG_PCT HEATMAP
        fig_s, axes_s = plt.subplots(1, n_groups, figsize=(5.5 * n_groups, 6), sharey=True)
        if n_groups == 1: axes_s = [axes_s]

        for idx, grp in enumerate(valid_groups):
            grp_data = group_lag_stats[group_lag_stats['group'] == grp]
            h_data_s = grp_data.pivot(index='Pair', columns='lag_ms', values='Sig_Pct')
            show_cbar = (idx == n_groups - 1)
            
            sns.heatmap(h_data_s, annot=True, cmap='Blues', fmt=".1f", 
                        vmin=0, vmax=100,
                        cbar=show_cbar, cbar_kws={'label': 'Significant Windows (%)'} if show_cbar else None,
                        ax=axes_s[idx])
            
            axes_s[idx].set_title(f"Group: {grp}", fontsize=14, fontweight='bold')
            axes_s[idx].set_xlabel("Lag (ms)", fontsize=12)
            axes_s[idx].set_ylabel("Electrode Direction" if idx == 0 else "", fontsize=12)

        plt.suptitle(f"[{model}] Consistency by Group (% Significant Windows)", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(PLOT_GROUP_SIG_HEATMAPS, dpi=300, bbox_inches='tight')
        plt.close(fig_s)