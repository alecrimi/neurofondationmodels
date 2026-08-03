
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
# Path to the CSV file downloaded from Kaggle
INPUT_CSV = r"neurofondationmodels\Michal\results\causality_results\granger_classical_offset_results.csv"
OUTPUT_DIR = r"neurofondationmodels\Michal\scripts\causality_testing\Results_processing"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TABLE1_CSV = os.path.join(OUTPUT_DIR, "Table1_Overall_Causality.csv")
TABLE2_CSV = os.path.join(OUTPUT_DIR, "Table2_Causality_by_Group.csv")
PLOT_LAG_PROFILE = os.path.join(OUTPUT_DIR, "plot_lag_profile.png")
PLOT_HEATMAP = os.path.join(OUTPUT_DIR, "plot_causality_heatmap.png")
PLOT_GROUP_HEATMAPS = os.path.join(OUTPUT_DIR, "plot_group_magnitude_heatmaps.png")
PLOT_GROUP_SIG_HEATMAPS = os.path.join(OUTPUT_DIR, "plot_group_significance_heatmaps.png")

# Statistical significance threshold
ALPHA = 0.05

# Visual settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=0.9)

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print(f"Loading data from {INPUT_CSV}...")
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: Could not find '{INPUT_CSV}'. Please ensure it is in the same directory.")
    exit()

# Create a clear string for the directional pair
df['Pair'] = df['covariate_X'] + " -> " + df['target_Y']

# Determine significance (1 if p < 0.05, 0 otherwise)
df['is_significant'] = (df['p_value'] < ALPHA).astype(int)

# ==========================================
# 2. CALCULATE LAG STATISTICS (For Table 1 & Plots)
# ==========================================
print("Calculating statistics across lags...")
lag_stats = df.groupby(['Pair', 'lag_ms']).agg(
    Median_F=('f_stat', 'median'),
    Sig_Pct=('is_significant', lambda x: x.mean() * 100)
).reset_index()

# Find the OPTIMAL LAG for each pair
idx_optimal = lag_stats.groupby('Pair')['Sig_Pct'].idxmax()
optimal_lags_df = lag_stats.loc[idx_optimal].copy()
optimal_lags_df.rename(columns={'lag_ms': 'Optimal_Lag_ms'}, inplace=True)

# Format Table 1
table1 = optimal_lags_df.sort_values(by='Sig_Pct', ascending=False).round(2)
table1.to_csv(TABLE1_CSV, index=False)

# ==========================================
# 3. CALCULATE GROUP STATISTICS (Table 2)
# ==========================================
print("Calculating group statistics at optimal lags...")
df_optimal_only = pd.merge(df, optimal_lags_df[['Pair', 'Optimal_Lag_ms']], 
                           left_on=['Pair', 'lag_ms'], 
                           right_on=['Pair', 'Optimal_Lag_ms'])

group_stats = df_optimal_only.groupby(['Pair', 'group']).agg(
    Sig_Pct=('is_significant', lambda x: x.mean() * 100)
).reset_index()

# Pivot the table so Groups are columns
table2 = group_stats.pivot(index='Pair', columns='group', values='Sig_Pct').reset_index()
table2.columns.name = None 
table2 = table2.round(2)
table2.to_csv(TABLE2_CSV, index=False)

# ==========================================
# 4. GENERATE PLOT A: LAG PROFILE
# ==========================================
print("Generating Lag Profile Plot...")
plt.figure(figsize=(12, 6))
sns.lineplot(data=lag_stats, x='lag_ms', y='Sig_Pct', hue='Pair', marker='o', linewidth=2.5, markersize=8)

plt.title("Lag Profile: Temporal Evolution of Granger Causality", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Look-back Lag (ms)", fontsize=14)
plt.ylabel("Significant Windows (%)", fontsize=14)
plt.legend(title="Electrode Direction", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(-5, 105)
plt.tight_layout()
plt.savefig(PLOT_LAG_PROFILE, dpi=300)
plt.close()

# ==========================================
# 5. GENERATE PLOT B: OVERALL HEATMAP
# ==========================================
print("Generating Overall Causality Heatmap...")
heatmap_data = lag_stats.pivot(index='Pair', columns='lag_ms', values='Median_F')

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt=".2f", cbar_kws={'label': 'Median F-Statistic'})
plt.title("Overall Causality Strength Heatmap", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Lag (ms)", fontsize=14)
plt.ylabel("Electrode Direction", fontsize=14)
plt.tight_layout()
plt.savefig(PLOT_HEATMAP, dpi=300)
plt.close()

# ==========================================
# 6. GENERATE GROUP-SPECIFIC HEATMAPS (F-STAT & SIG_PCT)
# ==========================================
print("Generating Group-Specific Heatmaps...")
valid_groups = sorted([g for g in df['group'].unique() if pd.notna(g) and g != "Unknown"])
n_groups = len(valid_groups)

if n_groups > 0:
    # Ensure BOTH Median_F and Sig_Pct are calculated per group
    group_lag_stats = df.groupby(['group', 'Pair', 'lag_ms']).agg(
        Median_F=('f_stat', 'median'),
        Sig_Pct=('is_significant', lambda x: x.mean() * 100)
    ).reset_index()

    # --- 6A: F-STAT HEATMAP ---
    fig_f, axes_f = plt.subplots(1, n_groups, figsize=(5.5 * n_groups, 6), sharey=True)
    if n_groups == 1: axes_f = [axes_f]

    vmin_f = group_lag_stats['Median_F'].min()
    vmax_f = group_lag_stats['Median_F'].max()

    for idx, grp in enumerate(valid_groups):
        grp_data = group_lag_stats[group_lag_stats['group'] == grp]
        heatmap_data_f = grp_data.pivot(index='Pair', columns='lag_ms', values='Median_F')
        show_cbar = (idx == n_groups - 1)
        
        sns.heatmap(heatmap_data_f, annot=True, cmap='YlOrRd', fmt=".2f", 
                    vmin=vmin_f, vmax=vmax_f,
                    cbar=show_cbar, cbar_kws={'label': 'Median F-Stat'} if show_cbar else None,
                    ax=axes_f[idx])
        
        axes_f[idx].set_title(f"Group: {grp}", fontsize=14, fontweight='bold')
        axes_f[idx].set_xlabel("Lag (ms)", fontsize=12)
        axes_f[idx].set_ylabel("Electrode Direction" if idx == 0 else "", fontsize=12)

    plt.suptitle("Causality Strength by Patient Group (Median F-Stat)", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_GROUP_HEATMAPS, dpi=300, bbox_inches='tight')
    plt.close(fig_f)

    # --- 6B: SIG_PCT HEATMAP ---
    fig_s, axes_s = plt.subplots(1, n_groups, figsize=(5.5 * n_groups, 6), sharey=True)
    if n_groups == 1: axes_s = [axes_s]

    for idx, grp in enumerate(valid_groups):
        grp_data = group_lag_stats[group_lag_stats['group'] == grp]
        heatmap_data_s = grp_data.pivot(index='Pair', columns='lag_ms', values='Sig_Pct')
        show_cbar = (idx == n_groups - 1)
        
        # We use 'Blues' to differentiate from F-Stat, and lock vmin=0, vmax=100 for percentages
        sns.heatmap(heatmap_data_s, annot=True, cmap='Blues', fmt=".1f", 
                    vmin=0, vmax=100,
                    cbar=show_cbar, cbar_kws={'label': 'Significant Windows (%)'} if show_cbar else None,
                    ax=axes_s[idx])
        
        axes_s[idx].set_title(f"Group: {grp}", fontsize=14, fontweight='bold')
        axes_s[idx].set_xlabel("Lag (ms)", fontsize=12)
        axes_s[idx].set_ylabel("Electrode Direction" if idx == 0 else "", fontsize=12)

    plt.suptitle("Consistency by Patient Group (% Significant Windows)", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_GROUP_SIG_HEATMAPS, dpi=300, bbox_inches='tight')
    plt.close(fig_s)