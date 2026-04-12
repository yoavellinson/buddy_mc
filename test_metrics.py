from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment as nisqa
import soundfile as sf
import re
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import io

def extract_audio_params(filename):
    """
    Robustly parses VCTK sample ID and DSP parameters.
    Handles multi-word loss functions like 'stft_comp'.
    """
    sample_match = re.match(r"(p\d+_\d+)", filename)
    sample_id = sample_match.group(1) if sample_match else "unknown"

    patterns = {
        "alpha": r"alpha_([\d.]+)",
        "zeta": r"zeta_([\d.]+)",
        "warmup_steps": r"warmup_steps_(\d+)",
        "beta_min": r"beta_min_([\d.]+)",
        "rec_loss": r"rec_loss_(.+)$", 
    }
    
    result = {"sample_id": sample_id}
    
    for key, pattern in patterns.items():
        # Using findall in case a parameter (like alpha) appears twice
        matches = re.findall(pattern, filename)
        if matches:
            # Take the last occurrence found in the filename
            val = matches[-1]
            
            if key == "warmup_steps":
                result[key] = int(val)
            elif key == "rec_loss":
                # Try to keep as float if numeric, else string (stft_comp)
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
            else:
                result[key] = float(val)
                
    return result



def nisqa_single(p):
    s,fs = sf.read(p)
    batch = torch.tensor(s.T)

    score = nisqa(batch, fs)
    score_L = score[0][0].item()
    score_R = score[1][0].item()
    score_ovrl = (score_L + score_R)*0.5
    diff = score_L - score_R
    return {"nisqa_left":round(score_L,3),"nisqa_right":round(score_R,3),"nisqa_ovrl":round(score_ovrl,3),'nisqa_diff':diff}


dir = Path('/home/workspace/yoavellinson/buddy_mc/experiments/buddy_grid_search/test05_04_2026/binaural_dereverberation/VCTK_16k_binaural_final_params/reconstructed')

files = list(dir.glob('*.wav'))
total_files = len(files)
results = []
for file in tqdm(files, total=total_files, desc="NISQA Scoring"):
    p = extract_audio_params(file.stem)
    nisqa_score = nisqa_single(file) 
    combined = {**p, **nisqa_score}
    results.append(combined)

df_raw = pd.DataFrame(results)
hp_columns = ['zeta', 'alpha', 'warmup_steps', 'beta_min', 'rec_loss']
metrics = ['nisqa_left', 'nisqa_right', 'nisqa_ovrl','nisqa_diff']
df_stats = df_raw.groupby(hp_columns)[metrics].agg(['mean', 'std']).reset_index()
df_stats.columns = [
    f"{col[0]}_{col[1]}" if col[1] else col[0] 
    for col in df_stats.columns.values
]

# Save both raw data and the summary
df_raw.to_csv(dir.parent / 'nisqa_raw_samples.csv', index=False)
df_stats.to_csv(dir.parent / 'nisqa_summary_stats.csv', index=False)

print("Processing complete. Summary saved to nisqa_summary_stats.csv")


# --- 1. Top 10 Configurations ---
top_configs = df_stats.sort_values(
    by=['nisqa_ovrl_mean', 'nisqa_ovrl_std'], 
    ascending=[False, True]
).head(10)

print("\n--- Top 10 Configurations (Overall Quality) ---")
print(top_configs[['zeta', 'alpha', 'rec_loss', 'nisqa_ovrl_mean', 'nisqa_ovrl_std']])

# --- 2. Binaural Balance Analysis ---
# A Scatter plot to see how symmetrical the reconstruction is.
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_raw, x='nisqa_left', y='nisqa_right', hue='rec_loss', alpha=0.5)

# Diagonal line represents perfect L/R balance
max_val = max(df_raw['nisqa_left'].max(), df_raw['nisqa_right'].max())
min_val = min(df_raw['nisqa_left'].min(), df_raw['nisqa_right'].min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Balance')

plt.title("Binaural Symmetry: Left vs Right NISQA")
plt.xlabel("NISQA Left")
plt.ylabel("NISQA Right")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(dir.parent / 'eda_binaural_symmetry.png')

# --- 3. Heatmap: HP Interaction on Quality ---
# Looking at how Alpha and Zeta interact for your primary loss function
# (Feel free to filter df_stats by a specific rec_loss if you have many)
plt.figure(figsize=(10, 6))
pivot_quality = df_stats.pivot_table(
    index='alpha', 
    columns='zeta', 
    values='nisqa_ovrl_mean'
)
sns.heatmap(pivot_quality, annot=True, cmap='YlGnBu', fmt=".3f")
plt.title("Mean NISQA Overall (Alpha vs Zeta)")
plt.savefig(dir.parent / 'eda_quality_heatmap.png')

# --- 4. Parallel Coordinates (The "Golden Path") ---
plt.figure(figsize=(12, 6))
plot_df = df_stats.copy()
# Convert rec_loss to numeric codes for plotting
plot_df['rec_loss_idx'] = plot_df['rec_loss'].astype('category').cat.codes
# Create 4 quality bins for coloring
plot_df['quality_rank'] = pd.qcut(plot_df['nisqa_ovrl_mean'], 4, labels=['Low', 'Mid', 'High', 'Best'])

parallel_coordinates(
    plot_df[['zeta', 'alpha', 'warmup_steps', 'beta_min', 'rec_loss_idx', 'quality_rank']], 
    'quality_rank', 
    colormap='viridis',
    alpha=0.6
)
plt.title("HP Paths: Identifying the 'Best' Cluster")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(dir.parent / 'eda_parallel_paths.png')

# --- 5. Boxplot: Variance across Top Configs ---
# This shows if your 'Best' config is consistent across all VCTK samples.
plt.figure(figsize=(14, 7))
# Create a readable label for the x-axis
df_raw['config_label'] = df_raw.apply(
    lambda x: f"z{x.zeta}_a{x.alpha}_{x.rec_loss}", axis=1
)
# Filter raw data to only show the top 10 configs identified earlier
top_labels = top_configs.apply(
    lambda x: f"z{x.zeta}_a{x.alpha}_{x.rec_loss}", axis=1
).tolist()
df_top_raw = df_raw[df_raw['config_label'].isin(top_labels)]

sns.boxplot(data=df_top_raw, x='config_label', y='nisqa_ovrl', palette="vlag")
plt.xticks(rotation=45, ha='right')
plt.title("Score Distribution for Top 10 Configurations")
plt.tight_layout()
plt.savefig(dir.parent / 'eda_top_configs_variance.png')

# --- 6. Flagging High-Diff Outliers ---
# Highlighting configs where the spatial image might be collapsing (drift > 0.4)
imbalance_threshold = 0.4
high_diff = df_stats[df_stats['nisqa_diff_mean'].abs() > imbalance_threshold]

if not high_diff.empty:
    print(f"\n--- WARNING: Configs with High Binaural Imbalance (abs_diff > {imbalance_threshold}) ---")
    print(high_diff[['zeta', 'alpha', 'rec_loss', 'nisqa_diff_mean', 'nisqa_ovrl_mean']])

print(f"\nEDA Complete. Plots saved to: {dir.parent}")