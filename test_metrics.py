import re
import torch
import soundfile as sf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Metric Imports
from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment as nisqa
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as pesq
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility as stoi
from pandas.plotting import parallel_coordinates

# --- Metrics Helper Functions ---

def _sisdr_time_safe(est, ref, eps=1e-8):
    """Zero-mean SI-SDR."""
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    ref_energy = (ref**2).sum(dim=-1, keepdim=True).clamp_min(eps)
    scale = (est * ref).sum(dim=-1, keepdim=True) / ref_energy
    s_target = scale * ref
    e_noise = est - s_target
    num = (s_target**2).sum(dim=-1).clamp_min(eps)
    den = (e_noise**2).sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(num / den)

def align_signals_max(est, ref, max_shift):
    """Exhaustively finds the best shift within max_shift to maximize correlation."""
    est_1d, ref_1d = est.view(-1), ref.view(-1)
    L = min(len(est_1d), len(ref_1d), 32000)
    ref_chunk = ref_1d[:L]
    est_chunk = torch.nn.functional.pad(est_1d[:L], (max_shift, max_shift))
    
    corr = torch.nn.functional.conv1d(est_chunk.view(1, 1, -1), ref_chunk.view(1, 1, -1)).view(-1)
    best_lag = torch.argmax(corr).item() - max_shift
    
    if best_lag > 0:
        e, r = est_1d[best_lag:], ref_1d[:len(est_1d)-best_lag]
    else:
        e, r = est_1d[:len(ref_1d)-abs(best_lag)], ref_1d[abs(best_lag):]
    return e.view(1, 1, -1), r.view(1, 1, -1), best_lag

# --- Updated Audio Param Extraction ---

def extract_audio_params(filename):
    """
    Supports the new format: pXXX_YYY_micZ_zeta_0.3_M_8_warmup_steps_5_lambda_stft_0.1_lambda_h_0.001_eta_0.8
    """
    # Capture pXXX_YYY_micZ
    sample_match = re.search(r"(p\d+_\d+_mic\d+)", filename)
    sample_id = sample_match.group(1) if sample_match else "unknown"
    
    patterns = {
        "zeta": r"zeta_([\d.]+)",
        "M": r"_M_(\d+)",
        "warmup": r"warmup_steps_(\d+)",
        "l_stft": r"lambda_stft_([\d.]+)",
        "l_h": r"lambda_h_([\d.]+)",
        "eta": r"_eta_([\d.]+)"
    }
    
    res = {"sample_id": sample_id}
    for k, v in patterns.items():
        m = re.search(v, filename)
        if m:
            val = m.group(1)
            res[k] = float(val) if '.' in val else int(val)
    return res

# --- Main Metric Calculation ---

def calculate_all_metrics(recon_path, original_dir, fs=16000):
    # 1. Load Reconstructed
    s_est, fs_recon = sf.read(recon_path)
    est_t = torch.from_numpy(s_est).float()
    params = extract_audio_params(recon_path.stem)
    
    orig_path = original_dir / f"{params['sample_id']}.wav"
    metrics = {**params}
    
    if orig_path.exists():
        s_ref, _ = sf.read(orig_path)
        ref_t = torch.from_numpy(s_ref).float()
        
        # Alignment for SI-SDR/PESQ/ESTOI
        e_a, r_a, lag = align_signals_max(est_t, ref_t, int(0.5 * fs))
        min_l = min(e_a.shape[-1], r_a.shape[-1])
        e_a, r_a = e_a[..., :min_l], r_a[..., :min_l]
        
        # SI-SDR
        metrics["sisdr"] = round(_sisdr_time_safe(e_a, r_a).item(), 3)
        
        # PESQ (Wideband mode 'wb' for 16kHz)
        metrics["pesq"] = round(pesq(e_a, r_a, fs, mode='wb').item(), 3)
        
        # ESTOI (extended=True for ESTOI)
        metrics["estoi"] = round(stoi(e_a, r_a, fs, extended=True).item(), 3)
        
        metrics["delay_sec"] = round(lag / fs, 4)
    else:
        metrics.update({"sisdr": -99, "pesq": 0, "estoi": 0, "delay_sec": 0})

    # NISQA (Non-intrusive)
    s_nisqa = torch.from_numpy(s_est.T).float()
    if s_nisqa.ndim == 1: s_nisqa = s_nisqa.unsqueeze(0)
    metrics["nisqa"] = round(nisqa(s_nisqa, fs).mean().item(), 3)
    
    return metrics

# --- Execution ---

recon_dir = Path('/home/workspace/yoavellinson/buddy_mc/experiments/monaural_testing_gridsearch/test14_04_2026/monaural_dereverberation/VCTK_16k_monaural_res_h_proj_no_eta/reconstructed')
original_dir = recon_dir.parent / 'original'

files = list(recon_dir.glob('*.wav'))
results = [calculate_all_metrics(f, original_dir) for f in tqdm(files, desc="Calculating Metrics")]

df_raw = pd.DataFrame(results)

# Summary Stats
meta = ['sample_id', 'nisqa', 'sisdr', 'pesq', 'estoi', 'delay_sec']
group_cols = [c for c in df_raw.columns if c not in meta]
df_stats = df_raw.groupby(group_cols)[['nisqa', 'sisdr', 'pesq', 'estoi']].agg(['mean', 'std']).reset_index()
df_stats.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in df_stats.columns.values]

df_raw.to_csv(recon_dir.parent / 'extended_metrics_raw.csv', index=False)
df_stats.to_csv(recon_dir.parent / 'extended_metrics_summary.csv', index=False)

print("\nTOP 5 BY PESQ (Perceptual Quality):")
print(df_stats.sort_values('pesq_mean', ascending=False).head(5))

print("\nTOP 5 BY SI-SDR (Signal Fidelity):")
print(df_stats.sort_values('sisdr_mean', ascending=False).head(5))