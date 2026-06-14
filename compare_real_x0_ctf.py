import csv
import os
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import open_dict

from testing.BinauralToMonoEulerHeunSamplerDPS import BinauralToMonoEulerHeunSamplerDPS
import utils.log as utils_logging


class DummyModel(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("compare_real_x0_ctf.py does not run the denoiser")


def _sisdr_db(est, ref, eps=1e-8):
    est = est.to(torch.float32)
    ref = ref.to(torch.float32)
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    scale = (est * ref).sum(dim=-1, keepdim=True) / ref.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
    target = scale * ref
    noise = est - target
    ratio = target.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def _per_frequency_projection_error(h_est, h_ref):
    h_est = h_est.detach().to(dtype=h_ref.dtype, device=h_ref.device)
    h_ref = h_ref.detach()
    h_hat_f = h_est.permute(2, 0, 1, 3).reshape(h_est.shape[2], -1)
    h_f = h_ref.permute(2, 0, 1, 3).reshape(h_ref.shape[2], -1)
    inner = (h_f.conj() * h_hat_f).sum(dim=-1).abs().pow(2)
    denom = (h_f.abs().pow(2).sum(dim=-1) * h_hat_f.abs().pow(2).sum(dim=-1)).clamp_min(1e-12)
    return (1.0 - inner / denom).clamp_min(0.0).real


def _write_per_freq_csv(path, sampler, h_tilde, h_ref):
    err = _per_frequency_projection_error(h_tilde, h_ref).detach().cpu()
    err_decayed = _per_frequency_projection_error(
        sampler.apply_guidance_decay_to_h(h_tilde),
        sampler.apply_guidance_decay_to_h(h_ref),
    ).detach().cpu()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["freq_bin", "ctf_proj_err", "ctf_proj_err_decayed"])
        for freq_bin, (e, ed) in enumerate(zip(err.tolist(), err_decayed.tolist())):
            writer.writerow([freq_bin, e, ed])


def _prepare_example(args, device, index):
    test_set = hydra.utils.instantiate(args.dset.test)
    original, rir, filename, h_orig, hrtf = test_set[index]

    seg_raw = torch.as_tensor(original, device=device, dtype=torch.float32)
    y_raw = torch.as_tensor(rir, device=device, dtype=torch.float32)
    h_orig = torch.as_tensor(h_orig, device=device, dtype=torch.float32)

    input_scaling = args.tester.get("input_scaling", {})
    target_sigma = input_scaling.get(
        "target_sigma",
        args.tester.posterior_sampling.warm_initialization.scaling_factor,
    )
    cond_to_target_std = input_scaling.get("cond_to_target_std", 0.3843)

    y = y_raw * (target_sigma * cond_to_target_std) / (y_raw.std() + 1e-8)
    x0 = seg_raw * target_sigma / (seg_raw.std() + 1e-8)

    if x0.ndim == 1:
        x0 = x0.unsqueeze(0).unsqueeze(0)
    elif x0.ndim == 2:
        x0 = x0.unsqueeze(0)

    if y.ndim == 2:
        y = y.unsqueeze(0)

    return x0, y, h_orig, str(filename)


def _make_sampler(args, device):
    diff_params = hydra.utils.instantiate(args.diff_params)
    model = DummyModel().to(device)
    sampler = BinauralToMonoEulerHeunSamplerDPS(model, diff_params, args)
    return sampler


@hydra.main(config_path="conf", config_name="conf_VCTK_binaural_to_mono.yaml", version_base=None)
def main(args):
    gpu = int(args.get("gpu", 0))
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(gpu)

    with open_dict(args):
        args.tester.sampling_params.h_estimator = "ls"
        args.tester.sampling_params.h_first_init = "estimate"
        args.tester.sampling_params.eta = 0.0
        args.tester.sampling_params.beta = 0.0
        args.dset.test.num_examples = max(int(args.dset.test.get("num_examples", 1)), 1)

    index = int(args.get("ctf_debug_index", 0))
    out_dir = Path(args.get("ctf_debug_out", "experiments_binaural_to_mono/ctf_real_x0_debug"))
    out_dir.mkdir(parents=True, exist_ok=True)

    x0, y, h_orig, filename = _prepare_example(args, device, index)
    sampler = _make_sampler(args, device)
    sampler.y = y
    sampler.Y = sampler.stft(y)
    sampler.h_debug_dir = str(out_dir)
    sample_name = Path(filename).stem
    sampler.h_debug_name = sample_name

    X0 = sampler.stft(x0)
    h_tilde_ctf = sampler.M_step_ls(X0, use_running=False)

    # This is the same pair-LS oracle reference currently used in the informed diagnostic path.
    h_orig_ctf_pair_ls = sampler.estimate_ctf_from_pair(x0, y)

    # This is a convention-aligned conversion from the time-domain BRIR:
    # conv_h(probe, h_orig) followed by the same LS CTF estimate used for h_tilde.
    h_orig_ctf_probe_ls = sampler.brir_time_to_ctf_probe_ls(
        h_orig,
        length=y.shape[-1],
        M=sampler.M,
    ).to(device=device, dtype=h_tilde_ctf.dtype)

    # This is the old direct conversion from the provided time-domain BRIR.
    h_orig_ctf_time = sampler.brir_time_to_ctf_internal(h_orig, M=sampler.M).to(
        device=device,
        dtype=h_tilde_ctf.dtype,
    )

    rows = []
    estimated_reverb_signals = {}
    for ref_name, h_ref in [
        ("h_orig_pair_ls", h_orig_ctf_pair_ls),
        ("h_orig_probe_ls", h_orig_ctf_probe_ls),
        ("h_orig_time_brir", h_orig_ctf_time),
    ]:
        metrics = sampler.compare_ctfs(h_tilde_ctf, h_ref)
        Y_hat = sampler.apply_ctf_stft(X0, h_ref)
        y_hat = sampler.istft(Y_hat, length=y.shape[-1])
        estimated_reverb_signals[ref_name] = y_hat.detach()
        sisdr = _sisdr_db(y_hat, y).squeeze(0).detach().cpu().tolist()
        row = {
            "sample": sample_name,
            "index": index,
            "comparison": f"h_tilde_real_x0_y_vs_{ref_name}",
            **metrics,
            "y_recon_sisdr_left_db": sisdr[0],
            "y_recon_sisdr_right_db": sisdr[1],
            "y_recon_sisdr_mean_db": sum(sisdr) / len(sisdr),
        }
        rows.append(row)
        _write_per_freq_csv(out_dir / f"{sample_name}_h_tilde_vs_{ref_name}_per_freq.csv", sampler, h_tilde_ctf, h_ref)

    # Also test whether the estimated h_tilde itself reconstructs y from the real x0.
    Y_hat_tilde = sampler.apply_ctf_stft(X0, h_tilde_ctf)
    y_hat_tilde = sampler.istft(Y_hat_tilde, length=y.shape[-1])
    estimated_reverb_signals["h_tilde_real_x0_y"] = y_hat_tilde.detach()
    sisdr_tilde = _sisdr_db(y_hat_tilde, y).squeeze(0).detach().cpu().tolist()
    rows.append({
        "sample": sample_name,
        "index": index,
        "comparison": "apply_h_tilde_real_x0_y_to_reconstruct_y",
        "ctf_proj_err_mean": 0.0,
        "ctf_proj_err_median": 0.0,
        "ctf_proj_err_max": 0.0,
        "ctf_proj_err_min": 0.0,
        "ctf_proj_err_decayed_mean": 0.0,
        "ctf_proj_err_decayed_median": 0.0,
        "ctf_proj_err_decayed_max": 0.0,
        "ctf_proj_err_decayed_min": 0.0,
        "mag_corr": 1.0,
        "late_direct_est_db": float(sampler.ctf_late_direct_db(h_tilde_ctf).detach().cpu()),
        "late_direct_ref_db": float(sampler.ctf_late_direct_db(h_tilde_ctf).detach().cpu()),
        "y_recon_sisdr_left_db": sisdr_tilde[0],
        "y_recon_sisdr_right_db": sisdr_tilde[1],
        "y_recon_sisdr_mean_db": sum(sisdr_tilde) / len(sisdr_tilde),
    })

    keys = list(rows[0].keys())
    csv_path = out_dir / f"{sample_name}_real_x0_ctf_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    sampler.save_h_tilde_wav(h_tilde_ctf, "_h_tilde_from_real_x0_y")
    sampler.save_h_tilde_wav(h_orig_ctf_pair_ls, "_h_orig_pair_ls")
    sampler.save_h_tilde_wav(h_orig_ctf_probe_ls, "_h_orig_probe_ls")
    sampler.save_h_tilde_wav(h_orig_ctf_time, "_h_orig_time_brir")
    for signal_name, y_hat_est in estimated_reverb_signals.items():
        utils_logging.write_audio_file(
            y_hat_est.cpu(),
            args.exp.sample_rate,
            sample_name + f"_estimated_reverb_y_from_{signal_name}",
            path=str(out_dir),
            stereo=True,
            normalize=False,
        )
    utils_logging.write_audio_file(x0.detach().cpu(), args.exp.sample_rate, sample_name + "_clean_mono_x0", path=str(out_dir), stereo=False, normalize=False)
    utils_logging.write_audio_file(y.detach().cpu(), args.exp.sample_rate, sample_name + "_observed_y", path=str(out_dir), stereo=True, normalize=False)

    print(f"Wrote {csv_path}")
    for row in rows:
        print(
            row["comparison"],
            f"ctf_mean={row['ctf_proj_err_mean']:.6f}",
            f"ctf_decayed_mean={row['ctf_proj_err_decayed_mean']:.6f}",
            f"y_sisdr_mean={row['y_recon_sisdr_mean_db']:.2f} dB",
        )


if __name__ == "__main__":
    import sys
    sys.argv.extend([
        "--config-name=conf_VCTK_binaural_to_mono.yaml",
        "tester=blind_dereverberation_binaural",
        "model_dir=experiments_binaural_to_mono",
        "dset=vctk_16k_4s_binaural",
        "+gpu=0",
        "+ctf_debug_index=0",
        "+ctf_debug_out=experiments_binaural_to_mono/ctf_real_x0_debug",
        "dset.test.num_examples=1",
        "tester.modes=[binaural_dereverberation]",
        "tester.sampling_params.T=500",
        "tester.sampling_params.M=48",
        "tester.sampling_params.alpha=0.35",
        "tester.sampling_params.beta=0",
        "tester.sampling_params.eta=0",
    ])
    main()
