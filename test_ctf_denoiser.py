import argparse
import math

import torch
import torch.nn.functional as F

from train_ctf_denoiser import CTFTransform


def _make_clean_audio(batch, length, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(batch, 1, length, device=device, generator=generator)

    # Mildly speech-like colored excitation: less pathological than a pure impulse,
    # still broadband enough for well-conditioned CTF LS estimation.
    kernel = torch.tensor([0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05], device=device)
    kernel = kernel.view(1, 1, -1)
    x = F.conv1d(F.pad(noise, (3, 3), mode="reflect"), kernel)
    x = x / x.std(dim=-1, keepdim=True).clamp_min(1e-8)
    return x


def _make_stereo_rir(length, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1000)
    t = torch.arange(length, device=device, dtype=torch.float32)
    decay = torch.exp(-t / max(1.0, length / 5.0))

    h = torch.randn(2, length, device=device, generator=generator) * decay[None]
    h[:, 0] += torch.tensor([1.0, 0.85], device=device)

    # Add a small interaural offset so the stereo channels are not identical.
    if length > 9:
        h[1] = torch.roll(h[1], shifts=3)
        h[1, :3] = 0.0

    h = h / h.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return h


def _convolve_mono_to_stereo(x, h):
    batch, _, length = x.shape
    rir_len = h.shape[-1]
    wav = x.repeat(1, 2, 1)
    weight = h.flip(-1).to(device=x.device, dtype=x.dtype)[:, None, :]
    y = F.conv1d(F.pad(wav, (rir_len - 1, rir_len - 1)), weight, groups=2)
    return y[..., :length]


def _estimate_and_score(x, y, args, m):
    ctf = CTFTransform(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        M=m,
        normalize=True,
        ridge=args.ridge,
    )
    h_ctf, X, Y = ctf.estimate_from_pair(x, y, return_specs=True)
    metrics = ctf.reconstruction_metrics(x, y, h_ctf, X=X, Y=Y)
    channels = ctf.complex_to_channels(h_ctf)
    return metrics, channels.shape


def _brir_time_to_ctf_internal(h_time, args, m):
    if h_time.ndim == 2:
        h_time = h_time.unsqueeze(0)
    batch, channels, rir_len = h_time.shape
    window = torch.hann_window(args.n_fft, periodic=False, device=h_time.device, dtype=h_time.dtype)
    ctf = torch.zeros(
        batch,
        channels,
        args.n_fft // 2 + 1,
        m,
        device=h_time.device,
        dtype=torch.complex64,
    )

    for tap in range(m):
        start = tap * args.hop_length
        frame = torch.zeros(batch, channels, args.n_fft, device=h_time.device, dtype=h_time.dtype)
        if start < rir_len:
            chunk = h_time[..., start:start + args.n_fft]
            frame[..., :chunk.shape[-1]] = chunk
        ctf[..., tap] = torch.fft.rfft(frame * window, n=args.n_fft, dim=-1)

    # Match BinauralToMonoEulerHeunSamplerDPS.apply_ctf_stft(), which uses H.conj().
    return ctf.conj()


def _score_direct_brir_ctf(x, y, rir, args, m):
    ctf = CTFTransform(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        M=m,
        normalize=True,
        ridge=args.ridge,
    )
    h_ctf = _brir_time_to_ctf_internal(rir, args, m)
    metrics = ctf.reconstruction_metrics(x, y, h_ctf)
    channels = ctf.complex_to_channels(h_ctf)
    return metrics, channels.shape


def _sampler_style_estimate(x, y, args, m):
    ctf = CTFTransform(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        M=m,
        normalize=True,
        ridge=args.ridge,
    )
    X = ctf.stft(x)
    Y = ctf.stft(y)
    if X.shape[1] == 1 and Y.shape[1] > 1:
        X = X.expand(-1, Y.shape[1], -1, -1)

    X_pad = F.pad(X, (m - 1, 0))
    X_hist = X_pad.unfold(-1, m, 1).flip(-1)

    # Mirrors BinauralToMonoEulerHeunSamplerDPS.compute_simple_correlations()
    # and M_step_ls(): q = solve(X^H X + ridge I, X^H y), H = conj(q).
    Rxx = torch.einsum('bcftm,bcftn->bcfmn', X_hist.conj(), X_hist)
    rxy = torch.einsum('bcftm,bcft->bcfm', X_hist.conj(), Y).unsqueeze(-1)
    eye = torch.eye(m, device=Rxx.device, dtype=Rxx.dtype)
    diag_power = Rxx.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-8)
    Rxx = Rxx + args.ridge * diag_power[..., None, None] * eye
    return torch.linalg.solve(Rxx, rxy).squeeze(-1).conj()


def _validate_training_matches_sampler_style(x, y, args, m):
    ctf = CTFTransform(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        M=m,
        normalize=True,
        ridge=args.ridge,
    )
    h_train, X, _ = ctf.estimate_from_pair(x, y, return_specs=True)
    h_sampler = _sampler_style_estimate(x, y, args, m)
    h_rel = (h_train - h_sampler).abs().pow(2).mean().sqrt() / h_sampler.abs().pow(2).mean().sqrt().clamp_min(1e-12)
    y_train = ctf.apply_ctf_stft(X, h_train)
    y_sampler = ctf.apply_ctf_stft(X, h_sampler)
    y_rel = (y_train - y_sampler).abs().pow(2).mean().sqrt() / y_sampler.abs().pow(2).mean().sqrt().clamp_min(1e-12)
    return float(h_rel.detach().cpu()), float(y_rel.detach().cpu())


def _fmt(metrics):
    return (
        f"stft_rel_mse={metrics['ctf_stft_rel_mse']:.6g} "
        f"time_rel_mse={metrics['ctf_time_rel_mse']:.6g} "
        f"time_sisdr={metrics['ctf_time_sisdr_db']:.2f} dB"
    )


def main():
    parser = argparse.ArgumentParser(description="Validate CTF LS estimation/reconstruction tightness.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--rir-length", type=int, default=256)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--loose-m", type=int, default=4)
    parser.add_argument("--tight-m", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--min-sisdr-db", type=float, default=25.0)
    parser.add_argument("--max-time-rel-mse", type=float, default=5e-3)
    parser.add_argument("--max-stft-rel-mse", type=float, default=5e-2)
    parser.add_argument("--max-sampler-convention-rel-diff", type=float, default=1e-6)
    parser.add_argument(
        "--check-direct-brir-ctf",
        action="store_true",
        help="Also require the direct chunked BRIR-to-CTF conversion to pass thresholds.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    x = _make_clean_audio(args.batch, args.length, device, args.seed)
    rir = _make_stereo_rir(args.rir_length, device, args.seed)
    y = _convolve_mono_to_stereo(x, rir)

    loose_metrics, loose_shape = _estimate_and_score(x, y, args, args.loose_m)
    tight_metrics, tight_shape = _estimate_and_score(x, y, args, args.tight_m)
    direct_metrics, direct_shape = _score_direct_brir_ctf(x, y, rir, args, args.tight_m)
    h_rel_diff, y_rel_diff = _validate_training_matches_sampler_style(x, y, args, args.tight_m)

    print("Synthetic CTF reconstruction validation")
    print(f"device={device} length={args.length} rir_length={args.rir_length}")
    print(f"n_fft={args.n_fft} hop={args.hop_length} ridge={args.ridge}")
    print(f"loose LS M={args.loose_m} shape={tuple(loose_shape)} {_fmt(loose_metrics)}")
    print(f"tight LS M={args.tight_m} shape={tuple(tight_shape)} {_fmt(tight_metrics)}")
    print(f"direct BRIR CTF M={args.tight_m} shape={tuple(direct_shape)} {_fmt(direct_metrics)}")
    print(
        f"train estimate_from_pair vs sampler-style LS: "
        f"h_rel_diff={h_rel_diff:.3g} yhat_rel_diff={y_rel_diff:.3g}"
    )

    failures = []
    if tight_shape[1] != 4:
        failures.append(f"expected real CTF channel count 4, got {tight_shape[1]}")
    if not math.isfinite(tight_metrics["ctf_time_sisdr_db"]):
        failures.append("tight CTF SI-SDR is not finite")
    if tight_metrics["ctf_time_sisdr_db"] < args.min_sisdr_db:
        failures.append(
            f"tight CTF SI-SDR {tight_metrics['ctf_time_sisdr_db']:.2f} dB "
            f"< {args.min_sisdr_db:.2f} dB"
        )
    if tight_metrics["ctf_time_rel_mse"] > args.max_time_rel_mse:
        failures.append(
            f"tight CTF time relative MSE {tight_metrics['ctf_time_rel_mse']:.6g} "
            f"> {args.max_time_rel_mse:.6g}"
        )
    if tight_metrics["ctf_stft_rel_mse"] > args.max_stft_rel_mse:
        failures.append(
            f"tight CTF STFT relative MSE {tight_metrics['ctf_stft_rel_mse']:.6g} "
            f"> {args.max_stft_rel_mse:.6g}"
        )
    if tight_metrics["ctf_time_sisdr_db"] + 1.0 < loose_metrics["ctf_time_sisdr_db"]:
        failures.append("tight M unexpectedly reconstructs worse than loose M by more than 1 dB")
    if h_rel_diff > args.max_sampler_convention_rel_diff:
        failures.append(
            f"training CTF differs from sampler-style LS: h_rel_diff={h_rel_diff:.6g} "
            f"> {args.max_sampler_convention_rel_diff:.6g}"
        )
    if y_rel_diff > args.max_sampler_convention_rel_diff:
        failures.append(
            f"training CTF application differs from sampler-style LS: yhat_rel_diff={y_rel_diff:.6g} "
            f"> {args.max_sampler_convention_rel_diff:.6g}"
        )
    if direct_shape[1] != 4:
        failures.append(f"expected direct BRIR CTF real channel count 4, got {direct_shape[1]}")
    if args.check_direct_brir_ctf:
        if direct_metrics["ctf_time_sisdr_db"] < args.min_sisdr_db:
            failures.append(
                f"direct BRIR CTF SI-SDR {direct_metrics['ctf_time_sisdr_db']:.2f} dB "
                f"< {args.min_sisdr_db:.2f} dB"
            )
        if direct_metrics["ctf_time_rel_mse"] > args.max_time_rel_mse:
            failures.append(
                f"direct BRIR CTF time relative MSE {direct_metrics['ctf_time_rel_mse']:.6g} "
                f"> {args.max_time_rel_mse:.6g}"
            )

    if failures:
        print("FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("PASSED")


if __name__ == "__main__":
    main()
