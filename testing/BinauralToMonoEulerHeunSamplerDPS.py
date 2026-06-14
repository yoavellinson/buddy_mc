from tqdm import tqdm
import utils.log as utils_logging
import torch
from nara_wpe.wpe import wpe
import os
import torch.nn as nn
from testing.EulerHeunSampler import EulerHeunSampler
import torch.nn.functional as F
from torch.nn.functional import pad

class BinauralToMonoEulerHeunSamplerDPS(EulerHeunSampler):
    """
        Euler Heun sampler for DPS 
        inverse problem solver
    """

    def __init__(self, model, diff_params, args):
        super().__init__(model, diff_params, args)
        self.zeta = self.args.tester.posterior_sampling.zeta
        self.stft_options = dict(size=510, shift=128)
        self.rec_loss_sisdr = SiSDRLoss() 
        self.rec_loss_stft = CompressedSTFTLoss()
        self.warmup_steps = self.args.tester.sampling_params.warmup_steps
        self.M = self.args.tester.sampling_params.M
        self.alpha = self.args.tester.sampling_params.alpha
        self.eta = self.args.tester.sampling_params.eta
        self.beta = self.args.tester.sampling_params.get("beta", 0.0)
        self.lambda_stft = self.args.tester.sampling_params.lambda_stft
        self.lambda_sisdr = self.args.tester.sampling_params.lambda_sisdr
        self.h_first_init = self.args.tester.sampling_params.get("h_first_init", "estimate")
        self.h_orig_warmup_mode = self.args.tester.sampling_params.get("h_orig_warmup_mode", "h_orig")
        self.reverb_tail_delay = self.args.tester.sampling_params.get("reverb_tail_delay", 3)
        self.blind_schedule = self.args.tester.sampling_params.get("blind_schedule", "standard")

    def initialize_x(self, shape, device, schedule):
        """
        Initialize mono latent x for binaural-to-mono conditional sampling.

        self.y: [B, 2, T] binaural observation
        shape:  [B, 1, T] mono latent shape
        """

        # Store STFT of binaural observation for likelihood losses
        self.Y = self.stft(self.y)  # [B, 2, F, TT]

        mode = self.args.tester.posterior_sampling.warm_initialization.mode
        sigma_max = schedule[0]

        if mode == "none":
            x = sigma_max * torch.randn(shape, device=device)

        elif mode == "reverb_scaled":
            # Use a single binaural channel as warm start.
            # Averaging channels creates comb-filter / double-speech artifacts
            # because of interaural delays.
            mono_init = self.y[:, :1, :]  # [B,1,T]

            scale = self.args.tester.posterior_sampling.warm_initialization.scaling_factor

            mono_init = scale * mono_init / (
                mono_init.std(dim=-1, keepdim=True) + 1e-8
            )

            x = mono_init + sigma_max * torch.randn(
                shape,
                device=device,
                dtype=mono_init.dtype,
            )

        elif mode == "wpe_scaled":
            print("Processing WPE")

            delay = self.args.tester.posterior_sampling.warm_initialization.wpe.delay
            iterations = self.args.tester.posterior_sampling.warm_initialization.wpe.iterations
            taps = self.args.tester.posterior_sampling.warm_initialization.wpe.taps

            x_pred_list = []

            for b in range(self.y.shape[0]):
                Y_b = self.Y[b]  # [2,F,TT]
                Y_np = Y_b.detach().cpu().numpy().transpose(2, 0, 1)  # [TT,2,F]

                Z = wpe(
                    Y_np,
                    taps=taps,
                    delay=delay,
                    iterations=iterations,
                    statistics_mode="full",
                )  # [TT,2,F]

                Z = torch.from_numpy(Z.transpose(1, 2, 0)).to(
                    device=self.y.device,
                    dtype=self.Y.dtype,
                )  # [2,F,TT]

                y_wpe = self.istft(Z.unsqueeze(0), length=self.y.shape[-1])  # [1,2,T]

                # Use one channel, not mean, to avoid comb-filter/double-speech artifacts
                mono_wpe = y_wpe[:, :1, :]  # left channel: [1,1,T]
                # mono_wpe = y_wpe[:, 1:2, :]  # right channel alternative

                x_pred_list.append(mono_wpe)

            x_pred = torch.cat(x_pred_list, dim=0).to(device=device, dtype=self.y.dtype)

            scale = self.args.tester.posterior_sampling.warm_initialization.scaling_factor
            x_pred = scale * x_pred / (x_pred.std(dim=-1, keepdim=True) + 1e-8)

            x = x_pred + sigma_max * torch.randn(
                shape,
                device=device,
                dtype=x_pred.dtype,
            )

        else:
            raise NotImplementedError(f"Unknown warm initialization mode: {mode}")

        assert x.shape == shape, f"initialize_x produced {x.shape}, expected {shape}"

        return x
    

    def get_Tweedie_estimate(self, x, t_i):
        """
        Compute denoised mono estimate x0_hat.

        x:      [B, 1, T] noisy mono latent
        self.y: [B, 2, T] binaural condition
        t_i:    scalar or [B] noise level / time
        """

        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B,T] -> [B,1,T]

        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B,1,T], got {x.shape}")

        if x.shape[1] != 1:
            raise ValueError(f"Expected mono latent x with C=1, got {x.shape}")

        if not hasattr(self, "y"):
            raise RuntimeError("self.y is not set. Call predict_conditional(y=...) first.")

        cond = self.y

        if cond.ndim == 2:
            cond = cond.unsqueeze(0)

        if cond.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch mismatch: x has batch {x.shape[0]}, cond has batch {cond.shape[0]}"
            )

        if cond.shape[1] != 2:
            raise ValueError(f"Expected binaural condition [B,2,T], got {cond.shape}")

        x_hat = self.diff_params.denoiser(
            xn=x,
            net=self.model,
            t=t_i,
            cond=cond,
        )

        return x_hat

    def reset_buffers(self):
        """Call this at the very beginning of predict() for every new file"""
        if hasattr(self, 'running_Rxx'):
            del self.running_Rxx
        if hasattr(self, 'running_rxy'):
            del self.running_rxy
        if hasattr(self, 'prev_h_tilde'):
            del self.prev_h_tilde
        if hasattr(self, 'saved_h_tilde_0'):
            self.saved_h_tilde_0 = False
        # Clear the CUDA cache to defragment memory
        torch.cuda.empty_cache()

    def stft(self, time_signal):
        """
        Args:
            time_signal: [J, Time_Samples] - Binaural waveform (e.g., [2, 131840])
        Returns:
            stft_signal: [J, F, T] - Complex STFT (e.g., [2, 257, 1031])
        """
        # 1. Extract Options
        B,C,T = time_signal.shape

        time_signal = time_signal.reshape(B*C,T)
        n_fft = self.stft_options['size']
        hop_length = self.stft_options['shift']
        win_length = n_fft # Standard for perfect reconstruction
        
        device = time_signal.device
        window = torch.hann_window(win_length, periodic=False).to(device)

        # 2. Compute STFT
        # return_complex=True is mandatory for your CTF estimation grad
        stft_signal = torch.stft(
            time_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,          # Matches 'fading' logic
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        _,F,T = stft_signal.shape
        return stft_signal.reshape(B,C,F,T)
    
    def istft(self, stft_signal, length=None):
        n_fft = self.stft_options["size"]
        hop_length = self.stft_options["shift"]
        win_length = n_fft

        device = stft_signal.device
        window = torch.hann_window(win_length, periodic=False).to(device)

        if stft_signal.ndim == 4:
            B, C, Freq, Frames = stft_signal.shape
            stft_signal = stft_signal.reshape(B * C, Freq, Frames)

            time_signal = torch.istft(
                stft_signal,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
                length=length,
                return_complex=False,
            )

            return time_signal.reshape(B, C, -1)

        elif stft_signal.ndim == 3:
            return torch.istft(
                stft_signal,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
                length=length,
                return_complex=False,
            )

        else:
            raise ValueError(f"Expected STFT shape [B,C,F,T] or [C,F,T], got {stft_signal.shape}")
    
    def _prepare_brir_time(self, h_time, channels=2):
        h_time = h_time.to(dtype=torch.float32)
        while h_time.ndim > 3 and 1 in h_time.shape:
            h_time = h_time.squeeze(0)

        if h_time.ndim == 2:
            if h_time.shape[0] == channels:
                h_time = h_time.unsqueeze(0)
            elif h_time.shape[1] == channels:
                h_time = h_time.T.unsqueeze(0)
            else:
                raise ValueError(f"Cannot infer channel axis for BRIR shape {h_time.shape}")
        elif h_time.ndim == 3:
            if h_time.shape[1] == channels:
                pass
            elif h_time.shape[2] == channels:
                h_time = h_time.transpose(1, 2)
            else:
                raise ValueError(f"Cannot infer channel axis for BRIR shape {h_time.shape}")
        else:
            raise ValueError(f"Expected BRIR with 2 or 3 dims, got {h_time.shape}")

        return h_time

    def estimate_ctf_from_pair(self, x_ref, y_ref):
        """Oracle/reference CTF in the same LS convention as h_tilde.

        This is the reliable CTF reference for diagnostics: it finds H such
        that apply_ctf_stft(STFT(x_ref), H) approximates STFT(y_ref).
        """
        if x_ref.ndim == 2:
            x_ref = x_ref.unsqueeze(1)
        if y_ref.ndim == 2:
            y_ref = y_ref.unsqueeze(0)
        old_Y = getattr(self, "Y", None)
        self.Y = self.stft(y_ref)
        H_ref = self.M_step_ls(self.stft(x_ref), use_running=False)
        if old_Y is not None:
            self.Y = old_Y
        return H_ref

    def brir_time_to_ctf_internal(self, h_time, M=None):
        if M is None:
            M = self.M

        h_time = self._prepare_brir_time(h_time).to(device=self.y.device)
        B, C, L = h_time.shape
        n_fft = self.stft_options["size"]
        hop = self.stft_options["shift"]
        window = torch.hann_window(n_fft, periodic=False, device=h_time.device, dtype=h_time.dtype)

        G = torch.zeros(
            B,
            C,
            n_fft // 2 + 1,
            M,
            device=h_time.device,
            dtype=torch.complex64,
        )

        for m in range(M):
            start = m * hop
            frame = torch.zeros(B, C, n_fft, device=h_time.device, dtype=h_time.dtype)
            if start < L:
                chunk = h_time[..., start:start + n_fft]
                frame[..., :chunk.shape[-1]] = chunk
            frame = frame * window
            G[..., m] = torch.fft.rfft(frame, n=n_fft, dim=-1)

        # Internal convention: apply_ctf_stft multiplies by H.conj().
        return G.conj()

    def brir_time_to_ctf_probe_ls(self, h_time, length=None, M=None, seed=0):
        """Convert a time-domain BRIR to the sampler's CTF convention by probing.

        The direct chunked conversion is sensitive to STFT centering, windowing,
        and CTF tap alignment. This route creates a deterministic dry probe,
        reverberates it with conv_h(), then estimates the CTF with the exact LS
        path used for h_tilde.
        """
        if M is None:
            M = self.M
        if length is None:
            if hasattr(self, "y") and self.y is not None:
                length = self.y.shape[-1]
            else:
                length = self.args.exp.audio_len

        h_time = self._prepare_brir_time(h_time).to(device=self.y.device)
        B = h_time.shape[0]
        generator = torch.Generator(device=self.y.device)
        generator.manual_seed(int(seed))
        x_probe = torch.randn(
            B,
            1,
            int(length),
            device=self.y.device,
            dtype=h_time.dtype,
            generator=generator,
        )
        x_probe = x_probe / x_probe.std(dim=-1, keepdim=True).clamp_min(1e-8)
        y_probe = self.conv_h(x_probe, h_time)

        old_M = self.M
        self.M = M
        try:
            h_ctf = self.estimate_ctf_from_pair(x_probe, y_probe)
        finally:
            self.M = old_M
        return h_ctf

    def ctf_late_direct_db(self, h_tilde):
        delay = min(max(1, int(self.reverb_tail_delay)), h_tilde.shape[-1] - 1)
        direct_energy = h_tilde[..., 0].abs().pow(2).mean().clamp_min(1e-12)
        late_energy = h_tilde[..., delay:].abs().pow(2).mean().clamp_min(1e-12)
        return 10.0 * torch.log10(late_energy / direct_energy)

    def compare_ctfs(self, h_est, h_ref):
        h_est = h_est.detach().to(dtype=h_ref.dtype, device=h_ref.device)
        h_ref = h_ref.detach()

        def per_frequency_projection_error(h_hat, h):
            # Eq.-style CTF error per frequency:
            # 1 - |h^H h_hat|^2 / (||h||^2 ||h_hat||^2).
            # Vectors are flattened over batch, channel, and CTF taps for each f.
            h_hat_f = h_hat.permute(2, 0, 1, 3).reshape(h_hat.shape[2], -1)
            h_f = h.permute(2, 0, 1, 3).reshape(h.shape[2], -1)
            inner = (h_f.conj() * h_hat_f).sum(dim=-1).abs().pow(2)
            denom = (
                h_f.abs().pow(2).sum(dim=-1)
                * h_hat_f.abs().pow(2).sum(dim=-1)
            ).clamp_min(1e-12)
            return (1.0 - inner / denom).clamp_min(0.0).real

        def summarize(prefix, err_f):
            return {
                f"{prefix}_mean": float(err_f.mean().detach().cpu()),
                f"{prefix}_median": float(err_f.median().detach().cpu()),
                f"{prefix}_max": float(err_f.max().detach().cpu()),
                f"{prefix}_min": float(err_f.min().detach().cpu()),
            }

        mag_est = h_est.abs().reshape(-1)
        mag_ref = h_ref.abs().reshape(-1)
        mag_est = mag_est - mag_est.mean()
        mag_ref = mag_ref - mag_ref.mean()
        mag_corr = (mag_est * mag_ref).sum() / (
            mag_est.pow(2).sum().sqrt() * mag_ref.pow(2).sum().sqrt()
        ).clamp_min(1e-12)

        h_est_decayed = self.apply_guidance_decay_to_h(h_est)
        h_ref_decayed = self.apply_guidance_decay_to_h(h_ref)

        err_f = per_frequency_projection_error(h_est, h_ref)
        err_decayed_f = per_frequency_projection_error(h_est_decayed, h_ref_decayed)

        metrics = {}
        metrics.update(summarize("ctf_proj_err", err_f))
        metrics.update(summarize("ctf_proj_err_decayed", err_decayed_f))
        metrics.update({
            "mag_corr": float(mag_corr.detach().cpu()),
            "late_direct_est_db": float(self.ctf_late_direct_db(h_est).detach().cpu()),
            "late_direct_ref_db": float(self.ctf_late_direct_db(h_ref).detach().cpu()),
        })
        return metrics

    def record_h_tilde_h_orig_comparison(self, h_tilde, step):
        if not hasattr(self, "h_orig_ctf"):
            return
        record = {"step": int(step)}
        record.update(self.compare_ctfs(h_tilde, self.h_orig_ctf))
        self.h_compare_records.append(record)

    def save_h_tilde_h_orig_comparison(self):
        debug_dir = getattr(self, "h_debug_dir", None)
        if debug_dir is None or not hasattr(self, "h_compare_records") or not self.h_compare_records:
            return

        os.makedirs(debug_dir, exist_ok=True)
        debug_name = getattr(self, "h_debug_name", "sample")
        path = os.path.join(debug_dir, debug_name + "_h_tilde_vs_h_orig_ctf.csv")
        keys = [
            "step",
            "ctf_proj_err_mean",
            "ctf_proj_err_median",
            "ctf_proj_err_max",
            "ctf_proj_err_min",
            "ctf_proj_err_decayed_mean",
            "ctf_proj_err_decayed_median",
            "ctf_proj_err_decayed_max",
            "ctf_proj_err_decayed_min",
            "mag_corr",
            "late_direct_est_db",
            "late_direct_ref_db",
        ]
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            for record in self.h_compare_records:
                f.write(",".join(str(record[k]) for k in keys) + "\n")

    def conv_h(self, wav, h):

        # wav: [B,1,T]
        # h: [2,L]
        B, _, T = wav.shape
        h = h.squeeze()
        if h.shape[0] != 2:
            h = h.T
        h = h.to(wav.device, wav.dtype)

        # duplicate mono speech to both ears
        wav = wav.repeat(1, 2, 1)  # [B,2,T]
        # grouped conv
        h = h.flip(-1)[:, None, :]  # [2,1,L]
        pad = h.shape[-1] - 1
        out = F.conv1d(
            F.pad(wav, (pad, pad)),
            h,
            groups=2
        )

        return out[..., :T]
       
    def get_likelihood_score_h_orig(self, x_den, x, h_orig,i):
        y_hat = self.conv_h(x_den,h_orig)
        Y_hat = self.stft(y_hat)
        rec = self.lambda_sisdr*self.rec_loss_sisdr(y_hat,self.y)
        rec += self.lambda_stft*self.rec_loss_stft(Y_hat,self.Y)
        rec_grads = torch.autograd.grad(outputs=rec, inputs=x)[0]
        normguide = torch.norm(rec_grads)/(self.args.exp.audio_len**0.5)
        return self.zeta / (normguide+1e-8) * rec_grads, rec

    def stepEM_h_orig(self, x_i, t_i, t_iplus1, gamma_i,i,h_orig, eps=1e-5):
        x_den, x_hat, t_hat = self.E_step(x_i, t_i, gamma_i)
        use_h_orig_guidance = (
            self.step_counter >= self.warmup_steps
            or self.h_orig_warmup_mode == "h_orig"
        )
        if use_h_orig_guidance:
            lh_score, rec_loss_value = self.get_likelihood_score_h_orig(x_den, x_hat, h_orig,i)
        else:
            lh_score = torch.zeros_like(x_hat)
            rec_loss_value = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)
        x_hat_ng = x_hat.detach()
        score = self.Tweedie2score(x_den, x_hat_ng, t_hat)

        if self.args.tester.posterior_sampling.constraint_speech_magnitude.use:
            s_scale = self.args.tester.posterior_sampling.constraint_speech_magnitude.speech_scaling
            x_den = x_den * (s_scale / (x_den.detach().std() + 1e-8))
                

        ode_integrand = self.diff_params._ode_integrand(x_hat_ng, t_hat, score) + lh_score
        dt = t_iplus1 - t_hat

        x_iplus1 = x_hat_ng + dt * ode_integrand
 
        h_tilde = None
        if self.step_counter >= self.warmup_steps:
            h_tilde = self.M_step(x_den.detach())
        
        return x_iplus1.detach_(), x_den.detach(),rec_loss_value,h_tilde
    
    def compute_simple_correlations(self, Xj):
        """
        Compute causal STFT auto/cross-correlations for CTF estimation.

        Preferred input shape:
            Xj:    [B, C, F, T]
            self.Y: [B, C, F, T]

        Legacy input shape [T, F, 1] is still accepted and returns the
        historical shapes.

        Returns:
            batched: Rxx [B, C, F, M, M], rxy [B, C, F, M, 1]
            legacy:  Rxx [F, M, M],       rxy [F, M, 1]
        """
        legacy_input = Xj.ndim == 3

        def to_bcft(z, name):
            if z.ndim == 4:
                return z
            if z.ndim == 3:
                # Legacy layout: [T, F, C] -> [1, C, F, T]
                return z.permute(2, 1, 0).unsqueeze(0)
            raise ValueError(f"Expected {name} with shape [B,C,F,T] or [T,F,C], got {z.shape}")

        X_bcft = to_bcft(Xj, "Xj")
        Y_bcft = to_bcft(self.Y, "self.Y").to(device=X_bcft.device, dtype=X_bcft.dtype)

        if X_bcft.shape[0] != Y_bcft.shape[0]:
            raise ValueError(f"Batch mismatch: Xj has {X_bcft.shape[0]}, self.Y has {Y_bcft.shape[0]}")
        if X_bcft.shape[2:] != Y_bcft.shape[2:]:
            raise ValueError(f"Frequency/time mismatch: Xj has {X_bcft.shape[2:]}, self.Y has {Y_bcft.shape[2:]}")

        if X_bcft.shape[1] != Y_bcft.shape[1]:
            if X_bcft.shape[1] == 1:
                X_bcft = X_bcft.expand(-1, Y_bcft.shape[1], -1, -1)
            elif Y_bcft.shape[1] == 1:
                Y_bcft = Y_bcft.expand(-1, X_bcft.shape[1], -1, -1)
            else:
                raise ValueError(f"Channel mismatch: Xj has {X_bcft.shape[1]}, self.Y has {Y_bcft.shape[1]}")

        M = self.M

        # Build causal history: [B, C, F, T, M]
        X_pad = torch.nn.functional.pad(X_bcft, (M - 1, 0))
        X_hist = X_pad.unfold(-1, M, 1).flip(-1)

        # We apply the CTF as y = sum_m x_m * conj(h_m).  Estimate
        # q = conj(h) from the normal equations (X^H X) q = X^H y.
        Rxx = torch.einsum('bcftm,bcftn->bcfmn', X_hist.conj(), X_hist)
        rxy = torch.einsum('bcftm,bcft->bcfm', X_hist.conj(), Y_bcft).unsqueeze(-1)

        if legacy_input:
            return Rxx.squeeze(0).squeeze(0), rxy.squeeze(0).squeeze(0)

        return Rxx, rxy
    
    def E_step(self, x_i, t_i, gamma_i):
        x_hat, t_hat = self.stochastic_timestep(x_i, t_i, gamma_i)
        x_hat = x_hat.detach().requires_grad_(True)
        
        #E step - Denoise using posterior sampleing
        x_den = self.get_Tweedie_estimate(x_hat, t_hat) #\hat{x_0}
        return x_den,x_hat,t_hat
    


    def M_step_ls(self, X_den_spec, use_running=True):
        Rxx_snapshot, rxy_snapshot = self.compute_simple_correlations(X_den_spec.detach())

        if use_running and float(self.beta) != 0.0:
            beta = float(self.beta)
            if not hasattr(self, "running_Rxx"):
                self.running_Rxx = Rxx_snapshot.detach()
                self.running_rxy = rxy_snapshot.detach()
            else:
                self.running_Rxx = beta * self.running_Rxx + (1.0 - beta) * Rxx_snapshot.detach()
                self.running_rxy = beta * self.running_rxy + (1.0 - beta) * rxy_snapshot.detach()
            Rxx = self.running_Rxx
            rxy = self.running_rxy
        else:
            Rxx = Rxx_snapshot
            rxy = rxy_snapshot

        eye = torch.eye(self.M, device=Rxx.device, dtype=Rxx.dtype)
        diag_power = Rxx.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-8)
        Rxx = Rxx + 1e-4 * diag_power[..., None, None] * eye

        q_column = torch.linalg.solve(Rxx, rxy)
        return q_column.squeeze(-1).conj()

    def ctf_to_time_ir(self, h_tilde):
        h_tilde = h_tilde.detach()
        B, C, _, M = h_tilde.shape
        n_fft = self.stft_options["size"]
        hop = self.stft_options["shift"]
        length = n_fft + (M - 1) * hop
        h_time = torch.zeros(
            B, C, length,
            device=h_tilde.device,
            dtype=h_tilde.real.dtype,
        )

        for m in range(M):
            h_frame = torch.fft.irfft(h_tilde[..., m], n=n_fft, dim=-1)
            start = m * hop
            h_time[..., start:start + n_fft] += h_frame

        return h_time

    def apply_guidance_decay_to_h(self, h_tilde):
        guidance_decay = torch.exp(
            -self.alpha * torch.arange(
                h_tilde.shape[-1],
                device=h_tilde.device,
                dtype=h_tilde.real.dtype,
            )
        )
        return h_tilde * guidance_decay

    def init_h_orig_direct_decay(self, h_like):
        """Initialize H from correct direct binaural cues plus a decaying tail.

        Uses tap 0 of the aligned h_orig CTF as the direct-path cue, which
        carries the oracle ILD/ITD convention. Later CTF taps repeat that cue
        with an exponential decay, giving a reverberant but constrained start.
        """
        if not hasattr(self, "h_orig_ctf"):
            raise RuntimeError("h_orig_ctf is required for h_first_init=h_orig_direct_decay")

        h_ref = self.h_orig_ctf.to(device=h_like.device, dtype=h_like.dtype)
        direct = h_ref[..., :1]
        decay = torch.exp(
            -self.alpha * torch.arange(
                h_like.shape[-1],
                device=h_like.device,
                dtype=h_like.real.dtype,
            )
        )
        return direct * decay.view(*([1] * (direct.ndim - 1)), -1)

    def estimate_observed_itd_samples_gcc_phat(self, y, max_delay_seconds=1e-3):
        """Estimate per-channel delay relative to channel 0 from observed y."""
        if y.ndim != 3:
            raise ValueError(f"Expected observed y with shape [B,C,T], got {y.shape}")

        B, C, T = y.shape
        sample_rate = getattr(self.args.exp, "sample_rate", 16000)
        max_delay = int(round(max_delay_seconds * sample_rate))
        n_fft = 1 << (2 * T - 1).bit_length()
        ref = y[:, :1]
        delays = torch.zeros(B, C, device=y.device, dtype=y.dtype)

        for c in range(1, C):
            sig_fft = torch.fft.rfft(y[:, c:c + 1], n=n_fft, dim=-1)
            ref_fft = torch.fft.rfft(ref, n=n_fft, dim=-1)
            cross = sig_fft * ref_fft.conj()
            cross = cross / cross.abs().clamp_min(1e-8)
            corr = torch.fft.irfft(cross, n=n_fft, dim=-1).squeeze(1)

            lag_ids = torch.cat((
                torch.arange(0, max_delay + 1, device=y.device),
                torch.arange(n_fft - max_delay, n_fft, device=y.device),
            ))
            lags = torch.cat((
                torch.arange(0, max_delay + 1, device=y.device),
                torch.arange(-max_delay, 0, device=y.device),
            )).to(dtype=y.dtype)
            best = corr.index_select(-1, lag_ids).argmax(dim=-1)
            delays[:, c] = lags.index_select(0, best)

        return delays

    def init_observed_y_direct_decay(self, h_like):
        """Initialize H from observed-y ILD/ITD direct cues plus decay."""
        if not hasattr(self, "y"):
            raise RuntimeError("observed y is required for h_first_init=observed_y_direct_decay")

        y = self.y.to(device=h_like.device)
        B, C, _ = y.shape
        _, _, F_bins, M = h_like.shape
        sample_rate = getattr(self.args.exp, "sample_rate", 16000)

        rms = y.pow(2).mean(dim=-1).sqrt()
        gains = rms / rms.mean(dim=1, keepdim=True).clamp_min(1e-8)
        delay_samples = self.estimate_observed_itd_samples_gcc_phat(y)
        delay_seconds = delay_samples / float(sample_rate)

        n_fft = self.stft_options.get("size", 2 * (F_bins - 1))
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).to(
            device=h_like.device,
            dtype=h_like.real.dtype,
        )
        if freqs.numel() != F_bins:
            freqs = torch.linspace(
                0.0,
                sample_rate / 2.0,
                F_bins,
                device=h_like.device,
                dtype=h_like.real.dtype,
            )

        # Internal H is conjugated by apply_ctf_stft, so store the opposite
        # phase sign of the physical delay.
        phase = torch.exp(
            1j
            * 2.0
            * torch.pi
            * delay_seconds[..., None].to(dtype=h_like.real.dtype)
            * freqs.view(1, 1, -1)
        )
        direct = gains[..., None].to(dtype=h_like.real.dtype) * phase
        direct = direct.to(dtype=h_like.dtype)

        decay = torch.exp(
            -self.alpha * torch.arange(M, device=h_like.device, dtype=h_like.real.dtype)
        )
        return direct[..., None] * decay.view(1, 1, 1, M)

    def save_h_tilde_wav(self, h_tilde, suffix, apply_decay=True):
        debug_dir = getattr(self, "h_debug_dir", None)
        if debug_dir is None:
            return

        debug_name = getattr(self, "h_debug_name", "sample")
        h_to_save = h_tilde.detach()
        if apply_decay:
            h_to_save = self.apply_guidance_decay_to_h(h_to_save)
        # apply_ctf_stft uses H.conj() in the forward model, so save the
        # effective audible operator rather than the internal H parameter.
        h_to_save = h_to_save.conj()
        h_time = self.ctf_to_time_ir(h_to_save)
        h_time = h_time / h_time.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        sample_rate = getattr(self.args.exp, "sample_rate", 16000)
        utils_logging.write_audio_file(
            h_time.cpu(),
            sample_rate,
            debug_name + suffix,
            path=debug_dir,
            stereo=h_time.shape[1] == 2,
            normalize=False,
        )

    def save_h_tilde_convolved_x0(self, h_tilde, suffix, apply_decay=True):
        debug_dir = getattr(self, "h_debug_dir", None)
        x0 = getattr(self, "h_debug_x0", None)
        if debug_dir is None or x0 is None:
            return

        if x0.ndim == 1:
            x0 = x0.unsqueeze(0).unsqueeze(0)
        elif x0.ndim == 2:
            x0 = x0.unsqueeze(0)
        x0 = x0.to(device=h_tilde.device, dtype=self.y.dtype)

        h_forward = h_tilde.detach()
        if apply_decay:
            h_forward = self.apply_guidance_decay_to_h(h_forward)
        y_hat_spec = self.apply_ctf_stft(self.stft(x0), h_forward)
        y_hat = self.istft(y_hat_spec, length=x0.shape[-1])

        debug_name = getattr(self, "h_debug_name", "sample")
        sample_rate = getattr(self.args.exp, "sample_rate", 16000)
        utils_logging.write_audio_file(
            y_hat.cpu(),
            sample_rate,
            debug_name + suffix,
            path=debug_dir,
            stereo=y_hat.shape[1] == 2,
            normalize=False,
        )

    def M_step(self,x_den):
        X_den = self.stft(x_den)
        h_new = self.M_step_ls(X_den)

        if not hasattr(self, 'prev_h_tilde'):
            if self.h_first_init == "estimate":
                h_tilde = h_new
            elif self.h_first_init == "h_orig" and hasattr(self, "h_orig_ctf"):
                h_init = self.h_orig_ctf.to(device=h_new.device, dtype=h_new.dtype)
                h_tilde = self.eta * h_init + (1.0 - self.eta) * h_new
            elif self.h_first_init == "h_orig_direct_decay" and hasattr(self, "h_orig_ctf"):
                h_init = self.init_h_orig_direct_decay(h_new)
                h_tilde = self.eta * h_init + (1.0 - self.eta) * h_new
            elif self.h_first_init == "observed_y_direct_decay":
                h_init = self.init_observed_y_direct_decay(h_new)
                h_tilde = self.eta * h_init + (1.0 - self.eta) * h_new
            else:
                raise ValueError(
                    f"Unknown or unavailable h_first_init: {self.h_first_init}. "
                    "Use h_first_init='estimate' or 'observed_y_direct_decay' for blind runs; "
                    "h_orig and h_orig_direct_decay require an informed h_orig_ctf reference."
                )
        else:
            h_tilde = self.eta * self.prev_h_tilde + (1.0 - self.eta) * h_new

        self.prev_h_tilde = h_tilde.detach()

        return h_tilde.to(device=X_den.device)
    
    def apply_ctf_stft(self, X, H):
        """
        Apply a causal CTF in the STFT domain.

        Preferred shapes:
            X: [B, 1, F, T] or [B, C, F, T]
            H: [B, C, F, M]

        Returns:
            Y: [B, C, F, T]

        Legacy shapes are also accepted:
            X: [F, T], H: [F, M] -> Y: [F, T]
        """
        legacy_input = X.ndim == 2 and H.ndim == 2

        if X.ndim == 2:
            X = X.unsqueeze(0).unsqueeze(0)  # [F,T] -> [1,1,F,T]
        elif X.ndim != 4:
            raise ValueError(f"Expected X with shape [B,C,F,T] or [F,T], got {X.shape}")

        if H.ndim == 2:
            H = H.unsqueeze(0).unsqueeze(0)  # [F,M] -> [1,1,F,M]
        elif H.ndim != 4:
            raise ValueError(f"Expected H with shape [B,C,F,M] or [F,M], got {H.shape}")

        H = H.to(device=X.device, dtype=X.dtype)

        if X.shape[0] != H.shape[0]:
            raise ValueError(f"Batch mismatch: X has {X.shape[0]}, H has {H.shape[0]}")
        if X.shape[2] != H.shape[2]:
            raise ValueError(f"Frequency mismatch: X has {X.shape[2]}, H has {H.shape[2]}")

        if X.shape[1] != H.shape[1]:
            if X.shape[1] == 1:
                X = X.expand(-1, H.shape[1], -1, -1)
            elif H.shape[1] == 1:
                H = H.expand(-1, X.shape[1], -1, -1)
            else:
                raise ValueError(f"Channel mismatch: X has {X.shape[1]}, H has {H.shape[1]}")

        M = H.shape[-1]
        X_padded = pad(X, (M - 1, 0))
        X_history = X_padded.unfold(-1, M, 1).flip(-1)  # [B,C,F,T,M]

        Y = torch.einsum('bcftm,bcfm->bcft', X_history, H.conj())

        if legacy_input:
            return Y.squeeze(0).squeeze(0)

        return Y
    
    def get_likelihood_score(self, x_den, x_hat, h_tilde):
        """
        x_den: Model's current prediction of CLEAN speech (connected to grad)
        x_hat: The noisy latent x_t (the tensor we take grad w.r.t)
        h_tilde: The current estimated RIR (should be detached)
        """
        
        X_den_spec = self.stft(x_den)  # [B,1,F,T]

        h_guidance = self.apply_guidance_decay_to_h(h_tilde.detach())

        Y_hat_spec = self.apply_ctf_stft(X_den_spec, h_guidance)  # [B,C,F,T]
        y_hat = self.istft(Y_hat_spec, length=self.y.shape[-1])
  
        loss_terms = []
        grad_terms = []
        sqrt_len = self.args.exp.audio_len**0.5

        if self.lambda_sisdr != 0:
            loss_sisdr = self.rec_loss_sisdr(y_hat, self.y)
            grad_sisdr = torch.autograd.grad(
                outputs=loss_sisdr,
                inputs=x_hat,
                retain_graph=self.lambda_stft != 0,
            )[0]
            grad_sisdr = grad_sisdr / (torch.norm(grad_sisdr) / sqrt_len + 1e-8)
            loss_terms.append(self.lambda_sisdr * loss_sisdr.detach())
            grad_terms.append(self.lambda_sisdr * grad_sisdr)

        if self.lambda_stft != 0:
            loss_stft = self.rec_loss_stft(Y_hat_spec, self.stft(self.y))
            grad_stft = torch.autograd.grad(outputs=loss_stft, inputs=x_hat)[0]
            grad_stft = grad_stft / (torch.norm(grad_stft) / sqrt_len + 1e-8)
            loss_terms.append(self.lambda_stft * loss_stft.detach())
            grad_terms.append(self.lambda_stft * grad_stft)

        if not grad_terms:
            rec_grads = torch.zeros_like(x_hat)
            rec = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)
        else:
            rec_grads = sum(grad_terms)
            rec = sum(loss_terms)

        normguide = torch.norm(rec_grads)/(sqrt_len)
        return self.zeta / (normguide+1e-8) * rec_grads, rec

    def EM_steps(
        self,
        x_i,
        t_i,
        t_iplus1,
        gamma_i,
        update_h=None,
        use_guidance=None,
        fixed_h_tilde=None,
    ):
        x_den,x_hat,t_hat = self.E_step(x_i, t_i, gamma_i)
        x_hat_ng = x_hat.detach()
        score = self.Tweedie2score(x_den, x_hat_ng, t_hat)

        if use_guidance is None:
            use_guidance = self.step_counter >= self.warmup_steps
        if update_h is None:
            update_h = use_guidance and fixed_h_tilde is None

        if use_guidance:
            if fixed_h_tilde is not None:
                h_tilde = fixed_h_tilde.detach()
            elif update_h:
                h_tilde = self.M_step(x_den)
            elif hasattr(self, "prev_h_tilde"):
                h_tilde = self.prev_h_tilde.detach()
            else:
                h_tilde = None

            if h_tilde is not None:
                lh_score, rec_loss_value = self.get_likelihood_score(x_den, x_hat, h_tilde)
            else:
                lh_score = torch.zeros_like(x_hat_ng)
                rec_loss_value = torch.zeros((), device=x_hat_ng.device, dtype=x_hat_ng.dtype)
        else:
            lh_score = torch.zeros_like(x_hat_ng)
            rec_loss_value = torch.zeros((), device=x_hat_ng.device, dtype=x_hat_ng.dtype)

        if self.args.tester.posterior_sampling.constraint_speech_magnitude.use:
            s_scale = self.args.tester.posterior_sampling.constraint_speech_magnitude.speech_scaling
            x_den = x_den * (s_scale / (x_den.detach().std() + 1e-8))
                
        ode_integrand = self.diff_params._ode_integrand(x_hat_ng, t_hat, score) + lh_score
        dt = t_iplus1 - t_hat
        x_iplus1 = x_hat_ng + dt * ode_integrand
        return x_iplus1.detach_(), x_den.detach(),rec_loss_value
    
    def predict_h_tilde(
        self,
        shape, 
        device
        ):
        if self.blind_schedule == "learn_freeze":
            return self.predict_h_tilde_learn_freeze(shape, device)

        self.reset_buffers()
        # get the noise schedule
        t = self.create_schedule().to(device)
        # sample prior
        x = self.initialize_x(shape,device, t)

        gamma = self.get_gamma(t).to(device)
        pbar = tqdm(range(0, self.T, 1))

        for i in pbar:
            self.step_counter=i
            x, x_den,rec_loss_value = self.EM_steps(x, t[i] , t[i+1], gamma[i])
            pbar.set_postfix({
               "rec": f"{rec_loss_value.item():.4f}"})
        if hasattr(self, "prev_h_tilde"):
            self.save_h_tilde_wav(self.prev_h_tilde, "_h_tilde_final")
            self.save_h_tilde_convolved_x0(self.prev_h_tilde, "_y_hat_from_h_tilde_final_x0")
        return x_den

    def predict_h_tilde_learn_freeze(self, shape, device):
        self.reset_buffers()
        t = self.create_schedule().to(device)
        x = self.initialize_x(shape, device, t)
        gamma = self.get_gamma(t).to(device)

        warmup_steps = min(max(int(self.warmup_steps), 0), self.T)

        pbar = tqdm(range(0, warmup_steps), desc="warmup denoise")
        for i in pbar:
            self.step_counter = i
            x, x_den, rec_loss_value = self.EM_steps(
                x, t[i], t[i + 1], gamma[i], update_h=False, use_guidance=False
            )
            pbar.set_postfix({"rec": f"{rec_loss_value.item():.4f}"})

        x_warm = x.detach().clone()

        pbar = tqdm(range(warmup_steps, self.T), desc="learn H")
        for i in pbar:
            self.step_counter = i
            x, x_den, rec_loss_value = self.EM_steps(
                x, t[i], t[i + 1], gamma[i], update_h=True, use_guidance=True
            )
            pbar.set_postfix({"rec": f"{rec_loss_value.item():.4f}"})

        if not hasattr(self, "prev_h_tilde"):
            raise RuntimeError("learn_freeze schedule did not estimate h_tilde")
        frozen_h_tilde = self.prev_h_tilde.detach().clone()
        self.save_h_tilde_wav(frozen_h_tilde, "_h_tilde_learned_before_freeze")
        self.save_h_tilde_convolved_x0(frozen_h_tilde, "_y_hat_from_h_tilde_learned_before_freeze_x0")

        x = x_warm
        pbar = tqdm(range(warmup_steps, self.T), desc="sample frozen H")
        for i in pbar:
            self.step_counter = self.T + i - warmup_steps
            x, x_den, rec_loss_value = self.EM_steps(
                x,
                t[i],
                t[i + 1],
                gamma[i],
                update_h=False,
                use_guidance=True,
                fixed_h_tilde=frozen_h_tilde,
            )
            pbar.set_postfix({"rec": f"{rec_loss_value.item():.4f}"})

        self.prev_h_tilde = frozen_h_tilde
        self.save_h_tilde_wav(frozen_h_tilde, "_h_tilde_final")
        self.save_h_tilde_convolved_x0(frozen_h_tilde, "_y_hat_from_h_tilde_final_x0")
        return x_den
    
    def predict_unconditional(self,
        y,  #observations 
        shape=None,
        h_init_time=None,
        **kwargs):
        if len(y.shape)==2:
            y = y.unsqueeze(0)
        self.y = y
        self.Y = self.stft(self.y)
        if h_init_time is None:
            h_init_time = kwargs.get("hrtf", None)
        self.h_init_time = h_init_time
        self.h_debug_dir = kwargs.get("h_debug_dir", None)
        self.h_debug_name = kwargs.get("h_debug_name", "sample")
        self.h_orig_ctf_x_ref = kwargs.get("h_orig_ctf_x_ref", None)
        self.h_debug_x0 = kwargs.get("h_debug_x0", self.h_orig_ctf_x_ref)
        h_orig = kwargs.get("h_orig", None)
        if h_orig is not None:
            h_orig = h_orig.to(device=y.device, dtype=y.dtype)
            h_ref_signal = self.h_orig_ctf_x_ref
            if h_ref_signal is not None:
                self.h_orig_ctf = self.estimate_ctf_from_pair(
                    h_ref_signal.to(device=y.device, dtype=y.dtype),
                    self.y,
                ).to(device=y.device, dtype=self.Y.dtype)
            else:
                self.h_orig_ctf = self.brir_time_to_ctf_probe_ls(
                    h_orig,
                    length=self.y.shape[-1],
                    M=self.M,
                ).to(device=y.device, dtype=self.Y.dtype)
        self.saved_h_tilde_0 = False
        if shape is None:
            shape = (y.shape[0], 1, y.shape[-1])
        x_den = self.predict_h_tilde(shape, y.device)

        self.h_init_time = None
        self.h_debug_dir = None
        self.h_debug_name = None
        self.h_orig_ctf_x_ref = None
        self.h_debug_x0 = None
        if hasattr(self, "h_orig_ctf"):
            del self.h_orig_ctf
        return x_den
    
    def predict_h_orig(
        self,
        shape, 
        device,
        h_orig
        ):
        self.reset_buffers()
        # get the noise schedule
        t = self.create_schedule().to(device)
        # sample prior
        x = self.initialize_x(shape,device, t)

        h_ref_signal = getattr(self, "h_orig_ctf_x_ref", None)
        if h_ref_signal is not None:
            self.h_orig_ctf = self.estimate_ctf_from_pair(
                h_ref_signal.to(device=device, dtype=self.y.dtype),
                self.y,
            ).to(device=device, dtype=self.Y.dtype)
        else:
            self.h_orig_ctf = self.brir_time_to_ctf_probe_ls(
                h_orig,
                length=self.y.shape[-1],
                M=self.M,
            ).to(device=device, dtype=self.Y.dtype)
        self.h_compare_records = []

        gamma = self.get_gamma(t).to(device)
        pbar = tqdm(range(0, self.T, 1))

        for i in pbar:
            self.step_counter=i
            x, x_den,rec_loss_value,h_tilde = self.stepEM_h_orig(x, t[i] , t[i+1], gamma[i],i,h_orig)
            if h_tilde is not None:
                self.record_h_tilde_h_orig_comparison(h_tilde, i)
            pbar.set_postfix({
               "rec": f"{rec_loss_value.item():.4f}"})

        self.save_h_tilde_h_orig_comparison()
        if hasattr(self, "prev_h_tilde"):
            self.save_h_tilde_wav(self.prev_h_tilde, "_h_tilde_final")
            self.save_h_tilde_wav(self.prev_h_tilde, "_h_tilde_final_raw", apply_decay=False)
            self.save_h_tilde_convolved_x0(self.prev_h_tilde, "_y_hat_from_h_tilde_final_x0")
            self.save_h_tilde_convolved_x0(
                self.prev_h_tilde,
                "_y_hat_from_h_tilde_final_x0_raw",
                apply_decay=False,
            )
        if hasattr(self, "h_orig_ctf"):
            self.save_h_tilde_wav(self.h_orig_ctf, "_h_orig_ctf")
            self.save_h_tilde_wav(self.h_orig_ctf, "_h_orig_ctf_raw", apply_decay=False)
        return x_den

    def predict_conditional(
        self,
        y,  #observations 
        h_orig=None,
        shape=None,
        blind=False,
        **kwargs
    ):  
        if len(y.shape)==2:
            y = y.unsqueeze(0)
        self.y = y
        self.h_debug_dir = kwargs.get("h_debug_dir", None)
        self.h_debug_name = kwargs.get("h_debug_name", "sample")
        self.h_orig_ctf_x_ref = kwargs.get("h_orig_ctf_x_ref", None)
        self.h_debug_x0 = kwargs.get("h_debug_x0", self.h_orig_ctf_x_ref)
        self.saved_h_tilde_0 = False
        if shape is None:
            # shape = y.shape
            shape = (y.shape[0], 1, y.shape[-1])

        x_den = self.predict_h_orig(shape, y.device, h_orig)

        self.h_debug_dir = None
        self.h_debug_name = None
        self.h_orig_ctf_x_ref = None
        self.h_debug_x0 = None
        if hasattr(self, "h_orig_ctf"):
            del self.h_orig_ctf
        return x_den

    
class CompressedSTFTLoss(nn.Module):
    def __init__(self, compression_factor=2/3, weight=512, summean=True):
        super().__init__()
        self.alpha = compression_factor
        self.weight = weight
        self.summean = summean

    def forward(self, Y_hat, Y):
        # Y_hat, Y: [..., F, T], complex

        Y_hat_comp = (Y_hat.abs() + 1e-8).pow(self.alpha) * torch.exp(1j * Y_hat.angle())
        Y_comp = (Y.abs() + 1e-8).pow(self.alpha) * torch.exp(1j * Y.angle())

        diff2 = (Y_comp - Y_hat_comp).abs().pow(2)  # [..., F, T]

        if self.summean:
            loss = torch.mean(torch.sum(diff2, dim=-2))  # sum F, mean rest
        else:
            loss = torch.mean(diff2)

        return self.weight * loss
    
def _sisdr_time_safe(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8, return_db: bool = True):
    """
    est, ref: [B, C, L] real tensors (time-domain)
    SI-SDR = 10*log10( ||s_target||^2 / ||e_noise||^2 ), with zero-mean per sample.
    Numerically safe (no div by zero, no log of zero).
    Returns [B, C] (dB if return_db).
    """
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    ref_energy = (ref**2).sum(dim=-1, keepdim=True).clamp_min(eps)  # [B,C,1]
    scale = (est * ref).sum(dim=-1, keepdim=True) / ref_energy      # [B,C,1]
    s_target = scale * ref                                          # [B,C,L]

    e_noise = est - s_target

    num = (s_target**2).sum(dim=-1).clamp_min(eps)  # [B,C]
    den = (e_noise**2).sum(dim=-1).clamp_min(eps)   # [B,C]
    ratio = (num / den) #.clamp(1e-4, 1e4)
    if return_db:
        return 10.0 * torch.log10(ratio)
    else:
        return ratio

class SiSDRLoss(nn.Module):
    """
    SI-SDR loss for batched 2-channel audio given complex STFTs.
    est_stft, ref_stft: [B, 2, F, T], complex dtype
    Returns scalar loss = -mean(SI-SDR_dB over batch & channels).
    """
    def __init__(self, eps=1e-8, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, est_time: torch.Tensor, ref_time: torch.Tensor) -> torch.Tensor:
   
        B, C, _ = ref_time.shape
        target_len = ref_time.size(-1)
        # Reshape back to [B, C, L] and keep in float32
        ref_time = ref_time.view(B, C, -1).to(torch.float32)
        est_time = est_time.view(B, C, -1).to(torch.float32)

        # Extra guard: if any sample is entirely zero, add a tiny dither (prevents 0/0)
        silent_ref = (ref_time.abs().sum(dim=-1, keepdim=True) == 0)
        if silent_ref.any():
            ref_time = ref_time + silent_ref * (self.eps * torch.randn_like(ref_time))

        sisdr_bc = _sisdr_time_safe(est_time, ref_time, eps=self.eps, return_db=True)  # [B, C]
        loss_per = -sisdr_bc  # maximize SI-SDR

        if self.reduction == 'mean':
            return loss_per.mean()
        elif self.reduction == 'sum':
            return loss_per.sum()
        else:
            return loss_per
