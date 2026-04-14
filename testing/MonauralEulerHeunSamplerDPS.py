from tqdm import tqdm
import torch
from nara_wpe.wpe import wpe
import math
import torch.nn as nn
from testing.EulerHeunSampler import EulerHeunSampler
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.nn.functional import pad

class MonauralEulerHeunSamplerDPS(EulerHeunSampler):
    """
        Euler Heun sampler for DPS 
        inverse problem solver
    """

    def __init__(self, model, diff_params, args):
        super().__init__(model, diff_params, args)
        self.zeta = self.args.tester.posterior_sampling.zeta
        self.stft_options = dict(size=512, shift=128)
        self.rec_loss_sisdr = SiSDRLoss() 
        self.rec_loss_stft = CompressedSTFTLoss()
        self.warmup_steps = self.args.tester.sampling_params.warmup_steps
        self.M = self.args.tester.sampling_params.M
        self.alpha = self.args.tester.sampling_params.alpha
        self.beta_min = self.args.tester.sampling_params.beta_min
        self.beta_max = self.args.tester.sampling_params.beta_max
        self.lambda_stft = self.args.tester.sampling_params.lambda_stft
        self.order = 1
        self.eta = self.args.tester.sampling_params.eta
        self.lambda_h = self.args.tester.sampling_params.lambda_h
        self.N=3

    def initialize_x(self, shape, device, schedule):
        Y = self.stft(self.y)
        self.Y = Y.permute(2,1,0)
        if self.args.tester.posterior_sampling.warm_initialization.mode == "none":
            x = schedule[0]*torch.randn(shape).to(device)

        elif self.args.tester.posterior_sampling.warm_initialization.mode == "reverb_scaled":
            x = self.args.tester.posterior_sampling.warm_initialization.scaling_factor * self.y.clone() / self.y.std() + schedule[0] * torch.randn(shape).to(device)
        
        elif self.args.tester.posterior_sampling.warm_initialization.mode == "wpe_scaled":
            print("Processing WPE")

            delay = self.args.tester.posterior_sampling.warm_initialization.wpe.delay
            iterations = self.args.tester.posterior_sampling.warm_initialization.wpe.iterations
            taps = self.args.tester.posterior_sampling.warm_initialization.wpe.taps
            
            Y = Y.cpu().numpy().transpose(2, 0, 1)
            Z =  wpe(
                Y,
                taps=taps,
                delay=delay,
                iterations=iterations,
                statistics_mode='full'
            )

            Z = Z.transpose(1, 2, 0)

            x_pred = self.istft(torch.from_numpy(Z)).to(self.y.device).type(self.y.dtype)
            if x_pred.shape[-1] > self.y.shape[-1]:
                x_pred = x_pred[..., :self.y.shape[-1]]

            x_pred = self.args.tester.posterior_sampling.warm_initialization.scaling_factor * x_pred / x_pred.std()
            x = x_pred + schedule[0] * torch.randn(shape).to(device)

        else:
            raise NotImplementedError
        
        return x
    
    def reset_buffers(self):
        if hasattr(self, 'running_Rxx'):
            del self.running_Rxx
        if hasattr(self, 'running_rxy'):
            del self.running_rxy
        if hasattr(self, '_warmup_count'):
            del self._warmup_count
        if hasattr(self, '_ema_initialized'):
            del self._ema_initialized
        torch.cuda.empty_cache()

    def stft(self, time_signal):
        """
        Args:
            time_signal: [J, Time_Samples] - Binaural waveform (e.g., [2, 131840])
        Returns:
            stft_signal: [J, F, T] - Complex STFT (e.g., [2, 257, 1031])
        """
        # 1. Extract Options
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
        
        return stft_signal
    
    def istft(self, stft_signal, length=None):
        """
        Args:
            stft_signal: [J, F, T] - Complex STFT
            length: Optional original length to trim padding exactly
        Returns:
            time_signal: [J, Time_Samples] - Reconstructed waveform
        """
        # 1. Extract Options
        n_fft = self.stft_options['size']
        hop_length = self.stft_options['shift']
        win_length = n_fft
        
        device = stft_signal.device
        window = torch.hann_window(win_length, periodic=False).to(device)

        # 2. Compute iSTFT
        # PyTorch natively handles the batch (J) dimension
        time_signal = torch.istft(
            stft_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=length, # Trims the 'fading' padding automatically
            return_complex=False
        )
        
        return time_signal
    


    def get_likelihood_score(self, X_den, x, h_tilde,i):
        Y_hat = self.apply_ctf_stft(X_den.permute(1,0,2).squeeze(),h_tilde)
        y_hat = self.istft(Y_hat)
        rec = self.rec_loss_sisdr(y_hat,self.y)
        rec = self.lambda_stft*self.rec_loss_stft(Y_hat,self.Y.permute(2,1,0))
        rec_grads = torch.autograd.grad(outputs=rec, inputs=x)[0]

        # Normalize weighting parameter zeta
        normguide = torch.norm(rec_grads)/(self.args.exp.audio_len**0.5)
        # if i < self.warmup_steps:
        #     zeta_i = 0.25 * self.zeta
        # else:
        #     zeta_i = self.zeta
        return self.zeta / (normguide+1e-8) * rec_grads, rec
    

    def generate_sweep_pair(self,sr=16000, duration=8.192, f1=62.5):
        """
        Generates the synchronized Sine Sweep and Inverse Filter pair.
        """
        f2 = sr / 2
        num_sample = int(duration * sr)
        taxis = np.arange(0, num_sample, 1) / (num_sample - 1)

        w1 = 2 * np.pi * f1 / sr
        w2 = 2 * np.pi * f2 / sr
        lw = np.log(w2 / w1)

        # Generate Sweep
        sweep = np.sin(w1 * (num_sample - 1) / lw * (np.exp(taxis * lw) - 1))

        # Generate Inverse Filter
        envelope = (w2 / w1) ** (-taxis)
        inv = np.flip(sweep) * envelope
        scaling = np.pi * num_sample * (w1 / w2 - 1) / (2 * (w2 - w1) * np.log(w1 / w2))
        inv = inv / scaling

        # Apply Hanning Ramps
        def ramp(x, l=256, r=128):
            x[:l] *= np.hanning(l * 2)[:l]
            x[-r:] *= np.hanning(r * 2)[r:]
            return x

        sweep = ramp(sweep)
        
        # Convert to Torch and Pad
        sweep_t = pad(torch.from_numpy(sweep).float(), (512, 512))
        inv_t = pad(torch.from_numpy(inv).float(), (512, 512))
        
        return sweep_t, inv_t
    
    def apply_ctf_stft(self,X, H):
        """
        Applies a CTF filter to a signal where both are already in the STFT domain.
        
        Args:
            X (Tensor): [F, T] complex - The STFT of the dry signal.
            H (Tensor): [F, L] complex - The CTF coefficients (room filter).
            
        Returns:
            Tensor: [F, T] complex - The filtered reverberant spectrogram.
        """
        F, T = X.shape
        L = H.shape[-1]
        
        # 1. Causal Padding on the Time dimension (dim 1)
        # We pad the "past" by L-1 frames to ensure the convolution 
        # starts correctly at t=0.
        X_padded = pad(X, (L - 1, 0)) # [F, T + L - 1]
        
        # 2. The Unfold Trick
        # Creates [F, T, L] where each [f, t] is a vector of the last L frames
        X_history = X_padded.unfold(1, L, 1) # [F, T, L]
        
        # If your CTF is stored in standard FIR order [h0, h1, ... hL-1],
        # we flip the history to match the dot product logic of convolution.
        X_history = X_history.flip(-1) 
        
        # 3. Complex Matmul (Batch Matrix Multiplication)
        # H: [F, L] -> [F, L, 1] to treat as a column vector for each freq bin
        H_vec = H.unsqueeze(2)
        
        # [F, T, L] @ [F, L, 1] -> [F, T, 1]
        Y = torch.matmul(X_history, H_vec)
        
        return Y.squeeze(-1)
    
    def get_rir_from_ctf(self,ctf, sr=16000):
        """
        Standalone function to extract a time-domain RIR from CTF coefficients.
        
        Args:
            ctf (Tensor): [F, L] complex tensor (Estimated CTF coefficients)
            sinesweep (Tensor): [Time] pre-generated sine sweep signal
            invfilter (Tensor): [Time] pre-generated inverse filter signal
            TF: Transformation object (must have .stft and .istft methods)
            sr (int): Sampling rate (default 16000)
            
        Returns:
            Tensor: [Time] The estimated time-domain Room Impulse Response
        """

        device = ctf.device
        L = ctf.shape[-1]
        
        # 1. Prepare Probe Signal
        # Convert sine sweep to STFT domain
        sinesweep,invfilter = self.generate_sweep_pair()
        sinesweep = sinesweep.to(device)
        sinesweep_spec = self.stft(sinesweep) # [F, T]

        # 2. Convolve CTF with Sine Sweep in STFT Domain
        ir_spec = self.apply_ctf_stft(sinesweep_spec,ctf)
        
        # 3. Transform back to Time Domain
        ir_time = self.istft(ir_spec) # [Time]

        # 4. Deconvolution via Inverse Filtering
        # Convolution with inverse filter collapses the sine sweep into an impulse
        invfilter = invfilter.to(device)
        rir = torchaudio.functional.convolve(invfilter, ir_time, mode="full")

        # 5. Alignment and Post-processing
        # Find peak (direct path) and trim leading zeros
        # Starts 2.5ms before the peak
        peak_idx = torch.argmax(rir.abs())
        start_offset = int(sr * 0.0025)
        
        rir = rir[max(0, peak_idx - start_offset) :]
        
        # Peak Normalization
        max_val = rir.abs().max()
        if max_val > 1.0:
            rir = rir / max_val

        # Return first 2 seconds
        return rir[:sr * 2]

    def compute_simple_correlations(self, Xj):
        """
        Xj: [T, F, 1]
        self.Y: [T, F, 1]

        Returns:
            Rxx: [F, M, M]
            rxy: [F, 1, M]
        """
        # [T, F, 1] -> [F, T]
        X_ft = Xj.squeeze(-1).transpose(0, 1)
        Y_ft = self.Y.squeeze(-1).transpose(0, 1)

        F_bins, T = X_ft.shape
        M = self.M

        # Build causal history: [F, T, M]
        X_pad = torch.nn.functional.pad(X_ft, (M - 1, 0))
        X_hist = X_pad.unfold(1, M, 1).flip(-1)

        # Rxx = sum_t x_t x_t^H
        Rxx = torch.einsum('ftm,ftn->fmn', X_hist, X_hist.conj())

        # rxy = sum_t y_t x_t^H
        rxy = torch.einsum('ft,ftm->fm', Y_ft, X_hist.conj()).unsqueeze(1)

        return Rxx, rxy
    
    def stepEM(self, x_i, t_i, t_iplus1, gamma_i,i, eps=1e-5):
        x_hat, t_hat = self.stochastic_timestep(x_i, t_i, gamma_i)
        x_hat = x_hat.detach().requires_grad_(True)
        
        #E step - Denoise using posterior sampleing
        x_den = self.get_Tweedie_estimate(x_hat, t_hat) #\hat{x_0}
        X_den = self.stft(x_den).permute(2, 1, 0)

        if i < self.warmup_steps:
            h_tilde = torch.zeros((X_den.shape[1], self.M), dtype=X_den.dtype, device=X_den.device)
            h_tilde[:, 0] = 1.0
        else:
            eye = torch.eye(self.M, device=X_den.device, dtype=X_den.dtype).unsqueeze(0)

            Rxx_snapshot, rxy_snapshot = self.compute_simple_correlations(X_den)

            
            Rxx_stable = Rxx_snapshot + (eps + self.lambda_h) * eye

            h_column = torch.linalg.solve(Rxx_stable, rxy_snapshot.conj().transpose(1, 2))
            h_new = h_column.transpose(1, 2).squeeze(1)

            if not hasattr(self, 'prev_h_tilde'):
                h_tilde = h_new
            else:
                eta = self.eta if i%self.M ==0 else 1
                h_tilde = eta * self.prev_h_tilde + (1.0 - eta) * h_new
            # else:
            #     h_tilde = self.prev_h_tilde
            self.prev_h_tilde = h_tilde.detach()

        decay = torch.exp(-self.alpha * torch.arange(self.M).to(h_tilde.device)).view(1, self.M)
        h_tilde = (h_tilde * decay).to(device=X_den.device)
        
   
        if self.args.tester.posterior_sampling.constraint_speech_magnitude.use:
            x_den = self.args.tester.posterior_sampling.constraint_speech_magnitude.speech_scaling / x_den.detach().std() * x_den #Match the sigma_data of dataset

        lh_score, rec_loss_value = self.get_likelihood_score(X_den, x_hat, h_tilde,i)
        x_hat_ng = x_hat.detach()
        score = self.Tweedie2score(x_den, x_hat_ng, t_hat)

        ode_integrand = self.diff_params._ode_integrand(x_hat_ng, t_hat, score) + lh_score
        dt = t_iplus1 - t_hat

        if t_iplus1 !=0 and self.order == 2: #second order correction
            t_prime = t_iplus1
            x_prime = x_hat + dt * ode_integrand
            x_prime.requires_grad_(True)
            x_den = self.get_Tweedie_estimate(x_prime, t_prime)
            X_den = self.stft(x_den).permute(2, 1, 0)

            lh_score_next, rec_loss_value = self.get_likelihood_score(X_den, x_prime, h_tilde,i)
            x_prime.detach_()

            score = self.Tweedie2score(x_den, x_prime, t_prime)

            ode_integrand_next = self.diff_params._ode_integrand(x_prime, t_prime, score) + lh_score_next
            ode_integrand_midpoint = .5 * (ode_integrand + ode_integrand_next)
            x_iplus1 = x_hat + dt * ode_integrand_midpoint
        else:
            x_iplus1 = x_hat_ng + dt * ode_integrand

        return x_iplus1.detach_(), x_den.detach(),rec_loss_value,h_tilde.detach()

    def predict(
        self,
        shape, 
        device,
        blind=False
    ):
        self.reset_buffers()
        # get the noise schedule
        t = self.create_schedule().to(device)

        # sample prior
        x = self.initialize_x(shape,device, t)

        # parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma = self.get_gamma(t).to(device)
        pbar = tqdm(range(0, self.T, 1))

        for i in pbar:
            self.step_counter=i
            x, x_den,rec_loss_value,h_tilde = self.stepEM(x, t[i] , t[i+1], gamma[i],i)
            pbar.set_postfix({
               "rec": f"{rec_loss_value.item():.4f}"})
        h = self.get_rir_from_ctf(h_tilde)

        return x_den,h

    
    def predict_unconditional(self, *args, **kwargs):
        raise ValueError("DPS not made for unconditional sampling")

    def predict_conditional(
        self,
        y,  #observations 
        shape=None,
        blind=False,
        **kwargs
    ):

        self.y = y

        if shape is None:
            shape = y.shape
        return self.predict(shape, y.device, blind)


class CompressedSTFTLoss(nn.Module):
    def __init__(self, compression_factor=2/3):
        super().__init__()
        self.alpha = compression_factor

    def forward(self, Y_hat, Y, freq_weight=None):
        if freq_weight is not None:
            Y_hat = Y_hat * freq_weight
            Y = Y * freq_weight

        S_hat = (Y_hat.abs() + 1e-8).pow(self.alpha) * torch.exp(1j * Y_hat.angle())
        S_y = (Y.abs() + 1e-8).pow(self.alpha) * torch.exp(1j * Y.angle())

        return ((S_y - S_hat).abs() ** 2).mean()

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
   
        B=1
        C,_ =ref_time.shape
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
        

def chirp_log_torch(t, f0, f1, t1, device=None):
    """
    PyTorch implementation of scipy.signal.chirp(method='logarithmic')
    
    Args:
        t (Tensor): Time vector (seconds)
        f0 (float): Frequency at t=0 (Hz)
        f1 (float): Frequency at t=t1 (Hz)
        t1 (float): Time at which f1 is reached (seconds)
    """
    if device is None:
        device = t.device
        
    # k = (f1/f0)^(1/t1)
    k = (f1 / f0) ** (1 / t1)
    
    # Phase = 2 * pi * f0 * (k^t - 1) / ln(k)
    # This is the integral of f(t) = f0 * k^t
    phi = 2 * torch.pi * f0 * (k**t - 1) / torch.log(torch.tensor(k, device=device))
    
    return torch.sin(phi)